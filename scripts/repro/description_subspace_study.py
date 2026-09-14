#!/usr/bin/env python3
"""Stage the class-description subspace residual without tuning on Novel images."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DATASETS = {
    'dtd': {'label': 'DTD', 'name': 'DescribableTextures', 'config': 'dtd.yaml'},
    'ucf101': {'label': 'UCF101', 'name': 'UCF101', 'config': 'ucf101.yaml'},
}
BRANCHES = ('S', 'M', 'R', 'Z', 'F')
REQUIRED_ARRAYS = {'labels', *BRANCHES}
CLIP_SHA256 = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
LEGACY_CHECKPOINT_SHA256 = 'fb42b9b5e6ab336d0c951837e74352069837fd3fd10da9911474061b9f704be9'
LEGACY_IDENTITY = {
    'version': 4,
    'kind': 'fusion_aligned_visual_text_detail',
    'inference': 'F+R',
    'objective': 'CE(S,y)+CE(stopgrad(F)+R,y)',
    'fusion_anchor': 'F=0.5*(S+M)',
    'coefficient': 1.0,
}
FORMULA = {
    'centered_descriptions': 'E_cv=d_cv-mean_valid(d_cv)',
    'description_subspace': 'Q_c=orthonormal_row_basis(E_c)',
    'rank_rule': 'sigma>max(rows,dimensions)*float32_epsilon*sigma_max',
    'projected_delta': 'delta_parallel=Q_c.T@Q_c@delta',
    'description_radius': 'r_c=sqrt(mean_valid(||d_cv-mean_valid(d_cv)||^2))',
    'bounded_delta': 'delta_hat=r_c*tanh(||delta_parallel||/r_c)*delta_parallel/(||delta_parallel||+epsilon)',
    'simple_fusion': 'F=0.5*(S+M)',
    'corrected': 'Z=F+R_subspace_radius',
    'objective': 'CE(S,y)+CE(stopgrad(F)+R_subspace_radius,y)',
    'operation_order': 'project_then_radius_bound',
    'fitted_hyperparameters': [],
}
DEPENDENCIES = (
    'train.py',
    'trainers/coop_text_detail.py',
    'trainers/coop_text_residual.py',
    'trainers/coop_text_residual_adamw.py',
    'trainers/coop_textconsistency_correction.py',
    'trainers/coop_textconsistency_v2_fg.py',
    'trainers/textconsistency_common.py',
    'trainers/textconsistency_runtime.py',
    'datasets/dtd.py',
    'datasets/ucf101.py',
    'configs/datasets/dtd.yaml',
    'configs/datasets/ucf101.yaml',
    'configs/repro/text_detail/subspace_radius_attentive.yaml',
    'templates/texture_prompts_full.json',
    'templates/ucf_prompts_full.json',
    'scripts/repro/description_subspace_study.py',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def harmonic_mean(base, novel):
    return float(2. * base * novel / (base + novel)) if base + novel else 0.


def zero_training_decision(hm_gain, novel_gain):
    values = [float(hm_gain), float(novel_gain)]
    if not all(np.isfinite(values)):
        raise ValueError('Zero-training decision requires finite gains')
    criteria = {
        'HM增益为正': values[0] > 0.,
        'Novel下降不超过1个百分点': values[1] >= -1.,
    }
    return {'passes': all(criteria.values()), 'criteria': criteria,
            'HM增益': values[0], 'Novel增益': values[1]}


def confirmation_decision(hm_gains, novel_mean_gain):
    gains = [float(value) for value in hm_gains]
    novel = float(novel_mean_gain)
    if len(gains) != 3 or not all(np.isfinite([*gains, novel])):
        raise ValueError('Confirmation requires three finite HM gains and one Novel gain')
    positive = sum(value > 0. for value in gains)
    criteria = {
        '平均HM增益为正': float(np.mean(gains)) > 0.,
        '至少两个种子HM增益为正': positive >= 2,
        '平均Novel下降不超过1个百分点': novel >= -1.,
    }
    return {'passes': all(criteria.values()), 'criteria': criteria,
            'hm_gains': gains, 'mean_hm_gain': float(np.mean(gains)),
            'novel_mean_gain': novel, 'positive_seeds': positive}


def validate_legacy_checkpoint(checkpoint, observed_sha256):
    """Authenticate the one preregistered legacy checkpoint before reinterpretation."""
    if observed_sha256 != LEGACY_CHECKPOINT_SHA256:
        raise ValueError('Unexpected legacy checkpoint SHA256')
    signature = checkpoint.get('repro_signature', {})
    correction = signature.get('correction', {})
    provenance = checkpoint.get('run_provenance', {})
    identity_matches = (
        checkpoint.get('epoch') == 20
        and signature.get('trainer') == 'CoOp_TextDetail_FusionAligned'
        and all(correction.get(key) == value for key, value in LEGACY_IDENTITY.items())
        and provenance.get('seed') == 1
        and provenance.get('dataset') == 'DescribableTextures'
        and provenance.get('shots') == 16
        and provenance.get('subsample') == 'base'
        and provenance.get('trainable_parameters') == 67584
        and provenance.get('model_signature') == signature
    )
    if not identity_matches:
        raise ValueError('Unexpected legacy checkpoint identity')
    return {
        'checkpoint_sha256': observed_sha256,
        'epoch': 20,
        'source_trainer': signature['trainer'],
        'source_correction': dict(correction),
        'reinterpretation': 'same learned tensors; apply frozen class-description subspace and radius only',
        'target_operation': 'description_subspace_then_radius',
    }


def planned_cases():
    phases = (
        ('DTD单种子训练筛选', 'dtd', (1,)),
        ('DTD三种子开发确认', 'dtd', (2, 3)),
        ('UCF101单种子独立筛选', 'ucf101', (1,)),
        ('UCF101三种子独立确认', 'ucf101', (2, 3)),
    )
    return [{'phase': phase, 'dataset': dataset, 'seed': seed}
            for phase, dataset, seeds in phases for seed in seeds]


def dependency_hashes():
    missing = [name for name in DEPENDENCIES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing protocol dependency: {missing[0]}')
    return {name: sha256_file(ROOT / name) for name in DEPENDENCIES}


def build_protocol(data_root, clip_path, legacy_model_dir, output_root, repository_commit):
    return {
        'protocol': '类别描述子空间约束残差-v1',
        'created_before_new_model_outputs': True,
        'repository_commit': repository_commit,
        'data_root': str(Path(data_root)),
        'clip_path': str(Path(clip_path)),
        'clip_sha256': CLIP_SHA256,
        'legacy_model_dir': str(Path(legacy_model_dir)),
        'legacy_checkpoint_sha256': LEGACY_CHECKPOINT_SHA256,
        'output_root': str(Path(output_root)),
        'formula': FORMULA,
        'training': {'shots': 16, 'epochs': 20, 'seeds': [1, 2, 3],
                     'description_count': 20, 'pooling': 'attentive', 'backbone': 'ViT-B/16'},
        'cases': planned_cases(),
        'maximum_new_trainings': 6,
        'stopping_rule': {
            'legacy_zero_training_reload': 'stop unless Z>F in HM and Novel gain >= -1pp',
            'single_seed': 'stop unless Z>F in HM and Novel gain >= -1pp',
            'three_seed': 'stop unless mean HM gain >0, at least 2/3 gains >0, and mean Novel gain >= -1pp',
        },
        'evidence_policy': {
            'DTD': 'development only because it informed the formula',
            'UCF101': 'independent confirmation; formula and gates frozen before new output',
            'Aircraft': 'historical post-hoc evidence only',
        },
        'files_sha256': dependency_hashes(),
    }


def verify_protocol(protocol):
    if (protocol.get('protocol') != '类别描述子空间约束残差-v1'
            or protocol.get('formula') != FORMULA
            or protocol.get('clip_sha256') != CLIP_SHA256
            or protocol.get('legacy_checkpoint_sha256') != LEGACY_CHECKPOINT_SHA256
            or protocol.get('cases') != planned_cases()
            or protocol.get('maximum_new_trainings') != 6):
        raise ValueError('Unknown or modified description-subspace protocol')
    observed = dependency_hashes()
    if protocol.get('files_sha256') != observed:
        changed = sorted(name for name in set(observed) | set(protocol.get('files_sha256', {}))
                         if observed.get(name) != protocol.get('files_sha256', {}).get(name))
        raise ValueError(f'Frozen source/configuration changed: {changed[:3]}')


def train_command(case, data_root, clip_path, model_dir):
    dataset = DATASETS[case['dataset']]
    return [
        sys.executable, '-u', str(ROOT / 'train.py'),
        '--root', str(data_root), '--seed', str(case['seed']),
        '--trainer', 'CoOp_TextDetail_SubspaceRadius',
        '--dataset-config-file', str(ROOT / 'configs/datasets' / dataset['config']),
        '--config-file', str(ROOT / 'configs/repro/text_detail/subspace_radius_attentive.yaml'),
        '--output-dir', str(model_dir),
        'DATASET.SUBSAMPLE_CLASSES', 'base', 'USE_CUDA', 'True',
        'TRAINER.TEXTCONSISTENCY.DEVICE', 'cuda',
        'TRAINER.TEXTCONSISTENCY.CLIP_PATH', str(clip_path), 'TEST.NO_TEST', 'True',
    ]


def collect_predictions(model, loader, device):
    import torch

    arrays = {key: [] for key in ('labels', *BRANCHES)}
    paths = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            outputs = model(batch['img'].to(device))
            labels = batch['label'].cpu().numpy()
            arrays['labels'].append(labels)
            for key in BRANCHES:
                value = outputs[key].cpu().numpy()
                if value.ndim != 2 or len(value) != len(labels) or not np.isfinite(value).all():
                    raise ValueError(f'Malformed {key} logits during export')
                arrays[key].append(value)
            paths.extend(str(path) for path in batch['impath'])
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Export has empty or duplicate sample paths')
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    np.testing.assert_array_equal(arrays['F'], .5 * arrays['S'] + .5 * arrays['M'])
    np.testing.assert_array_equal(arrays['Z'], arrays['F'] + arrays['R'])
    return arrays, paths


def prepare_export_directory(output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    if not output_dir.is_dir() or {path.name for path in output_dir.iterdir()} - {'运行信息'}:
        raise ValueError(f'Unexpected files appeared in fresh export directory: {output_dir}')


def _build_export_trainer(dataset_key, seed, model_dir, split, data_root, clip_path,
                          output_dir, legacy):
    import torch
    from dassl.config import get_cfg_default
    from dassl.engine import build_trainer
    from dassl.utils import set_random_seed
    import train

    cfg = get_cfg_default(); train.extend_cfg(cfg)
    cfg.merge_from_file(str(Path(model_dir) / 'effective_config.yaml'))
    expected = DATASETS[dataset_key]
    expected_trainer = 'CoOp_TextDetail_FusionAligned' if legacy else 'CoOp_TextDetail_SubspaceRadius'
    if (cfg.TRAINER.NAME != expected_trainer or cfg.DATASET.NAME != expected['name']
            or cfg.DATASET.NUM_SHOTS != 16 or cfg.OPTIM.MAX_EPOCH != 20
            or cfg.SEED != seed or cfg.DATASET.SUBSAMPLE_CLASSES != 'base'):
        raise ValueError('Checkpoint effective configuration differs from the fixed protocol')
    cfg.TRAINER.NAME = 'CoOp_TextDetail_SubspaceRadius'
    cfg.DATASET.ROOT = str(Path(data_root).resolve())
    cfg.DATASET.SUBSAMPLE_CLASSES = 'new' if split == 'novel' else 'base'
    cfg.TEST.SPLIT = 'test'; cfg.OUTPUT_DIR = str(Path(output_dir) / '运行信息')
    cfg.USE_CUDA = True; cfg.TRAINER.TEXTCONSISTENCY.DEVICE = 'cuda'
    cfg.TRAINER.TEXTCONSISTENCY.CLIP_PATH = str(Path(clip_path).resolve())
    cfg.freeze()
    set_random_seed(seed); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    return build_trainer(cfg)


def load_legacy_checkpoint(trainer, model_dir):
    import torch
    from trainers.textconsistency_common import resolve_checkpoint

    path = resolve_checkpoint(model_dir, 'textconsistency', 20)
    observed_sha256 = sha256_file(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    record = validate_legacy_checkpoint(checkpoint, observed_sha256)
    model = trainer._models['textconsistency']
    state = {key: value for key, value in checkpoint['state_dict'].items()
             if not key.endswith(('token_prefix', 'token_suffix'))}
    expected = {key: value for key, value in model.state_dict().items()
                if not key.endswith(('token_prefix', 'token_suffix'))}
    missing, extra = set(expected) - set(state), set(state) - set(expected)
    if missing or extra:
        raise ValueError(f'Legacy tensor set mismatch; missing={sorted(missing)}, unexpected={sorted(extra)}')
    if any(expected[key].shape != state[key].shape for key in expected):
        raise ValueError('Legacy checkpoint tensor shapes do not match the subspace model')
    model.load_state_dict(state, strict=False)
    record.update(path=str(path.resolve()), target_trainer='CoOp_TextDetail_SubspaceRadius')
    trainer.loaded_checkpoints = [record]
    return record


def export_checkpoint(dataset_key, seed, model_dir, split, data_root, clip_path,
                      output_dir, legacy=False):
    model_dir, output_dir = Path(model_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'Partial export must be removed or completed explicitly: {output_dir}')
    trainer = _build_export_trainer(dataset_key, seed, model_dir, split, data_root,
                                    clip_path, output_dir, legacy)
    checkpoint = (load_legacy_checkpoint(trainer, model_dir) if legacy
                  else trainer.load_model(model_dir, epoch=20)[0])
    arrays, paths = collect_predictions(trainer.model, trainer.test_loader, trainer.device)
    items = trainer.dm.dataset.test
    data_root = Path(data_root).resolve()
    rows = [{'sample_id': str(Path(path).resolve().relative_to(data_root)), 'label': int(label)}
            for path, label in zip(paths, arrays['labels'])]
    expected_rows = [{'sample_id': str(Path(item.impath).resolve().relative_to(data_root)),
                      'label': int(item.label)} for item in items]
    if rows != expected_rows or arrays['S'].shape != (len(items), len(trainer.dm.dataset.classnames)):
        raise ValueError('Export rows/classes differ from the deterministic dataset loader')
    prepare_export_directory(output_dir)
    archive_path = output_dir / '预测.npz'
    np.savez_compressed(archive_path, **arrays)
    record = {
        'status': 'exported', 'dataset': dataset_key, 'seed': seed, 'split': split,
        'new_training': not legacy, 'rows': rows,
        'class_order': list(trainer.dm.dataset.classnames),
        'provenance': trainer.run_provenance, 'checkpoint': checkpoint,
        'predictions_sha256': sha256_file(archive_path),
        'branch_semantics': {'Z': 'F+R_subspace_radius', 'F': '0.5*(S+M)',
                             'R': 'description-subspace projected and radius-bounded residual logits'},
    }
    atomic_json(output_dir / '导出.json', record)
    return record


def load_export(folder):
    folder = Path(folder)
    record = json.loads((folder / '导出.json').read_text())
    archive_path = folder / '预测.npz'
    if record.get('predictions_sha256') != sha256_file(archive_path):
        raise ValueError(f'Prediction archive hash mismatch: {archive_path}')
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_ARRAYS:
            raise ValueError(f'Prediction archive has missing or extra arrays: {archive_path}')
        arrays = {name: archive[name].copy() for name in archive.files}
    if arrays['labels'].ndim != 1 or arrays['labels'].dtype != np.int64 or not len(arrays['labels']):
        raise ValueError(f'Malformed labels: {archive_path}')
    for name in BRANCHES:
        value = arrays[name]
        if (value.ndim != 2 or len(value) != len(arrays['labels'])
                or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all()):
            raise ValueError(f'Malformed {name} logits: {archive_path}')
    np.testing.assert_array_equal(arrays['F'], .5 * arrays['S'] + .5 * arrays['M'])
    np.testing.assert_array_equal(arrays['Z'], arrays['F'] + arrays['R'])
    if len(record.get('rows', [])) != len(arrays['labels']):
        raise ValueError(f'Export metadata does not match predictions: {archive_path}')
    return record, arrays


def _case_root(output_root, dataset_key, seed):
    return Path(output_root) / DATASETS[dataset_key]['label'] / f'种子{seed}'


def _export_valid(folder):
    try:
        load_export(folder)
        return True
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, AssertionError):
        return False


def run_training_case(case, data_root, clip_path, output_root, status):
    root = _case_root(output_root, case['dataset'], case['seed'])
    model_dir = root / '模型'
    checkpoint = model_dir / 'textconsistency/model.pth.tar-20'
    if checkpoint.is_file():
        status['events'].append({'case': case, 'step': '训练', 'status': 'resumed_existing_checkpoint'})
    else:
        root.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'Dassl.pytorch'),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        with (root / '训练.log').open('w') as log:
            subprocess.run(train_command(case, data_root, clip_path, model_dir), cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Training finished without epoch-20 checkpoint: {checkpoint}')
        status['events'].append({'case': case, 'step': '训练', 'status': 'completed'})
    for split, label in (('base', '基础测试集'), ('novel', '新类别测试集')):
        folder = root / label
        if _export_valid(folder):
            status['events'].append({'case': case, 'step': label, 'status': 'resumed_existing_export'})
        else:
            if folder.exists():
                raise ValueError(f'Incomplete export exists and will not be overwritten: {folder}')
            export_checkpoint(case['dataset'], case['seed'], model_dir, split,
                              data_root, clip_path, folder)
            status['events'].append({'case': case, 'step': label, 'status': 'completed'})


def analyze_exports(base_folder, novel_folder):
    _, base = load_export(base_folder); _, novel = load_export(novel_folder)
    metrics = {}
    for split, arrays in (('base', base), ('novel', novel)):
        metrics[split] = {
            branch: float(100. * (arrays[branch].argmax(1) == arrays['labels']).mean())
            for branch in ('F', 'Z')
        }
    hm_f = harmonic_mean(metrics['base']['F'], metrics['novel']['F'])
    hm_z = harmonic_mean(metrics['base']['Z'], metrics['novel']['Z'])
    return {'base': metrics['base'], 'novel': metrics['novel'], 'hm_F': hm_f, 'hm_Z': hm_z,
            'hm_gain': hm_z - hm_f, 'novel_gain': metrics['novel']['Z'] - metrics['novel']['F']}


def analyze_dataset(output_root, dataset_key, seeds):
    rows = []
    for seed in seeds:
        root = _case_root(output_root, dataset_key, seed)
        row = analyze_exports(root / '基础测试集', root / '新类别测试集')
        row['seed'] = seed; rows.append(row)
    hm_gains = [row['hm_gain'] for row in rows]
    novel_mean_gain = float(np.mean([row['novel_gain'] for row in rows]))
    decision = (zero_training_decision(hm_gains[0], novel_mean_gain) if len(rows) == 1
                else confirmation_decision(hm_gains, novel_mean_gain))
    return {'dataset': dataset_key, 'seeds': list(seeds), 'formula': FORMULA,
            'results': rows, 'decision': decision}


def write_report(path, title, analysis):
    lines = [f'# {title}', '',
             f"结论：{'通过当前阶段' if analysis['decision']['passes'] else '未通过当前阶段'}。", '',
             '| seed | F Base | Z Base | F Novel | Z Novel | F HM | Z HM | HM增益 | Novel增益 |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in analysis['results']:
        lines.append(f"| {row['seed']} | {row['base']['F']:.2f} | {row['base']['Z']:.2f} | "
                     f"{row['novel']['F']:.2f} | {row['novel']['Z']:.2f} | "
                     f"{row['hm_F']:.2f} | {row['hm_Z']:.2f} | {row['hm_gain']:+.2f} | "
                     f"{row['novel_gain']:+.2f} |")
    lines.extend(['', '判定使用冻结协议；没有搜索残差系数、门控阈值或额外损失权重。'])
    Path(path).write_text('\n'.join(lines) + '\n')


def _write_analysis(folder, title, analysis):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / '结果分析.json', analysis)
    write_report(folder / '结果报告.md', title, analysis)


def run_legacy_reload(protocol, status):
    root = Path(protocol['output_root']) / '旧检查点零训练重载'
    for split, label in (('base', '基础测试集'), ('novel', '新类别测试集')):
        folder = root / label
        if _export_valid(folder):
            status['events'].append({'phase': '旧检查点零训练重载', 'step': label,
                                     'status': 'resumed_existing_export'})
        else:
            if folder.exists():
                raise ValueError(f'Incomplete export exists and will not be overwritten: {folder}')
            export_checkpoint('dtd', 1, protocol['legacy_model_dir'], split,
                              protocol['data_root'], protocol['clip_path'], folder, legacy=True)
            status['events'].append({'phase': '旧检查点零训练重载', 'step': label,
                                     'status': 'completed'})
    row = analyze_exports(root / '基础测试集', root / '新类别测试集'); row['seed'] = 1
    analysis = {'dataset': 'dtd', 'seeds': [1], 'new_trainings': 0, 'formula': FORMULA,
                'results': [row], 'decision': zero_training_decision(row['hm_gain'], row['novel_gain'])}
    _write_analysis(root / '筛选结论', 'DTD旧检查点零训练重载', analysis)
    return analysis


def run_study(args):
    import torch

    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text()); verify_protocol(protocol)
    if not torch.cuda.is_available():
        raise RuntimeError('An available CUDA GPU is required to export or train')
    clip_path = Path(protocol['clip_path']).resolve()
    if not clip_path.is_file() or sha256_file(clip_path) != CLIP_SHA256:
        raise FileNotFoundError(f'Missing or incorrect CLIP checkpoint: {clip_path}')
    output_root = Path(protocol['output_root']).resolve(); output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / '执行状态.json'
    status = {'state': 'running', 'protocol': str(protocol_path),
              'gpu': torch.cuda.get_device_name(0), 'started': time.time(),
              'events': [], 'current_phase': '旧检查点零训练重载', 'new_trainings_completed': 0}
    atomic_json(status_path, status)
    try:
        legacy = run_legacy_reload(protocol, status); atomic_json(status_path, status)
        if not legacy['decision']['passes']:
            status.update(state='stopped_by_preregistered_rule', conclusion='旧检查点零训练重载未通过')
            return status
        phases = (
            ('DTD单种子训练筛选', 'dtd', (1,)),
            ('DTD三种子开发确认', 'dtd', (2, 3)),
            ('UCF101单种子独立筛选', 'ucf101', (1,)),
            ('UCF101三种子独立确认', 'ucf101', (2, 3)),
        )
        for phase, dataset, new_seeds in phases:
            status['current_phase'] = phase; atomic_json(status_path, status)
            for seed in new_seeds:
                case = {'phase': phase, 'dataset': dataset, 'seed': seed}
                before = _case_root(output_root, dataset, seed) / '模型/textconsistency/model.pth.tar-20'
                existed = before.is_file()
                run_training_case(case, protocol['data_root'], protocol['clip_path'], output_root, status)
                if not existed:
                    status['new_trainings_completed'] += 1
                atomic_json(status_path, status)
            seeds = (1,) if '单种子' in phase else (1, 2, 3)
            analysis = analyze_dataset(output_root, dataset, seeds)
            result_dir = output_root / DATASETS[dataset]['label'] / ('单种子筛选' if len(seeds) == 1 else '三种子确认')
            _write_analysis(result_dir, phase, analysis)
            if not analysis['decision']['passes']:
                status.update(state='stopped_by_preregistered_rule', conclusion=f'{phase}未通过')
                return status
        status.update(state='completed', conclusion='DTD开发与UCF101独立确认均通过固定规则')
        return status
    except BaseException as error:
        status.update(state='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status['ended'] = time.time(); atomic_json(status_path, status)


def write_protocol(args):
    target = Path(args.output).resolve()
    if target.exists():
        raise FileExistsError(f'Protocol already exists: {target}')
    protocol = build_protocol(args.data_root, args.clip_path, args.legacy_model_dir,
                              args.output_root, args.repository_commit)
    atomic_json(target, protocol)
    return {'status': 'prepared', 'protocol': str(target),
            'maximum_new_trainings': protocol['maximum_new_trainings']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan')
    plan.add_argument('--data-root', required=True); plan.add_argument('--clip-path', required=True)
    plan.add_argument('--legacy-model-dir', required=True); plan.add_argument('--output-root', required=True)
    plan.add_argument('--repository-commit', required=True); plan.add_argument('--output', required=True)
    run = commands.add_parser('run'); run.add_argument('--protocol', required=True)
    extract = commands.add_parser('export')
    extract.add_argument('--dataset', choices=DATASETS, required=True); extract.add_argument('--seed', type=int, required=True)
    extract.add_argument('--model-dir', required=True); extract.add_argument('--split', choices=('base', 'novel'), required=True)
    extract.add_argument('--data-root', required=True); extract.add_argument('--clip-path', required=True)
    extract.add_argument('--output-dir', required=True); extract.add_argument('--legacy', action='store_true')
    inspect = commands.add_parser('analyze')
    inspect.add_argument('--output-root', required=True); inspect.add_argument('--dataset', choices=DATASETS, required=True)
    inspect.add_argument('--seeds', required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        result = write_protocol(args)
    elif args.command == 'run':
        result = run_study(args)
    elif args.command == 'export':
        result = export_checkpoint(args.dataset, args.seed, args.model_dir, args.split,
                                   args.data_root, args.clip_path, args.output_dir, args.legacy)
    else:
        result = analyze_dataset(args.output_root, args.dataset,
                                 tuple(int(value) for value in args.seeds.split(',')))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
