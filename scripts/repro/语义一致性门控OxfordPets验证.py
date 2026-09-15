#!/usr/bin/env python3
"""Run the frozen semantic-alignment gate on independent OxfordPets evidence."""
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

BRANCHES_2D = ('S', 'M', 'R', 'Z', 'F')
REQUIRED_ARRAYS = {'labels', 'Q', *BRANCHES_2D}
CLIP_SHA256 = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
OXFORD_PETS_SPLIT_SHA256 = 'd7f3ffe65b3919f95132e51422894ae18686b83c331fd8c0adcdc3dc2de6301e'
OXFORD_PETS_SPLIT_COUNTS = {'train': 2944, 'val': 736, 'test': 3669}
FORMULA = {
    'simple_fusion': 'F=0.5*(S+M)',
    'radius_residual': 'U=F+R_hat',
    'semantic_direction': 'D=center(M-S)',
    'residual_direction': 'R0=center(R_hat)',
    'gate': 'q=clamp(cos(R0,D),0,1)',
    'corrected': 'G=F+q*R_hat',
    'training_objective': 'CE(S,y)+CE(stopgrad(F)+R_hat,y)',
    'gate_training': 'inference_only_after_ungated_residual_training',
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
    'datasets/oxford_pets.py',
    'configs/datasets/oxford_pets.yaml',
    'configs/repro/text_detail/semantic_gated_radius_attentive.yaml',
    'templates/pets_prompts_full.json',
    'scripts/repro/语义一致性门控OxfordPets验证.py',
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


def planned_cases():
    return [
        {'phase': 'OxfordPets单种子独立筛选', 'seed': 1},
        {'phase': 'OxfordPets三种子独立确认', 'seed': 2},
        {'phase': 'OxfordPets三种子独立确认', 'seed': 3},
    ]


def screening_decision(hm_gain, novel_gain):
    values = [float(hm_gain), float(novel_gain)]
    if not all(np.isfinite(values)):
        raise ValueError('Screening decision requires finite gains')
    criteria = {'HM增益为正': values[0] > 0., 'Novel准确率不下降': values[1] >= 0.}
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
        '平均Novel准确率不下降': novel >= 0.,
    }
    return {'passes': all(criteria.values()), 'criteria': criteria,
            'hm_gains': gains, 'mean_hm_gain': float(np.mean(gains)),
            'novel_mean_gain': novel, 'positive_seeds': positive}


def dependency_hashes():
    missing = [name for name in DEPENDENCIES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing protocol dependency: {missing[0]}')
    return {name: sha256_file(ROOT / name) for name in DEPENDENCIES}


def validate_ucf_confirmation_result(path, expected_sha256=None):
    path = Path(path)
    resolved = path if path.is_absolute() else ROOT / path
    observed = sha256_file(resolved)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError('UCF101 confirmation result hash mismatch')
    result = json.loads(resolved.read_text())
    aggregate = result.get('aggregate', {})
    if (len(result.get('results', [])) != 3
            or not aggregate.get('passes_frozen_rule')
            or aggregate.get('mean_hm_gain', 0.) <= 0.
            or aggregate.get('mean_novel_gain', -1.) < 0.
            or aggregate.get('positive_hm_seeds', 0) < 2):
        raise ValueError('UCF101 prerequisite did not pass the frozen confirmation rule')
    try:
        portable_path = str(resolved.resolve().relative_to(ROOT))
    except ValueError:
        portable_path = str(resolved.resolve())
    return {'path': portable_path, 'sha256': observed,
            'mean_hm_gain': aggregate['mean_hm_gain'],
            'mean_novel_gain': aggregate['mean_novel_gain'],
            'positive_hm_seeds': aggregate['positive_hm_seeds']}


def validate_dataset(data_root):
    dataset_root = Path(data_root).resolve() / 'oxford_pets'
    split_path = dataset_root / 'split_zhou_OxfordPets.json'
    if not split_path.is_file() or sha256_file(split_path) != OXFORD_PETS_SPLIT_SHA256:
        raise FileNotFoundError(f'Missing or incorrect OxfordPets fixed split: {split_path}')
    split = json.loads(split_path.read_text())
    counts = {name: len(split.get(name, [])) for name in OXFORD_PETS_SPLIT_COUNTS}
    if set(split) != set(OXFORD_PETS_SPLIT_COUNTS) or counts != OXFORD_PETS_SPLIT_COUNTS:
        raise ValueError(f'Unexpected OxfordPets split counts: {counts}')
    rows = [row for name in ('train', 'val', 'test') for row in split[name]]
    names = [row[0] for row in rows]
    labels = {int(row[1]) for row in rows}
    if (any(not isinstance(row, list) or len(row) != 3 for row in rows)
            or len(names) != len(set(names)) or labels != set(range(37))):
        raise ValueError('OxfordPets split rows, uniqueness, or labels are invalid')
    missing = [name for name in names if not (dataset_root / 'images' / name).is_file()]
    if missing:
        raise FileNotFoundError(f'OxfordPets split image is missing: {missing[0]}')
    for name in ('trainval.txt', 'test.txt'):
        if not (dataset_root / 'annotations' / name).is_file():
            raise FileNotFoundError(f'OxfordPets annotation is missing: {name}')
    return {'split_sha256': OXFORD_PETS_SPLIT_SHA256, 'split_counts': counts,
            'classes': len(labels), 'unique_split_images': len(names)}


def build_protocol(data_root, clip_path, output_root, repository_commit,
                   ucf_confirmation_result):
    ucf = validate_ucf_confirmation_result(ucf_confirmation_result)
    return {
        'protocol': '语义一致性门控-OxfordPets独立验证-v1',
        'created_before_OxfordPets_outputs': True,
        'repository_commit': repository_commit,
        'data_root': str(Path(data_root)),
        'clip_path': str(Path(clip_path)),
        'clip_sha256': CLIP_SHA256,
        'output_root': str(Path(output_root)),
        'formula': FORMULA,
        'dataset_receipt_expected': {
            'split_sha256': OXFORD_PETS_SPLIT_SHA256,
            'split_counts': OXFORD_PETS_SPLIT_COUNTS,
            'classes': 37,
            'unique_split_images': sum(OXFORD_PETS_SPLIT_COUNTS.values()),
        },
        'UCF101_confirmation_prerequisite': ucf,
        'training': {'dataset': 'OxfordPets', 'shots': 16, 'epochs': 20,
                     'seeds': [1, 2, 3], 'pooling': 'attentive',
                     'backbone': 'ViT-B/16', 'trainer_identity_version': 8},
        'cases': planned_cases(),
        'maximum_new_trainings': 3,
        'stopping_rule': {
            'after_seed1': 'stop unless G>F in HM and G>=F in Novel accuracy',
            'after_three_seeds': ('stop unless mean HM gain is positive, at least 2/3 HM gains '
                                  'are positive, and mean Novel accuracy does not fall'),
        },
        'evidence_policy': {
            'DTD': 'development only; formula selected before UCF101',
            'UCF101': 'first independent confirmation; source of the frozen prerequisite',
            'OxfordPets': 'second independent dataset; no formula or threshold changes after output',
        },
        'files_sha256': dependency_hashes(),
    }


def verify_protocol(protocol):
    if (protocol.get('protocol') != '语义一致性门控-OxfordPets独立验证-v1'
            or protocol.get('formula') != FORMULA
            or protocol.get('clip_sha256') != CLIP_SHA256
            or protocol.get('cases') != planned_cases()
            or protocol.get('maximum_new_trainings') != 3):
        raise ValueError('Unknown or modified OxfordPets semantic-gate protocol')
    observed = dependency_hashes()
    if protocol.get('files_sha256') != observed:
        changed = sorted(name for name in set(observed) | set(protocol.get('files_sha256', {}))
                         if observed.get(name) != protocol.get('files_sha256', {}).get(name))
        raise ValueError(f'Frozen source/configuration changed: {changed[:3]}')
    if protocol.get('training') != {'dataset': 'OxfordPets', 'shots': 16, 'epochs': 20,
                                    'seeds': [1, 2, 3], 'pooling': 'attentive',
                                    'backbone': 'ViT-B/16', 'trainer_identity_version': 8}:
        raise ValueError('OxfordPets training protocol changed')
    observed_dataset = validate_dataset(protocol['data_root'])
    if protocol.get('dataset_receipt_expected') != observed_dataset:
        raise ValueError('OxfordPets dataset receipt differs from the fixed protocol')
    ucf = protocol['UCF101_confirmation_prerequisite']
    validate_ucf_confirmation_result(ucf['path'], ucf['sha256'])


def train_command(seed, data_root, clip_path, model_dir):
    return [
        sys.executable, '-u', str(ROOT / 'train.py'),
        '--root', str(data_root), '--seed', str(seed),
        '--trainer', 'CoOp_TextDetail_SemanticGatedRadius',
        '--dataset-config-file', str(ROOT / 'configs/datasets/oxford_pets.yaml'),
        '--config-file', str(ROOT / 'configs/repro/text_detail/semantic_gated_radius_attentive.yaml'),
        '--output-dir', str(model_dir),
        'DATASET.SUBSAMPLE_CLASSES', 'base', 'USE_CUDA', 'True',
        'TRAINER.TEXTCONSISTENCY.DEVICE', 'cuda',
        'TRAINER.TEXTCONSISTENCY.CLIP_PATH', str(clip_path), 'TEST.NO_TEST', 'True',
    ]


def collect_predictions(model, loader, device):
    import torch

    arrays = {key: [] for key in ('labels', 'Q', *BRANCHES_2D)}
    paths = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            outputs = model(batch['img'].to(device))
            labels = batch['label'].cpu().numpy()
            arrays['labels'].append(labels)
            for key in BRANCHES_2D:
                value = outputs[key].cpu().numpy()
                if value.ndim != 2 or len(value) != len(labels) or not np.isfinite(value).all():
                    raise ValueError(f'Malformed {key} logits during export')
                arrays[key].append(value)
            gate = outputs['Q'].cpu().numpy()
            if gate.shape != (len(labels),) or not np.isfinite(gate).all():
                raise ValueError('Malformed Q gate during export')
            arrays['Q'].append(gate)
            paths.extend(str(path) for path in batch['impath'])
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Export has empty or duplicate sample paths')
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    np.testing.assert_array_equal(arrays['F'], .5 * arrays['S'] + .5 * arrays['M'])
    np.testing.assert_allclose(arrays['Z'], arrays['F'] + arrays['Q'][:, None] * arrays['R'],
                               rtol=1e-6, atol=1e-6)
    if not ((0. <= arrays['Q']) & (arrays['Q'] <= 1.)).all():
        raise ValueError('Q gate is outside [0, 1]')
    return arrays, paths


def export_checkpoint(seed, model_dir, split, data_root, clip_path, output_dir):
    import torch
    from dassl.config import get_cfg_default
    from dassl.engine import build_trainer
    from dassl.utils import set_random_seed
    import train

    model_dir, output_dir = Path(model_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'Partial export must be removed or completed explicitly: {output_dir}')
    cfg = get_cfg_default(); train.extend_cfg(cfg)
    cfg.merge_from_file(str(model_dir / 'effective_config.yaml'))
    if (cfg.TRAINER.NAME != 'CoOp_TextDetail_SemanticGatedRadius'
            or cfg.DATASET.NAME != 'OxfordPets' or cfg.DATASET.NUM_SHOTS != 16
            or cfg.OPTIM.MAX_EPOCH != 20 or cfg.SEED != seed
            or cfg.DATASET.SUBSAMPLE_CLASSES != 'base'):
        raise ValueError('Checkpoint effective configuration differs from the fixed protocol')
    cfg.DATASET.ROOT = str(Path(data_root).resolve())
    cfg.DATASET.SUBSAMPLE_CLASSES = 'new' if split == 'novel' else 'base'
    cfg.TEST.SPLIT = 'test'; cfg.OUTPUT_DIR = str(output_dir / '运行信息')
    cfg.USE_CUDA = True; cfg.TRAINER.TEXTCONSISTENCY.DEVICE = 'cuda'
    cfg.TRAINER.TEXTCONSISTENCY.CLIP_PATH = str(Path(clip_path).resolve())
    cfg.freeze()
    set_random_seed(seed); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    trainer = build_trainer(cfg); trainer.load_model(model_dir, epoch=20)
    arrays, paths = collect_predictions(trainer.model, trainer.test_loader, trainer.device)
    items = trainer.dm.dataset.test
    data_root = Path(data_root).resolve()
    rows = [{'sample_id': str(Path(path).resolve().relative_to(data_root)), 'label': int(label)}
            for path, label in zip(paths, arrays['labels'])]
    expected_rows = [{'sample_id': str(Path(item.impath).resolve().relative_to(data_root)),
                      'label': int(item.label)} for item in items]
    if rows != expected_rows or arrays['S'].shape != (len(items), len(trainer.dm.dataset.classnames)):
        raise ValueError('Export rows/classes differ from the deterministic dataset loader')
    # Trainer initialization creates cfg.OUTPUT_DIR (a child of this folder).
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / '预测.npz'
    np.savez_compressed(archive_path, **arrays)
    record = {
        'status': 'exported', 'dataset': 'oxford_pets', 'seed': seed, 'split': split,
        'rows': rows, 'class_order': list(trainer.dm.dataset.classnames),
        'provenance': trainer.run_provenance, 'checkpoint': trainer.loaded_checkpoints,
        'predictions_sha256': sha256_file(archive_path),
        'branch_semantics': {'F': '0.5*(S+M)', 'R': 'description-radius bounded residual',
                             'Q': 'semantic alignment scalar', 'Z': 'F+Q*R'},
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
    labels = arrays['labels']
    if labels.ndim != 1 or labels.dtype != np.int64 or not len(labels):
        raise ValueError(f'Malformed labels: {archive_path}')
    for key in BRANCHES_2D:
        value = arrays[key]
        if (value.ndim != 2 or len(value) != len(labels)
                or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all()):
            raise ValueError(f'Malformed {key} logits: {archive_path}')
    if arrays['Q'].shape != (len(labels),) or not np.isfinite(arrays['Q']).all():
        raise ValueError(f'Malformed Q gate: {archive_path}')
    np.testing.assert_array_equal(arrays['F'], .5 * arrays['S'] + .5 * arrays['M'])
    np.testing.assert_allclose(arrays['Z'], arrays['F'] + arrays['Q'][:, None] * arrays['R'],
                               rtol=1e-6, atol=1e-6)
    if len(record.get('rows', [])) != len(labels):
        raise ValueError(f'Export metadata does not match predictions: {archive_path}')
    return record, arrays


def case_root(output_root, seed):
    return Path(output_root) / 'OxfordPets' / f'种子{seed}'


def export_valid(folder):
    try:
        load_export(folder)
        return True
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, AssertionError):
        return False


def run_case(seed, data_root, clip_path, output_root, status):
    root = case_root(output_root, seed)
    model_dir = root / '模型'
    checkpoint = model_dir / 'textconsistency/model.pth.tar-20'
    existed = checkpoint.is_file()
    if existed:
        status['events'].append({'seed': seed, 'step': '训练', 'status': 'resumed_existing_checkpoint'})
    else:
        root.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'Dassl.pytorch'),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        with (root / '训练.log').open('w') as log:
            subprocess.run(train_command(seed, data_root, clip_path, model_dir), cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Training finished without epoch-20 checkpoint: {checkpoint}')
        status['events'].append({'seed': seed, 'step': '训练', 'status': 'completed'})
    for split, label in (('base', '基础测试集'), ('novel', '新类别测试集')):
        folder = root / label
        if export_valid(folder):
            status['events'].append({'seed': seed, 'step': label,
                                     'status': 'resumed_existing_export'})
        else:
            if folder.exists():
                raise ValueError(f'Incomplete export exists and will not be overwritten: {folder}')
            export_checkpoint(seed, model_dir, split, data_root, clip_path, folder)
            status['events'].append({'seed': seed, 'step': label, 'status': 'completed'})
    return not existed


def analyze_dataset(output_root, seeds):
    rows = []
    for seed in seeds:
        splits = {}
        for split, label in (('base', '基础测试集'), ('novel', '新类别测试集')):
            _, arrays = load_export(case_root(output_root, seed) / label)
            labels = arrays['labels']
            fusion, ungated, gated = arrays['F'], arrays['F'] + arrays['R'], arrays['Z']
            splits[split] = {
                'F': float(100. * (fusion.argmax(1) == labels).mean()),
                'U': float(100. * (ungated.argmax(1) == labels).mean()),
                'G': float(100. * (gated.argmax(1) == labels).mean()),
                'gate_mean': float(arrays['Q'].mean()),
                'gate_zero_fraction': float((arrays['Q'] == 0.).mean()),
            }
        row = {'seed': seed, 'base': splits['base'], 'novel': splits['novel']}
        for branch in ('F', 'U', 'G'):
            row[f'hm_{branch}'] = harmonic_mean(splits['base'][branch], splits['novel'][branch])
        row['hm_gain'] = row['hm_G'] - row['hm_F']
        row['novel_gain'] = row['novel']['G'] - row['novel']['F']
        rows.append(row)
    gains = [row['hm_gain'] for row in rows]
    novel_mean_gain = float(np.mean([row['novel_gain'] for row in rows]))
    decision = (screening_decision(gains[0], novel_mean_gain) if len(rows) == 1
                else confirmation_decision(gains, novel_mean_gain))
    return {'dataset': 'OxfordPets', 'evidence_role': 'independent_confirmation',
            'seeds': list(seeds), 'formula': FORMULA, 'results': rows,
            'decision': decision}


def write_report(path, title, analysis):
    lines = [f'# {title}', '',
             f"结论：{'通过当前阶段' if analysis['decision']['passes'] else '未通过当前阶段'}。", '',
             '| seed | F Base | G Base | F Novel | G Novel | F HM | G HM | HM增益 |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in analysis['results']:
        lines.append(f"| {row['seed']} | {row['base']['F']:.2f} | {row['base']['G']:.2f} | "
                     f"{row['novel']['F']:.2f} | {row['novel']['G']:.2f} | "
                     f"{row['hm_F']:.2f} | {row['hm_G']:.2f} | {row['hm_gain']:+.2f} |")
    lines.extend(['', 'U=F+R 是未门控半径残差消融；G=F+qR 是冻结语义一致性门控。',
                  'OxfordPets 输出生成后不允许修改公式、阈值或判定规则。'])
    Path(path).write_text('\n'.join(lines) + '\n')


def write_analysis(folder, title, analysis):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    atomic_json(folder / '结果分析.json', analysis)
    write_report(folder / '结果报告.md', title, analysis)


def run_study(protocol_path):
    import torch

    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text()); verify_protocol(protocol)
    if not torch.cuda.is_available():
        raise RuntimeError('An available CUDA GPU is required to run OxfordPets confirmation')
    clip_path = Path(protocol['clip_path']).resolve()
    if not clip_path.is_file() or sha256_file(clip_path) != CLIP_SHA256:
        raise FileNotFoundError(f'Missing or incorrect CLIP checkpoint: {clip_path}')
    output_root = Path(protocol['output_root']).resolve(); output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / '执行状态.json'
    status = {'state': 'running', 'protocol': str(protocol_path),
              'gpu': torch.cuda.get_device_name(0), 'started': time.time(),
              'events': [], 'current_phase': 'OxfordPets单种子独立筛选',
              'new_trainings_completed': 0}
    atomic_json(status_path, status)
    try:
        if run_case(1, protocol['data_root'], clip_path, output_root, status):
            status['new_trainings_completed'] += 1
        seed1 = analyze_dataset(output_root, (1,))
        write_analysis(output_root / 'OxfordPets/单种子独立筛选',
                       'OxfordPets 单种子独立筛选', seed1)
        atomic_json(status_path, status)
        if not seed1['decision']['passes']:
            status.update(state='stopped_by_preregistered_rule', conclusion='OxfordPets seed1未通过')
            return status
        status['current_phase'] = 'OxfordPets三种子独立确认'; atomic_json(status_path, status)
        for seed in (2, 3):
            if run_case(seed, protocol['data_root'], clip_path, output_root, status):
                status['new_trainings_completed'] += 1
            atomic_json(status_path, status)
        all_seeds = analyze_dataset(output_root, (1, 2, 3))
        write_analysis(output_root / 'OxfordPets/三种子独立确认',
                       'OxfordPets 三种子独立确认', all_seeds)
        status.update(
            state=('completed' if all_seeds['decision']['passes']
                   else 'stopped_by_preregistered_rule'),
            conclusion=('OxfordPets三种子独立确认通过' if all_seeds['decision']['passes']
                        else 'OxfordPets三种子独立确认未通过'))
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
    protocol = build_protocol(args.data_root, args.clip_path, args.output_root,
                              args.repository_commit, args.ucf_confirmation_result)
    atomic_json(target, protocol)
    return {'status': 'prepared', 'protocol': str(target),
            'maximum_new_trainings': protocol['maximum_new_trainings']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan')
    plan.add_argument('--data-root', required=True); plan.add_argument('--clip-path', required=True)
    plan.add_argument('--output-root', required=True); plan.add_argument('--repository-commit', required=True)
    plan.add_argument('--ucf-confirmation-result', required=True); plan.add_argument('--output', required=True)
    run = commands.add_parser('run'); run.add_argument('--protocol', required=True)
    inspect = commands.add_parser('analyze')
    inspect.add_argument('--output-root', required=True); inspect.add_argument('--seeds', required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        result = write_protocol(args)
    elif args.command == 'run':
        result = run_study(args.protocol)
    else:
        result = analyze_dataset(args.output_root,
                                 tuple(int(value) for value in args.seeds.split(',')))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
