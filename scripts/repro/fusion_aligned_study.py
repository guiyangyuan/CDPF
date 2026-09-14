#!/usr/bin/env python3
"""Run staged validation of the fusion-aligned text-detail residual.

The changed model trains and deploys the residual on the same anchor:
    F = 0.5 * (S + M)
    Z = F + R
    L = CE(S, y) + CE(stopgrad(F) + R, y)

DTD is the development dataset. UCF101 remains untouched until the frozen
mechanism passes DTD seed 1 and then the DTD three-seed confirmation.
"""
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
FORMULA = {
    'simple_fusion': 'F = 0.5 * (S + M)',
    'corrected': 'Z = F + R',
    'objective': 'CE(S, y) + CE(stopgrad(F) + R, y)',
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
    'configs/repro/text_detail/fusion_aligned_attentive.yaml',
    'templates/texture_prompts_full.json',
    'templates/ucf_prompts_full.json',
    'scripts/repro/fusion_aligned_study.py',
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


def planned_cases():
    phases = (
        ('DTD单种子筛选', 'dtd', (1,)),
        ('DTD三种子确认', 'dtd', (2, 3)),
        ('UCF101三种子确认', 'ucf101', (1, 2, 3)),
    )
    return [{'phase': phase, 'dataset': dataset, 'seed': seed}
            for phase, dataset, seeds in phases for seed in seeds]


def screening_decision(hm_gain, novel_gain):
    values = [float(hm_gain), float(novel_gain)]
    if not all(np.isfinite(values)):
        raise ValueError('Screening decision requires finite gains')
    criteria = {
        '融合对齐残差HM优于简单融合': values[0] > 0.,
        '新类别准确率不下降': values[1] >= 0.,
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


def build_protocol(data_root, clip_path, output_root, repository_commit):
    return {
        'protocol': '融合对齐残差跨数据集验证-v1',
        'created_before_results': True,
        'repository_commit': repository_commit,
        'data_root': str(Path(data_root)),
        'clip_path': str(Path(clip_path)),
        'clip_sha256': CLIP_SHA256,
        'output_root': str(Path(output_root)),
        'formula': FORMULA,
        'training': {'shots': 16, 'epochs': 20, 'seeds': [1, 2, 3],
                     'pooling': 'attentive', 'backbone': 'ViT-B/16'},
        'dataset_order': ['dtd', 'ucf101'],
        'cases': planned_cases(),
        'stopping_rule': {
            'after_DTD_seed1': 'stop unless Z>F in HM and Z>=F in Novel accuracy',
            'after_DTD_three_seeds': 'stop unless mean HM gain is positive, at least 2/3 HM gains are positive, and mean Novel does not fall',
            'after_UCF101_three_seeds': 'same three-seed rule; report as independent confirmation',
            'no_one_percentage_point_requirement': True,
        },
        'evidence_policy': {
            'DTD': 'development and staged screening; no threshold search',
            'UCF101': 'untouched independent confirmation until all DTD gates pass',
            'Aircraft': 'historical post-hoc evidence only',
        },
        'files_sha256': dependency_hashes(),
    }


def verify_protocol(protocol):
    if (protocol.get('protocol') != '融合对齐残差跨数据集验证-v1'
            or protocol.get('formula') != FORMULA
            or protocol.get('clip_sha256') != CLIP_SHA256
            or protocol.get('cases') != planned_cases()):
        raise ValueError('Unknown or modified validation protocol')
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
        '--trainer', 'CoOp_TextDetail_FusionAligned',
        '--dataset-config-file', str(ROOT / 'configs/datasets' / dataset['config']),
        '--config-file', str(ROOT / 'configs/repro/text_detail/fusion_aligned_attentive.yaml'),
        '--output-dir', str(model_dir),
        'DATASET.SUBSAMPLE_CLASSES', 'base', 'USE_CUDA', 'True',
        'TRAINER.TEXTCONSISTENCY.DEVICE', 'cuda',
        'TRAINER.TEXTCONSISTENCY.CLIP_PATH', str(clip_path), 'TEST.NO_TEST', 'True',
    ]


def _collect_predictions(model, loader, device):
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


def export_checkpoint(dataset_key, seed, model_dir, split, data_root, clip_path, output_dir):
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
    expected = DATASETS[dataset_key]
    if (cfg.TRAINER.NAME != 'CoOp_TextDetail_FusionAligned'
            or cfg.DATASET.NAME != expected['name'] or cfg.DATASET.NUM_SHOTS != 16
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
    arrays, paths = _collect_predictions(trainer.model, trainer.test_loader, trainer.device)
    items = trainer.dm.dataset.test
    rows = [{'sample_id': str(Path(path).resolve().relative_to(Path(data_root).resolve())),
             'label': int(label)} for path, label in zip(paths, arrays['labels'])]
    expected_rows = [{'sample_id': str(Path(item.impath).resolve().relative_to(Path(data_root).resolve())),
                      'label': int(item.label)} for item in items]
    if rows != expected_rows or arrays['S'].shape != (len(items), len(trainer.dm.dataset.classnames)):
        raise ValueError('Export rows/classes differ from the deterministic dataset loader')
    prepare_export_directory(output_dir)
    archive_path = output_dir / '预测.npz'
    np.savez_compressed(archive_path, **arrays)
    record = {
        'status': 'exported', 'dataset': dataset_key, 'seed': seed, 'split': split,
        'rows': rows, 'class_order': list(trainer.dm.dataset.classnames),
        'provenance': trainer.run_provenance, 'checkpoint': trainer.loaded_checkpoints,
        'predictions_sha256': sha256_file(archive_path),
        'branch_semantics': {'Z': 'F+R', 'F': '0.5*(S+M)', 'R': 'not a classifier'},
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
    if arrays['labels'].ndim != 1 or arrays['labels'].dtype != np.int64 or len(arrays['labels']) == 0:
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


def _case_root(output_root, case):
    return Path(output_root) / DATASETS[case['dataset']]['label'] / f"种子{case['seed']}"


def _export_valid(folder):
    try:
        load_export(folder)
        return True
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, AssertionError):
        return False


def run_case(case, data_root, clip_path, output_root, status):
    root = _case_root(output_root, case)
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
            export_checkpoint(case['dataset'], case['seed'], model_dir, split, data_root, clip_path, folder)
            status['events'].append({'case': case, 'step': label, 'status': 'completed'})


def harmonic_mean(base, novel):
    return float(2 * base * novel / (base + novel)) if base + novel else 0.


def analyze_dataset(output_root, dataset_key, seeds):
    rows = []
    for seed in seeds:
        split_metrics = {}
        for split, label in (('base', '基础测试集'), ('novel', '新类别测试集')):
            _, arrays = load_export(_case_root(output_root, {'dataset': dataset_key, 'seed': seed}) / label)
            split_metrics[split] = {
                'F': float(100 * (arrays['F'].argmax(1) == arrays['labels']).mean()),
                'Z': float(100 * (arrays['Z'].argmax(1) == arrays['labels']).mean()),
            }
        rows.append({
            'seed': seed, 'base': split_metrics['base'], 'novel': split_metrics['novel'],
            'hm_F': harmonic_mean(split_metrics['base']['F'], split_metrics['novel']['F']),
            'hm_Z': harmonic_mean(split_metrics['base']['Z'], split_metrics['novel']['Z']),
        })
    hm_gains = [row['hm_Z'] - row['hm_F'] for row in rows]
    novel_gain = float(np.mean([row['novel']['Z'] - row['novel']['F'] for row in rows]))
    decision = (screening_decision(hm_gains[0], novel_gain) if len(rows) == 1
                else confirmation_decision(hm_gains, novel_gain))
    return {'dataset': dataset_key, 'seeds': list(seeds), 'formula': FORMULA,
            'results': rows,
            'comparison': {'Z对F_HM逐种子增益': hm_gains, 'Z对F_Novel平均增益': novel_gain},
            'decision': decision}


def write_dataset_report(path, analysis):
    comparison, decision = analysis['comparison'], analysis['decision']
    lines = [
        f"# {DATASETS[analysis['dataset']]['label']} 融合对齐残差验证", '',
        f"结论：{'通过当前阶段' if decision['passes'] else '未通过当前阶段'}。", '',
        '| seed | F Base | Z Base | F Novel | Z Novel | F HM | Z HM |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in analysis['results']:
        lines.append(f"| {row['seed']} | {row['base']['F']:.2f} | {row['base']['Z']:.2f} | "
                     f"{row['novel']['F']:.2f} | {row['novel']['Z']:.2f} | "
                     f"{row['hm_F']:.2f} | {row['hm_Z']:.2f} |")
    lines.extend([
        '', f"Z 相对 F 的 HM 逐种子增益：{comparison['Z对F_HM逐种子增益']}。",
        f"Z 相对 F 的 Novel 平均增益：{comparison['Z对F_Novel平均增益']:+.4f} 个百分点。", '',
        '判定规则在实验前固定，没有搜索阈值、融合系数或 +1pp 门槛。',
    ])
    Path(path).write_text('\n'.join(lines) + '\n')


def run_study(args):
    import torch

    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol)
    data_root = Path(protocol['data_root']).resolve()
    clip_path = Path(protocol['clip_path']).resolve()
    output_root = Path(protocol['output_root']).resolve()
    if not torch.cuda.is_available():
        raise RuntimeError('An available CUDA GPU is required to run the study')
    if not clip_path.is_file() or sha256_file(clip_path) != CLIP_SHA256:
        raise FileNotFoundError(f'Missing or incorrect CLIP checkpoint: {clip_path}')
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / '执行状态.json'
    status = {'state': 'running', 'protocol': str(protocol_path),
              'gpu': torch.cuda.get_device_name(0), 'started': time.time(),
              'events': [], 'current_phase': None}
    atomic_json(status_path, status)
    try:
        for phase in ('DTD单种子筛选', 'DTD三种子确认', 'UCF101三种子确认'):
            status['current_phase'] = phase; atomic_json(status_path, status)
            for case in [item for item in planned_cases() if item['phase'] == phase]:
                run_case(case, data_root, clip_path, output_root, status)
                atomic_json(status_path, status)
            dataset = 'dtd' if phase.startswith('DTD') else 'ucf101'
            seeds = (1,) if phase == 'DTD单种子筛选' else (1, 2, 3)
            analysis = analyze_dataset(output_root, dataset, seeds)
            result_dir = output_root / DATASETS[dataset]['label'] / ('单种子筛选' if len(seeds) == 1 else '三种子确认')
            result_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(result_dir / '结果分析.json', analysis)
            write_dataset_report(result_dir / '结果报告.md', analysis)
            if not analysis['decision']['passes']:
                status.update(state='stopped_by_preregistered_rule', conclusion=f'{phase}未通过')
                return status
        status.update(state='completed', conclusion='DTD和UCF101均通过固定规则')
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
    protocol = build_protocol(args.data_root, args.clip_path, args.output_root, args.repository_commit)
    atomic_json(target, protocol)
    return {'status': 'prepared', 'protocol': str(target), 'training_cases': len(protocol['cases'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan')
    plan.add_argument('--data-root', required=True); plan.add_argument('--clip-path', required=True)
    plan.add_argument('--output-root', required=True); plan.add_argument('--repository-commit', required=True)
    plan.add_argument('--output', required=True)
    run = commands.add_parser('run'); run.add_argument('--protocol', required=True)
    extract = commands.add_parser('export')
    extract.add_argument('--dataset', choices=DATASETS, required=True); extract.add_argument('--seed', type=int, required=True)
    extract.add_argument('--model-dir', required=True); extract.add_argument('--split', choices=('base', 'novel'), required=True)
    extract.add_argument('--data-root', required=True); extract.add_argument('--clip-path', required=True)
    extract.add_argument('--output-dir', required=True)
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
                                   args.data_root, args.clip_path, args.output_dir)
    else:
        seeds = tuple(int(value) for value in args.seeds.split(','))
        result = analyze_dataset(args.output_root, args.dataset, seeds)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
