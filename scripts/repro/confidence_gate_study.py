#!/usr/bin/env python3
"""Run the preregistered DTD -> UCF101 confidence-gate validation.

The runner resumes complete cases, stops after a failed phase, and writes
Chinese-named artifacts. The model is unchanged; G is computed from exported
S/M/R logits by ``confidence_gated_fusion.py``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.repro.confidence_gated_fusion import (  # noqa: E402
    FORMULA,
    confidence_gated_fusion,
    harmonic_mean,
    load_export,
    sha256_file,
)


DATASETS = {
    'dtd': {'label': 'DTD', 'name': 'DescribableTextures', 'config': 'dtd.yaml'},
    'ucf101': {'label': 'UCF101', 'name': 'UCF101', 'config': 'ucf101.yaml'},
}
VARIANTS = {
    'uniform': {'label': '均值描述', 'config': 'uniform.yaml'},
    'attentive': {'label': '图片加权', 'config': 'attentive.yaml'},
}
BRANCHES = ('S', 'M', 'R', 'Z', 'F')
CLIP_SHA256 = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
DEPENDENCIES = (
    'train.py',
    'trainers/coop_text_residual.py',
    'trainers/coop_text_residual_adamw.py',
    'trainers/coop_text_detail.py',
    'trainers/coop_textconsistency_correction.py',
    'trainers/coop_textconsistency_v2_fg.py',
    'trainers/textconsistency_common.py',
    'trainers/textconsistency_runtime.py',
    'datasets/dtd.py',
    'datasets/ucf101.py',
    'configs/datasets/dtd.yaml',
    'configs/datasets/ucf101.yaml',
    'configs/repro/text_detail/uniform.yaml',
    'configs/repro/text_detail/attentive.yaml',
    'templates/texture_prompts_full.json',
    'templates/ucf_prompts_full.json',
    'scripts/repro/confidence_gated_fusion.py',
    'scripts/repro/confidence_gate_study.py',
)


def planned_cases():
    cases = []
    phases = (
        ('DTD单种子筛选', 'dtd', (1,)),
        ('DTD三种子确认', 'dtd', (2, 3)),
        ('UCF101三种子确认', 'ucf101', (1, 2, 3)),
    )
    for phase, dataset, seeds in phases:
        for seed in seeds:
            for variant in ('uniform', 'attentive'):
                cases.append({'phase': phase, 'dataset': dataset, 'seed': seed, 'variant': variant})
    return cases


def screening_decision(residual_hm_gain, attention_hm_gain, residual_novel_gain):
    values = [float(residual_hm_gain), float(attention_hm_gain), float(residual_novel_gain)]
    if not all(np.isfinite(values)):
        raise ValueError('Screening decision requires finite gains')
    criteria = {
        '门控优于同模型简单融合': values[0] > 0.,
        '图片加权优于均值描述': values[1] > 0.,
        '新类别准确率不下降': values[2] >= 0.,
    }
    return {'passes': all(criteria.values()), 'criteria': criteria,
            '门控HM增益': values[0], '图片加权HM增益': values[1], '门控Novel增益': values[2]}


def confirmation_decision(residual_hm_gains, attention_hm_gains, residual_novel_mean_gain):
    residual = [float(value) for value in residual_hm_gains]
    attention = [float(value) for value in attention_hm_gains]
    novel = float(residual_novel_mean_gain)
    if len(residual) != 3 or len(attention) != 3 or not all(np.isfinite([*residual, *attention, novel])):
        raise ValueError('Confirmation requires two finite three-seed gain lists and one Novel gain')
    residual_positive = sum(value > 0. for value in residual)
    attention_positive = sum(value > 0. for value in attention)
    criteria = {
        '门控平均HM增益为正': float(np.mean(residual)) > 0.,
        '门控至少两个种子HM增益为正': residual_positive >= 2,
        '门控平均Novel准确率不下降': novel >= 0.,
        '图片加权平均HM增益为正': float(np.mean(attention)) > 0.,
        '图片加权至少两个种子HM增益为正': attention_positive >= 2,
    }
    return {
        'passes': all(criteria.values()), 'criteria': criteria,
        'residual_hm_gains': residual, 'attention_hm_gains': attention,
        'residual_hm_mean_gain': float(np.mean(residual)),
        'attention_hm_mean_gain': float(np.mean(attention)),
        'residual_novel_mean_gain': novel,
        'residual_positive_seeds': residual_positive,
        'attention_positive_seeds': attention_positive,
    }


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def dependency_hashes():
    missing = [relative for relative in DEPENDENCIES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing protocol dependency: {missing[0]}')
    return {relative: sha256_file(ROOT / relative) for relative in DEPENDENCIES}


def build_protocol(data_root, clip_path, output_root, repository_commit):
    return {
        'protocol': '置信度门控跨数据集验证-v1',
        'created_before_independent_results': True,
        'repository_commit': repository_commit,
        'data_root': str(Path(data_root)),
        'clip_path': str(Path(clip_path)),
        'clip_sha256': CLIP_SHA256,
        'output_root': str(Path(output_root)),
        'formula': FORMULA,
        'training': {'shots': 16, 'epochs': 20, 'seeds': [1, 2, 3],
                     'variants': list(VARIANTS), 'backbone': 'ViT-B/16'},
        'dataset_order': ['dtd', 'ucf101'],
        'cases': planned_cases(),
        'stopping_rule': {
            'after_DTD_seed1': 'stop unless G_attentive>F_attentive in HM, G_attentive>G_uniform in HM, and G_attentive>=F_attentive in Novel accuracy',
            'after_DTD_three_seeds': 'stop unless both HM comparisons have positive mean and at least 2/3 positive seeds, while mean Novel does not fall',
            'no_one_percentage_point_requirement': True,
        },
        'evidence_policy': {
            'Aircraft': 'post-hoc discovery only',
            'DTD': 'independent staged validation; formula cannot be changed',
            'UCF101': 'second independent confirmation; formula cannot be changed',
        },
        'files_sha256': dependency_hashes(),
    }


def verify_protocol(protocol):
    if (protocol.get('protocol') != '置信度门控跨数据集验证-v1'
            or protocol.get('formula') != FORMULA or protocol.get('clip_sha256') != CLIP_SHA256):
        raise ValueError('Unknown or modified validation protocol')
    observed = dependency_hashes()
    if protocol.get('files_sha256') != observed:
        changed = sorted(set(observed) | set(protocol.get('files_sha256', {})))
        changed = [name for name in changed if observed.get(name) != protocol.get('files_sha256', {}).get(name)]
        raise ValueError(f'Frozen source/configuration changed: {changed[:3]}')


def train_command(case, data_root, clip_path, model_dir):
    dataset = DATASETS[case['dataset']]
    variant = VARIANTS[case['variant']]
    return [
        sys.executable, '-u', str(ROOT / 'train.py'),
        '--root', str(data_root), '--seed', str(case['seed']),
        '--trainer', 'CoOp_TextDetail',
        '--dataset-config-file', str(ROOT / 'configs/datasets' / dataset['config']),
        '--config-file', str(ROOT / 'configs/repro/text_detail' / variant['config']),
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
    np.testing.assert_array_equal(arrays['Z'], arrays['S'] + arrays['R'])
    np.testing.assert_array_equal(arrays['F'], .5 * arrays['S'] + .5 * arrays['M'])
    return arrays, paths


def prepare_export_directory(output_dir):
    """Accept only the runtime subdirectory created while building the trainer."""
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
    if (cfg.TRAINER.NAME != 'CoOp_TextDetail' or cfg.DATASET.NAME != expected['name']
            or cfg.DATASET.NUM_SHOTS != 16 or cfg.OPTIM.MAX_EPOCH != 20 or cfg.SEED != seed
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
        'branch_semantics': {'Z': 'S+R', 'F': '0.5*(S+M)', 'R': 'not a classifier'},
    }
    atomic_json(output_dir / '导出.json', record)
    return record


def _case_root(output_root, case):
    return (Path(output_root) / DATASETS[case['dataset']]['label']
            / VARIANTS[case['variant']]['label'] / f"种子{case['seed']}")


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
    command = train_command(case, data_root, clip_path, model_dir)
    if checkpoint.is_file():
        status['events'].append({'case': case, 'step': '训练', 'status': 'resumed_existing_checkpoint'})
    else:
        model_dir.parent.mkdir(parents=True, exist_ok=True)
        log_path = root / '训练.log'
        env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'Dassl.pytorch'),
                   OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        with log_path.open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Training finished without the epoch-20 checkpoint: {checkpoint}')
        status['events'].append({'case': case, 'step': '训练', 'status': 'completed'})
    for split, label in (('base', '基础测试集'), ('novel', '新类别测试集')):
        folder = root / label
        if _export_valid(folder):
            status['events'].append({'case': case, 'step': label, 'status': 'resumed_existing_export'})
            continue
        if folder.exists():
            raise ValueError(f'Incomplete export exists and will not be overwritten automatically: {folder}')
        export_checkpoint(case['dataset'], case['seed'], model_dir, split, data_root, clip_path, folder)
        status['events'].append({'case': case, 'step': label, 'status': 'completed'})


def analyze_dataset(output_root, dataset_key, seeds):
    per_variant = {variant: [] for variant in VARIANTS}
    paired = {}
    for variant in VARIANTS:
        for seed in seeds:
            case = {'dataset': dataset_key, 'seed': seed, 'variant': variant}
            per_split = {}
            for split, label in (('base', '基础测试集'), ('novel', '新类别测试集')):
                record, arrays = load_export(_case_root(output_root, case) / label)
                pair_key = (seed, split)
                if pair_key not in paired:
                    paired[pair_key] = (record, arrays)
                else:
                    anchor_record, anchor_arrays = paired[pair_key]
                    if record['rows'] != anchor_record['rows'] or record['class_order'] != anchor_record['class_order']:
                        raise ValueError(f'Uniform/attentive metadata mismatch: seed={seed}, split={split}')
                    for name in ('labels', 'S', 'M', 'F'):
                        np.testing.assert_array_equal(arrays[name], anchor_arrays[name])
                computed = confidence_gated_fusion(arrays['S'], arrays['M'], arrays['R'])
                per_split[split] = {
                    'F': float(100 * (computed['simple_fusion'].argmax(1) == arrays['labels']).mean()),
                    'G': float(100 * (computed['gated_fusion'].argmax(1) == arrays['labels']).mean()),
                }
            per_variant[variant].append({
                'seed': seed,
                'base': per_split['base'], 'novel': per_split['novel'],
                'hm_F': harmonic_mean(per_split['base']['F'], per_split['novel']['F']),
                'hm_G': harmonic_mean(per_split['base']['G'], per_split['novel']['G']),
            })
    attentive, uniform = per_variant['attentive'], per_variant['uniform']
    residual_hm = [row['hm_G'] - row['hm_F'] for row in attentive]
    attention_hm = [left['hm_G'] - right['hm_G'] for left, right in zip(attentive, uniform)]
    novel_gain = float(np.mean([row['novel']['G'] - row['novel']['F'] for row in attentive]))
    result = {
        'dataset': dataset_key, 'seeds': list(seeds), 'formula': FORMULA,
        'variants': per_variant,
        'comparison': {
            '门控对简单融合_HM逐种子增益': residual_hm,
            '图片加权对均值描述_HM逐种子增益': attention_hm,
            '门控对简单融合_Novel平均增益': novel_gain,
        },
    }
    if len(seeds) == 1:
        result['decision'] = screening_decision(residual_hm[0], attention_hm[0], novel_gain)
    else:
        result['decision'] = confirmation_decision(residual_hm, attention_hm, novel_gain)
    return result


def write_dataset_report(path, analysis):
    decision = analysis['decision']
    comparison = analysis['comparison']
    lines = [
        f"# {DATASETS[analysis['dataset']]['label']} 置信度门控验证", '',
        f"结论：{'通过当前阶段' if decision['passes'] else '未通过当前阶段'}。", '',
        f"门控相对简单融合的 HM 逐种子增益：{comparison['门控对简单融合_HM逐种子增益']}。",
        f"图片加权相对均值描述的 HM 逐种子增益：{comparison['图片加权对均值描述_HM逐种子增益']}。",
        f"门控相对简单融合的 Novel 平均增益：{comparison['门控对简单融合_Novel平均增益']:+.4f} 个百分点。", '',
        '判定规则在实验前固定，未使用 +1 个百分点作为硬门槛。',
    ]
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
    status = {'state': 'running', 'protocol': str(protocol_path), 'gpu': torch.cuda.get_device_name(0),
              'started': time.time(), 'events': [], 'current_phase': None}
    atomic_json(status_path, status)
    try:
        phases = ('DTD单种子筛选', 'DTD三种子确认', 'UCF101三种子确认')
        for phase in phases:
            status['current_phase'] = phase; atomic_json(status_path, status)
            for case in [item for item in planned_cases() if item['phase'] == phase]:
                run_case(case, data_root, clip_path, output_root, status)
                atomic_json(status_path, status)
            dataset_key = 'dtd' if phase.startswith('DTD') else 'ucf101'
            seeds = (1,) if phase == 'DTD单种子筛选' else (1, 2, 3)
            analysis = analyze_dataset(output_root, dataset_key, seeds)
            result_dir = output_root / DATASETS[dataset_key]['label'] / ('单种子筛选' if len(seeds) == 1 else '三种子确认')
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


def analyze_command(args):
    seeds = tuple(int(value) for value in args.seeds.split(','))
    result = analyze_dataset(args.output_root, args.dataset, seeds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


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
        result = analyze_command(args)
    if args.command != 'analyze':
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
