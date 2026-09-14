#!/usr/bin/env python3
"""Fixed Aircraft text-detail pilots and staged three-seed confirmation.

Training requires an explicit `run` or `confirm` command and CUDA. `export`
only loads a checkpoint.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Dassl.pytorch'))

import numpy as np
import torch
from dassl.config import get_cfg_default
from dassl.data.datasets import build_dataset
from dassl.engine import build_trainer
from dassl.utils import set_random_seed
import train
from trainers.textconsistency_runtime import source_digest, sample_digest
from trainers.textconsistency_common import sha256_file

VARIANTS = ('attentive', 'centered')
REFERENCE = ROOT / 'reports/experiments/text-adamw-20260914'
REFERENCE_SOURCE = '143031dbd46e1c024bf62259d64906dc07ab209bea21e2faccb82dc8cef7d6e6'
REFERENCE_HASHES = {'classname/export.json': 'e497e13a3e40360b122c85423900ce491d8e7496682280f586a1783ede55c69a', 'classname/logits.npz': 'b6ec87f85d34aaf43f6cfbc055c6c6367ab4992cd6c1807ec730bae2c7a24a4a', 'description/export.json': 'd91e1c64a97e5cb0df21e3e8622efa13f6a118e574d77e1cf7cbab94e6ef3c26', 'description/logits.npz': 'a56ea5ba5c494308b228a1ec6fd3c241560c084c3ce3abc1a3f239b5434ddc97', 'contrast/export.json': '1e070f3398e6b6f67fbd14a48a9ae982846c0633006e80508e9f1b171d62cd41', 'contrast/logits.npz': '74630d8bcc146ea8253aad816b4de9eb6e38ecb8029726c1c1414909bb17f9cf'}
BRANCHES = ('S', 'M', 'R', 'Z', 'F')
MODEL_SOURCE_SHA = '9a50256a2067f586c21b46f6de86514daa28b3a916197df379c59e89a28a9433'
CONFIG_HASHES = {'attentive': '02262578008c0bbf47b5c7ba2c50d246bc3ff2a264cb58030d56b2b882d419c0', 'centered': 'be602a0006754136bb89a22d68479e66c432b48c81b7c531c42ba43d1ec5ea85'}
CLIP_SHA = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
TEMPLATE_SHA = 'd86833c2483c1d12907abeb6198412f1d0a020bd12999a3f170967a7f549e3b8'
SAMPLE_HASHES = {
    'train_samples_sha256': 'ae5851085ef5e74b7cb612cda7454fde300601ef14ad21b707bbbfbbd3c85e9a',
    'val_samples_sha256': '7ac44ab40ea24086902991743d3b51720e467204034efd2b5a98352416fd02bb',
}

# Fixed before the confirmation batch. Generated artifacts use Chinese names;
# framework-owned checkpoint/config names remain unchanged for compatibility.
CONFIRMATION_SOURCE_SHA = 'f1d22f0bfc16cf1a754b990da12b5526325ab49f192c68877dcf8f13c6237ba1'
CONFIRMATION_CONFIGS = {
    'classname': {'label': '类别名', 'trainer': 'CoOp_TextResidual_AdamW',
                  'path': ROOT / 'configs/repro/text_residual_adamw/classname.yaml',
                  'sha256': '2585fc2bd629a19aa97ca9554ef1d7abd4cb62f73458879a0b3e062436f6f147'},
    'uniform': {'label': '均值描述', 'trainer': 'CoOp_TextDetail',
                'path': ROOT / 'configs/repro/text_detail/uniform.yaml',
                'sha256': 'e5ea760fa821aed782a545f6ccb45008e118b41833e9ca919adbe9ba713de9bd'},
    'attentive': {'label': '图片加权', 'trainer': 'CoOp_TextDetail',
                  'path': ROOT / 'configs/repro/text_detail/attentive.yaml',
                  'sha256': '02262578008c0bbf47b5c7ba2c50d246bc3ff2a264cb58030d56b2b882d419c0'},
}
CONFIRMATION_SEEDS = (1, 2, 3)
CONFIRMATION_SAMPLE_HASHES = {
    1: 'ae5851085ef5e74b7cb612cda7454fde300601ef14ad21b707bbbfbbd3c85e9a',
    2: 'ae05002965cddf63fa83c5537d249e45f968af7683fa8b9b87b6dd048739585e',
    3: 'aa24df6792e0120acda0d6b5ed48e1e5c500586e1da207ac4fb095fd18a5ab8a',
}
CONFIRMATION_REFERENCE = ROOT / 'reports/experiments/text-detail-20260914/attentive'
CONFIRMATION_REFERENCE_HASHES = {
    'export.json': '788ae81e1c9038a2a55f5da039fb57884b7c0444bb8b1e416bbb1f02d9a9063b',
    'logits.npz': '50f0c2ac380f97874b5c065d069e525a6a59ccd71226bcdf7b835a4bb56392fd',
}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def confirmation_decision(hm_gains, novel_mean_gain):
    """Apply the preregistered three-seed confirmation rule."""
    values = [float(value) for value in hm_gains]
    novel_mean_gain = float(novel_mean_gain)
    if len(values) != 3 or not all(np.isfinite(values)) or not np.isfinite(novel_mean_gain):
        raise ValueError('Confirmation requires three finite HM gains and one finite Novel gain')
    mean_hm_gain = float(np.mean(values))
    positive_seeds = sum(value > 0 for value in values)
    criteria = {
        'mean_hm_gain_at_least_1pp': mean_hm_gain >= 1.,
        'positive_hm_gain_in_at_least_2_seeds': positive_seeds >= 2,
        'mean_novel_accuracy_not_lower': novel_mean_gain >= 0.,
    }
    return {'passes': all(criteria.values()), 'criteria': criteria,
            'mean_hm_gain_pp': mean_hm_gain, 'positive_seeds': positive_seeds,
            'mean_novel_gain_pp': novel_mean_gain}


def check_confirmation_freeze():
    if source_digest() != CONFIRMATION_SOURCE_SHA:
        raise ValueError('Confirmation model source changed after protocol freeze')
    for item in CONFIRMATION_CONFIGS.values():
        if sha256_file(item['path']) != item['sha256']:
            raise ValueError(f"Confirmation configuration changed: {item['path']}")
    for name, expected in CONFIRMATION_REFERENCE_HASHES.items():
        if sha256_file(CONFIRMATION_REFERENCE / name) != expected:
            raise ValueError('Frozen seed1 attentive development reference changed')


def confirmation_train_command(data, weights, model_dir, variant, seed):
    item = CONFIRMATION_CONFIGS[variant]
    return [sys.executable, str(ROOT / 'train.py'), '--root', str(data), '--seed', str(seed),
            '--trainer', item['trainer'], '--dataset-config-file',
            str(ROOT / 'configs/datasets/fgvc_aircraft.yaml'), '--config-file', str(item['path']),
            '--output-dir', str(model_dir), 'DATASET.SUBSAMPLE_CLASSES', 'base',
            'USE_CUDA', 'True', 'TRAINER.TEXTCONSISTENCY.DEVICE', 'cuda',
            'TRAINER.TEXTCONSISTENCY.CLIP_PATH', str(weights)]


@torch.inference_mode()
def collect_predictions(model, batches, device):
    arrays = {key: [] for key in ('labels', *BRANCHES)}
    paths = []
    model.eval()
    for batch in batches:
        outputs = model(batch['img'].to(device))
        labels = batch['label'].cpu().numpy()
        if labels.dtype != np.int64 or labels.ndim != 1:
            raise ValueError('Malformed labels')
        arrays['labels'].append(labels)
        for key in BRANCHES:
            value = outputs[key].cpu().numpy()
            if value.ndim != 2 or len(value) != len(labels) or not np.isfinite(value).all():
                raise ValueError(f'Malformed {key} logits')
            arrays[key].append(value)
        paths.extend(str(path) for path in batch['impath'])
    if not paths or len(paths) != len(set(paths)):
        raise ValueError('Empty or duplicate export rows')
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    np.testing.assert_array_equal(arrays['Z'], arrays['S'] + arrays['R'])
    np.testing.assert_array_equal(arrays['F'], .5 * arrays['S'] + .5 * arrays['M'])
    return arrays, paths


def export_confirmation(model_dir, out, split, data, weights):
    model_dir, out = Path(model_dir).resolve(), Path(out).resolve()
    if split not in ('base_val', 'base_test', 'novel_test'):
        raise ValueError(f'Unknown confirmation export split: {split}')
    if out.exists():
        raise FileExistsError(f'Use a fresh export directory: {out}')
    cfg = get_cfg_default(); train.extend_cfg(cfg)
    cfg.merge_from_file(str(model_dir / 'effective_config.yaml'))
    if (cfg.DATASET.NAME != 'FGVCAircraft' or cfg.DATASET.NUM_SHOTS != 16
            or cfg.OPTIM.MAX_EPOCH != 20 or cfg.SEED not in CONFIRMATION_SEEDS):
        raise ValueError('Confirmation exporter only accepts the frozen Aircraft protocol')
    target = 'new' if split == 'novel_test' else 'base'
    cfg.DATASET.ROOT = str(data); cfg.DATASET.SUBSAMPLE_CLASSES = target
    cfg.TEST.SPLIT = 'val' if split == 'base_val' else 'test'
    cfg.OUTPUT_DIR = str(out / '运行信息')
    cfg.USE_CUDA = True; cfg.TRAINER.TEXTCONSISTENCY.DEVICE = 'cuda'
    cfg.TRAINER.TEXTCONSISTENCY.CLIP_PATH = str(weights)
    cfg.freeze()
    set_random_seed(cfg.SEED); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    trainer = build_trainer(cfg)
    trainer.load_model(model_dir, epoch=20)
    items = trainer.dm.dataset.val if split == 'base_val' else trainer.dm.dataset.test
    loader = trainer.val_loader if split == 'base_val' else trainer.test_loader
    arrays, paths = collect_predictions(trainer.model, loader, trainer.device)
    if arrays['S'].shape[1] != 50 or len(arrays['S']) != len(items):
        raise ValueError('Expected all rows and 50 classes for the requested Aircraft split')
    rows = [{'sample_id': str(Path(path).resolve().relative_to(Path(data).resolve())),
             'label': int(label)} for path, label in zip(paths, arrays['labels'])]
    expected_rows = [{'sample_id': str(Path(item.impath).resolve().relative_to(Path(data).resolve())),
                      'label': int(item.label)} for item in items]
    if rows != expected_rows:
        raise ValueError('Export order differs from the dataset order')
    out.mkdir(parents=True, exist_ok=True)
    predictions = out / '预测.npz'
    np.savez_compressed(predictions, **arrays)
    record = {
        'status': 'exported', 'split': split, 'seed': int(cfg.SEED), 'rows': rows,
        'class_order': list(trainer.dm.dataset.classnames), 'provenance': trainer.run_provenance,
        'checkpoint': trainer.loaded_checkpoints, 'predictions_sha256': sha256_file(predictions),
        'correct': {key: int((arrays[key].argmax(1) == arrays['labels']).sum())
                    for key in BRANCHES if key != 'R'},
        'accuracy': {key: float(100 * (arrays[key].argmax(1) == arrays['labels']).mean())
                     for key in BRANCHES if key != 'R'},
        'branch_semantics': {'Z': 'S+R', 'F': '.5*S+.5*M', 'R': 'not a standalone classifier'},
    }
    write_json(out / '导出.json', record)
    return record


def load_confirmation_export(folder):
    folder = Path(folder)
    record = json.loads((folder / '导出.json').read_text())
    predictions = folder / '预测.npz'
    if sha256_file(predictions) != record['predictions_sha256']:
        raise ValueError(f'Prediction archive changed: {predictions}')
    with np.load(predictions, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return record, arrays


def export(args):
    model_dir, out = Path(args.model_dir).resolve(), Path(args.output_dir).resolve()
    if out.exists():
        raise FileExistsError(f'Use a fresh export directory: {out}')
    cfg = get_cfg_default(); train.extend_cfg(cfg)
    cfg.merge_from_file(str(model_dir / 'effective_config.yaml'))
    if (cfg.TRAINER.NAME != 'CoOp_TextDetail' or cfg.SEED != 1
            or cfg.DATASET.NAME != 'FGVCAircraft' or cfg.DATASET.SUBSAMPLE_CLASSES != 'base'
            or cfg.DATASET.NUM_SHOTS != 16 or cfg.OPTIM.MAX_EPOCH != 20
            or cfg.TRAINER.TEXTCONSISTENCY.TEXT_RESIDUAL_ADAMW_LR != .001):
        raise ValueError('This exporter is restricted to the fixed Aircraft pilot')
    cfg.OUTPUT_DIR = str(out / 'runtime')
    cfg.USE_CUDA = True; cfg.TRAINER.TEXTCONSISTENCY.DEVICE = 'cuda'
    cfg.freeze()
    set_random_seed(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    trainer = build_trainer(cfg)
    trainer.load_model(model_dir, epoch=20)
    for key, expected in SAMPLE_HASHES.items():
        if trainer.run_provenance[key] != expected:
            raise ValueError(f'Aircraft pilot split differs from fixed anchor: {key}')
    arrays, paths = collect_predictions(trainer.model, trainer.val_loader, trainer.device)
    if arrays['S'].shape != (200, 50):
        raise ValueError('Expected 200 Base validation images and 50 classes')
    rows = [{'sample_id': str(Path(path).resolve().relative_to(Path(cfg.DATASET.ROOT).resolve())),
             'label': int(label)} for path, label in zip(paths, arrays['labels'])]
    expected_rows = [{'sample_id': str(Path(item.impath).resolve().relative_to(Path(cfg.DATASET.ROOT).resolve())),
                      'label': int(item.label)} for item in trainer.dm.dataset.val]
    if rows != expected_rows:
        raise ValueError('Validation output order differs from the dataset')
    np.savez_compressed(out / 'logits.npz', **arrays)
    result = {'status': 'exported', 'split': 'base_val', 'rows': rows,
              'class_order': list(trainer.dm.dataset.classnames),
              'provenance': trainer.run_provenance, 'checkpoint': trainer.loaded_checkpoints,
              'logits_sha256': sha256_file(out / 'logits.npz'),
              'correct': {key: int((arrays[key].argmax(1) == arrays['labels']).sum())
                          for key in BRANCHES if key != 'R'},
              'neighbor_indices': trainer.model.neighbor_indices, 'branch_semantics': {'Z': 'S+clip_scale*einsum(image,per_image_text_delta)', 'F': '.5*S+.5*M', 'R': 'not a standalone classifier'}}
    write_json(out / 'export.json', result)
    return result


def analyze(output):
    arrays, records = {}, {}
    folders = {name: REFERENCE / name for name in ('classname', 'description', 'contrast')}
    folders.update({name: output / name / 'export' for name in VARIANTS})
    for name, folder in folders.items():
        record = json.loads((folder / 'export.json').read_text())
        if sha256_file(folder / 'logits.npz') != record['logits_sha256']:
            raise ValueError('Export file changed')
        expected_source = MODEL_SOURCE_SHA if name in VARIANTS else REFERENCE_SOURCE
        if record['provenance']['code_sha256'] != expected_source:
            raise ValueError('Unexpected model source for fixed reference or new candidate')
        with np.load(folder / 'logits.npz', allow_pickle=False) as z:
            arrays[name] = {k: z[k].copy() for k in z.files}
        records[name] = record
    for name, record in records.items():
        if record['rows'] != records['classname']['rows'] or record['class_order'] != records['classname']['class_order']:
            raise ValueError('Different validation samples/classes')
        for branch in ('S', 'M', 'F'):
            np.testing.assert_array_equal(arrays[name][branch], arrays['classname'][branch])
    pred = lambda name, branch: arrays[name][branch].argmax(1) == arrays[name]['labels']
    def compare(candidate, controls):
        result = {}; proposed = pred(candidate, 'Z')
        for name, branch in controls:
            reference = pred(name, branch); delta = int(proposed.sum()) - int(reference.sum())
            result[f'{candidate}.Z - {name}.{branch}'] = {'gain_pp': delta / 2,
                'corrected': int((proposed & ~reference).sum()), 'harmed': int((~proposed & reference).sum()),
                'passes_1pp': delta >= 2}
        return result
    controls = [('classname', 'S'), ('classname', 'F'), ('classname', 'Z'), ('description', 'Z'), ('contrast', 'Z')]
    primary = compare('centered', controls + [('attentive', 'Z')])
    secondary = compare('attentive', controls)
    return {'status': 'completed', 'scope': 'Aircraft seed1 Base validation development only',
        'n': 200, 'correct': {name: {branch: int(pred(name, branch).sum()) for branch in ('S', 'M', 'Z', 'F')}
                            for name in folders}, 'comparisons': primary,
        'continuation_signal': all(x['passes_1pp'] for x in primary.values()),
        'attentive_comparisons': secondary, 'attentive_signal': all(x['passes_1pp'] for x in secondary.values()),
        'reference_source_sha256': REFERENCE_SOURCE, 'new_source_sha256': MODEL_SOURCE_SHA,
        'reference_logits_exact_S_M_F': True, 'automatic_expansion': False}


def analyze_uniform_gate(output):
    reference_record = json.loads((CONFIRMATION_REFERENCE / 'export.json').read_text())
    with np.load(CONFIRMATION_REFERENCE / 'logits.npz', allow_pickle=False) as archive:
        reference = {name: archive[name].copy() for name in archive.files}
    folder = output / CONFIRMATION_CONFIGS['uniform']['label'] / '种子1' / '基础验证集'
    uniform_record, uniform = load_confirmation_export(folder)
    if (reference_record['rows'] != uniform_record['rows']
            or reference_record['class_order'] != uniform_record['class_order']):
        raise ValueError('Seed1 development control uses different samples or classes')
    for branch in ('S', 'M', 'F'):
        np.testing.assert_array_equal(reference[branch], uniform[branch])
    attentive_correct = int((reference['Z'].argmax(1) == reference['labels']).sum())
    uniform_correct = int((uniform['Z'].argmax(1) == uniform['labels']).sum())
    gain = (attentive_correct - uniform_correct) / 2.
    return {
        'scope': 'Aircraft seed1 Base validation development gate', 'rows': len(uniform['labels']),
        'attentive_correct': attentive_correct, 'attentive_accuracy': attentive_correct / 2.,
        'uniform_correct': uniform_correct, 'uniform_accuracy': uniform_correct / 2.,
        'attentive_minus_uniform_pp': gain, 'threshold_pp': 1., 'passes': gain >= 1.,
        'same_rows_classes_and_exact_S_M_F': True,
        'reference_source_sha256': reference_record['provenance']['code_sha256'],
        'uniform_source_sha256': uniform_record['provenance']['code_sha256'],
    }


def harmonic_mean(base, novel):
    return 0. if base + novel == 0 else 2 * base * novel / (base + novel)


def analyze_confirmation(output, gate):
    exports, arrays = {}, {}
    split_folders = {'base': '基础测试集', 'novel': '新类别测试集'}
    for seed in CONFIRMATION_SEEDS:
        for variant, item in CONFIRMATION_CONFIGS.items():
            for split, split_folder in split_folders.items():
                key = (seed, variant, split)
                folder = output / item['label'] / f'种子{seed}' / split_folder
                exports[key], arrays[key] = load_confirmation_export(folder)
                record = exports[key]
                if (record['seed'] != seed or record['provenance']['code_sha256'] != CONFIRMATION_SOURCE_SHA
                        or record['provenance']['trainable_parameters'] != 67584):
                    raise ValueError('Unexpected seed, source, or parameter count in confirmation export')
        for split in split_folders:
            anchor_record, anchor_arrays = exports[(seed, 'classname', split)], arrays[(seed, 'classname', split)]
            for variant in ('uniform', 'attentive'):
                record, values = exports[(seed, variant, split)], arrays[(seed, variant, split)]
                if record['rows'] != anchor_record['rows'] or record['class_order'] != anchor_record['class_order']:
                    raise ValueError('Paired methods use different test rows or class order')
                for branch in ('labels', 'S', 'M', 'F'):
                    np.testing.assert_array_equal(values[branch], anchor_arrays[branch])
    per_seed = {}
    for seed in CONFIRMATION_SEEDS:
        per_seed[str(seed)] = {}
        for variant, item in CONFIRMATION_CONFIGS.items():
            base = exports[(seed, variant, 'base')]['accuracy']['Z']
            novel = exports[(seed, variant, 'novel')]['accuracy']['Z']
            per_seed[str(seed)][variant] = {'label': item['label'], 'base': base, 'novel': novel,
                                            'hm': harmonic_mean(base, novel)}
    summary = {}
    for variant, item in CONFIRMATION_CONFIGS.items():
        summary[variant] = {'label': item['label']}
        for metric in ('base', 'novel', 'hm'):
            values = [per_seed[str(seed)][variant][metric] for seed in CONFIRMATION_SEEDS]
            summary[variant][metric] = {'mean': float(np.mean(values)),
                                         'sample_std': float(np.std(values, ddof=1)), 'values': values}
    hm_gains = [per_seed[str(seed)]['attentive']['hm'] - per_seed[str(seed)]['uniform']['hm']
                for seed in CONFIRMATION_SEEDS]
    novel_gain = summary['attentive']['novel']['mean'] - summary['uniform']['novel']['mean']
    decision = confirmation_decision(hm_gains, novel_gain)
    comparisons = {}
    for control in ('uniform', 'classname'):
        comparisons[f'attentive_minus_{control}'] = {
            metric: {
                'mean_gain_pp': summary['attentive'][metric]['mean'] - summary[control][metric]['mean'],
                'paired_gains_pp': [per_seed[str(seed)]['attentive'][metric]
                                    - per_seed[str(seed)][control][metric] for seed in CONFIRMATION_SEEDS],
            } for metric in ('base', 'novel', 'hm')
        }
    return {
        'status': 'completed', 'scope': 'Aircraft 16-shot Base/Novel test, seeds 1/2/3',
        'development_gate': gate, 'per_seed': per_seed, 'summary': summary,
        'attentive_vs_uniform_decision': decision, 'comparisons': comparisons,
        'same_rows_classes_and_exact_S_M_F_within_each_seed': True,
        'trainable_parameters_each_model': 67584, 'code_sha256': CONFIRMATION_SOURCE_SHA,
        'checkpoint_rule': 'epoch 20 fixed before testing; no test-selected checkpoint or retry',
    }


def write_confirmation_report(path, analysis):
    gate = analysis['development_gate']
    lines = [
        '# 图片加权独立作用与新类别确认结果', '',
        f"seed1 Base开发验证集：图片加权 {gate['attentive_accuracy']:.1f}%，均值描述 "
        f"{gate['uniform_accuracy']:.1f}%，差值 {gate['attentive_minus_uniform_pp']:+.1f}pp。",
        '',
    ]
    if analysis.get('status') != 'completed':
        lines.extend([
            '**均值对照门槛未通过，因此按预登记规则没有启动多种子和Novel训练。**', '',
            '这表示现有证据不能把收益单独归因于图片加权。', '',
        ])
    else:
        decision = analysis['attentive_vs_uniform_decision']
        verdict = '通过' if decision['passes'] else '未通过'
        lines.extend([
            f"**三种子确认判定：{verdict}。**", '',
            '| 方法 | Base均值±标准差 | Novel均值±标准差 | HM均值±标准差 |',
            '|---|---:|---:|---:|',
        ])
        for variant in ('classname', 'uniform', 'attentive'):
            row = analysis['summary'][variant]
            lines.append(f"| {row['label']} | {row['base']['mean']:.2f}±{row['base']['sample_std']:.2f} | "
                         f"{row['novel']['mean']:.2f}±{row['novel']['sample_std']:.2f} | "
                         f"{row['hm']['mean']:.2f}±{row['hm']['sample_std']:.2f} |")
        lines.extend(['', '| 种子 | 方法 | Base | Novel | HM |', '|---:|---|---:|---:|---:|'])
        for seed in CONFIRMATION_SEEDS:
            for variant in ('classname', 'uniform', 'attentive'):
                row = analysis['per_seed'][str(seed)][variant]
                lines.append(f"| {seed} | {row['label']} | {row['base']:.2f} | {row['novel']:.2f} | {row['hm']:.2f} |")
        comparison = analysis['comparisons']['attentive_minus_uniform']
        lines.extend([
            '', '## 固定判定', '',
            f"- 图片加权相对均值描述的三种子HM平均差：{decision['mean_hm_gain_pp']:+.2f}pp。",
            f"- HM为正的种子数：{decision['positive_seeds']}/3。",
            f"- Novel平均差：{decision['mean_novel_gain_pp']:+.2f}pp。",
            f"- Base平均差：{comparison['base']['mean_gain_pp']:+.2f}pp。",
            '',
        ])
    lines.extend([
        '## 实验边界', '',
        '三组都使用相同的学生提示、512→64→512共享残差参数、优化器、训练轮数、数据增强和样本。'
        '均值描述与图片加权只改变类内描述权重，参数量均为67,584；同一种子内S/M/F逐元素一致。', '',
        'Base/Novel使用官方测试划分和固定第20轮，未按测试结果挑轮次、调权重或自动重试。'
        '这是Aircraft单数据集确认；无论是否通过，都不能仅凭这一批声称达到投稿录用条件。', '',
    ])
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def run_confirmation(args):
    data, weights, output = [Path(value).resolve() for value in
                             (args.data_root, args.clip_path, args.output_root)]
    if output.exists():
        raise FileExistsError(f'No resume or overwrite: {output}')
    for protected in (ROOT, data, weights):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('Keep confirmation output separate from code, data and CLIP weights')
    if not torch.cuda.is_available():
        raise RuntimeError('Confirmation training requires an available CUDA GPU')
    if sha256_file(weights) != CLIP_SHA or sha256_file(ROOT / 'templates/airplane_prompts_full.json') != TEMPLATE_SHA:
        raise ValueError('CLIP or Aircraft description asset differs from the frozen protocol')
    for seed in CONFIRMATION_SEEDS:
        if not (data / f'fgvc_aircraft/split_fewshot/shot_16-seed_{seed}.pkl').is_file():
            raise FileNotFoundError(f'Missing prepared Aircraft seed {seed} few-shot cache')
    check_confirmation_freeze()
    preflight = {}
    for seed in CONFIRMATION_SEEDS:
        cfg = get_cfg_default(); train.extend_cfg(cfg)
        cfg.merge_from_file(str(ROOT / 'configs/datasets/fgvc_aircraft.yaml'))
        cfg.merge_from_file(str(CONFIRMATION_CONFIGS['uniform']['path']))
        cfg.DATASET.ROOT = str(data); cfg.SEED = seed
        dataset = build_dataset(cfg)
        digest = sample_digest(dataset.train_x, data)
        if len(dataset.train_x) != 800 or len(dataset.val) != 200 or len(dataset.classnames) != 50:
            raise ValueError('Prepared Base split shape differs from the fixed Aircraft protocol')
        if digest != CONFIRMATION_SAMPLE_HASHES[seed]:
            raise ValueError(f'Aircraft seed {seed} training split changed')
        preflight[str(seed)] = {'train_rows': 800, 'val_rows': 200,
                                'base_test_rows': len(dataset.test), 'train_sha256': digest}
    output.mkdir(parents=True)
    protocol = {
        'purpose': 'isolate image weighting, then confirm Base/Novel across three seeds',
        'dataset': 'FGVCAircraft', 'shots': 16, 'seeds': list(CONFIRMATION_SEEDS),
        'variants': {name: {'label': item['label'], 'trainer': item['trainer'],
                            'config_sha256': item['sha256']} for name, item in CONFIRMATION_CONFIGS.items()},
        'development_gate': 'seed1 Base validation: attentive - uniform >= 1.0pp',
        'expansion_rule': 'run all three variants on seeds 1/2/3 only if development gate passes',
        'confirmation_rule': {
            'mean_paired_hm_gain_attentive_minus_uniform_pp': '>= 1.0',
            'positive_hm_seed_count': '>= 2 of 3',
            'mean_novel_gain_attentive_minus_uniform_pp': '>= 0.0',
        },
        'fixed_epoch': 20, 'automatic_retry': False, 'preflight': preflight,
        'code_sha256': CONFIRMATION_SOURCE_SHA,
    }
    write_json(output / '实验协议.json', protocol)
    status = {'state': 'running', 'current_stage': None, 'stages': [], 'expanded': False}
    write_json(output / '执行状态.json', status)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / 'Dassl.pytorch'), env.get('PYTHONPATH', '')])
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    started = time.monotonic(); timeout = 5400.

    def remaining():
        value = timeout - (time.monotonic() - started)
        if value <= 0:
            raise TimeoutError('Confirmation wallclock cap reached')
        return value

    def train_stage(variant, seed):
        check_confirmation_freeze()
        item = CONFIRMATION_CONFIGS[variant]
        folder = output / item['label'] / f'种子{seed}'
        model_dir = folder / '模型'
        status['current_stage'] = f"{item['label']}-种子{seed}-训练"
        write_json(output / '执行状态.json', status)
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / '训练.log').open('w') as log:
            subprocess.run(confirmation_train_command(data, weights, model_dir, variant, seed),
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=remaining())
        status['stages'].append({'stage': status['current_stage'], 'status': 'passed'})
        write_json(output / '执行状态.json', status)
        return model_dir

    def export_stage(variant, seed, split, folder_name):
        check_confirmation_freeze()
        item = CONFIRMATION_CONFIGS[variant]
        folder = output / item['label'] / f'种子{seed}'
        status['current_stage'] = f"{item['label']}-种子{seed}-{folder_name}"
        write_json(output / '执行状态.json', status)
        with (folder / f'{folder_name}导出.log').open('w') as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                export_confirmation(folder / '模型', folder / folder_name, split, data, weights)
        status['stages'].append({'stage': status['current_stage'], 'status': 'passed'})
        write_json(output / '执行状态.json', status)

    try:
        train_stage('uniform', 1)
        export_stage('uniform', 1, 'base_val', '基础验证集')
        gate = analyze_uniform_gate(output)
        write_json(output / '均值对照分析.json', gate)
        if not gate['passes']:
            analysis = {'status': 'development_gate_failed', 'development_gate': gate,
                        'automatic_expansion': False}
            write_json(output / '结果分析.json', analysis)
            write_confirmation_report(output / '结果报告.md', analysis)
            status.update(state='completed', current_stage='均值对照未通过', expanded=False,
                          conclusion='image weighting not independently confirmed')
            return status
        status['expanded'] = True
        write_json(output / '执行状态.json', status)
        for seed in CONFIRMATION_SEEDS:
            for variant in ('classname', 'uniform', 'attentive'):
                if not (variant == 'uniform' and seed == 1):
                    train_stage(variant, seed)
                export_stage(variant, seed, 'base_test', '基础测试集')
                export_stage(variant, seed, 'novel_test', '新类别测试集')
        analysis = analyze_confirmation(output, gate)
        write_json(output / '结果分析.json', analysis)
        write_confirmation_report(output / '结果报告.md', analysis)
        status.update(state='completed', current_stage='全部完成',
                      conclusion=('image weighting confirmed' if analysis['attentive_vs_uniform_decision']['passes']
                                  else 'image weighting not confirmed'))
    except BaseException as error:
        status.update(state='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status['elapsed_seconds'] = time.monotonic() - started
        write_json(output / '执行状态.json', status)
    return status


def run(args):
    data, weights, output = [Path(value).resolve() for value in (args.data_root, args.clip_path, args.output_root)]
    if output.exists():
        raise FileExistsError(f'No resume or overwrite: {output}')
    for protected in (ROOT, data, weights):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('Keep pilot output separate from code, data and CLIP weights')
    if not torch.cuda.is_available():
        raise RuntimeError('Pilot training requires an available CUDA GPU')
    if sha256_file(weights) != CLIP_SHA or sha256_file(ROOT / 'templates/airplane_prompts_full.json') != TEMPLATE_SHA:
        raise ValueError('CLIP or description asset differs from the existing pilot')
    if not (data / 'fgvc_aircraft/split_fewshot/shot_16-seed_1.pkl').is_file():
        raise FileNotFoundError('Reuse the prepared seed1 few-shot cache before training')
    configs = {name: ROOT / f'configs/repro/text_detail/{name}.yaml' for name in VARIANTS}
    if source_digest() != MODEL_SOURCE_SHA or any(sha256_file(path) != CONFIG_HASHES[name]
                                                 for name, path in configs.items()):
        raise ValueError('Frozen correction pilot code or configuration differs; do not silently change the protocol')
    if any(sha256_file(REFERENCE / path) != value for path, value in REFERENCE_HASHES.items()):
        raise ValueError('Historical reference files changed')
    # Read only the small split/cache metadata; do not decode or rehash all images.
    check = get_cfg_default(); train.extend_cfg(check)
    check.merge_from_file(str(ROOT / 'configs/datasets/fgvc_aircraft.yaml'))
    check.merge_from_file(str(configs['centered']))
    check.DATASET.ROOT = str(data); check.SEED = 1
    dataset = build_dataset(check)
    if len(dataset.train_x) != 800 or len(dataset.val) != 200 or len(dataset.classnames) != 50:
        raise ValueError('Unexpected Aircraft Base few-shot split sizes')
    for split, items in (('train', dataset.train_x), ('val', dataset.val)):
        if sample_digest(items, data) != SAMPLE_HASHES[split + '_samples_sha256']:
            raise ValueError(f'Unexpected few-shot {split} split; fix before spending GPU time')
    input_hashes = {str(path): sha256_file(path) for path in
                    (*configs.values(), ROOT / 'configs/datasets/fgvc_aircraft.yaml', Path(__file__).resolve(),
                     data / 'fgvc_aircraft/split_fewshot/shot_16-seed_1.pkl', *(REFERENCE / p for p in REFERENCE_HASHES))}
    source = source_digest()
    output.mkdir(parents=True)
    stages = []
    for variant in VARIANTS:
        model, exported = output / variant / 'model', output / variant / 'export'
        command = [sys.executable, '-u', str(ROOT / 'train.py'), '--root', str(data), '--seed', '1',
                   '--dataset-config-file', str(ROOT / 'configs/datasets/fgvc_aircraft.yaml'),
                   '--config-file', str(configs[variant]), '--output-dir', str(model),
                   'USE_CUDA', 'True', 'TRAINER.TEXTCONSISTENCY.DEVICE', 'cuda',
                   'TRAINER.TEXTCONSISTENCY.CLIP_PATH', str(weights), 'TEST.NO_TEST', 'True']
        stages.extend([(variant + '-train', command), (variant + '-export',
                       [sys.executable, '-u', str(Path(__file__).resolve()), 'export',
                        '--model-dir', str(model), '--output-dir', str(exported)])])
    status = {'state': 'running', 'source_sha256': source, 'input_hashes': input_hashes,
              'gpu': torch.cuda.get_device_name(0), 'stages': [], 'planned_commands': stages,
              'max_training_runs': 2, 'wallclock_cap_seconds': 1800, 'minimum_gain_pp': 1.0}
    write_json(output / 'status.json', status)
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'Dassl.pytorch'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    start = time.monotonic()
    try:
        for name, command in stages:
            if source_digest() != source or any(sha256_file(Path(path)) != digest for path, digest in input_hashes.items()):
                raise ValueError('Source or fixed configuration changed during the pilot')
            remaining = 1800 - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError('Pilot wallclock cap reached')
            status['current_stage'] = name; write_json(output / 'status.json', status)
            print(f'Starting {name}; log: {output / (name + ".log")}', flush=True)
            with (output / (name + '.log')).open('w') as log:
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=remaining)
            status['stages'].append({'name': name, 'status': 'passed'})
            write_json(output / 'status.json', status)
        result = analyze(output)
        write_json(output / 'analysis.json', result)
        status.update(state='completed', continuation_signal=result['continuation_signal'])
    except BaseException as error:
        status.update(state='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status['elapsed_seconds'] = time.monotonic() - start
        write_json(output / 'status.json', status)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    launch = commands.add_parser('run')
    for option in ('data-root', 'clip-path', 'output-root'):
        launch.add_argument('--' + option, required=True)
    extract = commands.add_parser('export')
    extract.add_argument('--model-dir', required=True); extract.add_argument('--output-dir', required=True)
    confirm = commands.add_parser('confirm')
    for option in ('data-root', 'clip-path', 'output-root'):
        confirm.add_argument('--' + option, required=True)
    args = parser.parse_args()
    actions = {'run': run, 'export': export, 'confirm': run_confirmation}
    print(json.dumps(actions[args.command](args), indent=2, ensure_ascii=False))
