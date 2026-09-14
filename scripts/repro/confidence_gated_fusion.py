#!/usr/bin/env python3
"""Apply and audit the frozen entropy-confidence residual gate.

The formula has no fitted coefficient:
    F = 0.5 * (S + M)
    q = 1 - H(softmax(S)) / log(C)
    G = F + q * R

This script consumes exported logits only. It does not alter checkpoints or
training and deliberately labels the Aircraft result as post-hoc exploration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


METHODS = ('类别名', '均值描述', '图片加权')
SEEDS = (1, 2, 3)
SPLITS = {'base': '基础测试集', 'novel': '新类别测试集'}
REQUIRED_ARRAYS = {'labels', 'S', 'M', 'R', 'Z', 'F'}
FORMULA = {
    'simple_fusion': 'F = 0.5 * (S + M)',
    'confidence': 'q = 1 - H(softmax(S)) / log(C)',
    'gated_fusion': 'G = F + q * R',
    'fitted_hyperparameters': [],
}


def _validate_logits(name, value):
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f'{name} logits must be a two-dimensional array with at least two classes')
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError(f'{name} logits must be finite floating-point values')
    return array


def normalized_entropy_confidence(student_logits):
    """Return q in [0, 1] using normalized predictive entropy."""
    student = _validate_logits('student', student_logits)
    work = student.astype(np.float64, copy=False)
    shifted = work - work.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    entropy = -(probabilities * log_probabilities).sum(axis=1)
    confidence = np.clip(1. - entropy / math.log(student.shape[1]), 0., 1.)
    return confidence.astype(student.dtype, copy=False)


def confidence_gated_fusion(student_logits, description_mean_logits, residual_logits):
    """Compute the fixed simple fusion and confidence-gated residual fusion."""
    student = _validate_logits('student', student_logits)
    description = _validate_logits('description mean', description_mean_logits)
    residual = _validate_logits('residual', residual_logits)
    if student.shape != description.shape or student.shape != residual.shape:
        raise ValueError('student, description mean, and residual logits must have the same shape')
    dtype = np.result_type(student.dtype, description.dtype, residual.dtype)
    simple = (.5 * student.astype(dtype, copy=False) + .5 * description.astype(dtype, copy=False))
    confidence = normalized_entropy_confidence(student).astype(dtype, copy=False)
    gated = simple + confidence[:, None] * residual.astype(dtype, copy=False)
    if not np.isfinite(gated).all():
        raise ValueError('gated fusion logits must be finite')
    return {'simple_fusion': simple, 'confidence': confidence, 'gated_fusion': gated}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_export(folder):
    folder = Path(folder)
    record_path, archive_path = folder / '导出.json', folder / '预测.npz'
    record = json.loads(record_path.read_text())
    expected_hash = record.get('predictions_sha256')
    if expected_hash != sha256_file(archive_path):
        raise ValueError(f'Prediction archive hash mismatch: {archive_path}')
    with np.load(archive_path, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_ARRAYS:
            raise ValueError(f'Prediction archive has missing or extra arrays: {archive_path}')
        arrays = {name: archive[name].copy() for name in archive.files}
    labels = arrays['labels']
    if labels.ndim != 1 or labels.dtype != np.int64 or len(labels) == 0:
        raise ValueError(f'Malformed labels: {archive_path}')
    for name in REQUIRED_ARRAYS - {'labels'}:
        value = _validate_logits(name, arrays[name])
        if len(value) != len(labels):
            raise ValueError(f'Logit row count differs from labels: {archive_path}')
    np.testing.assert_array_equal(arrays['Z'], arrays['S'] + arrays['R'])
    np.testing.assert_array_equal(arrays['F'], .5 * arrays['S'] + .5 * arrays['M'])
    if len(record.get('rows', [])) != len(labels) or len(record.get('class_order', [])) != arrays['S'].shape[1]:
        raise ValueError(f'Export metadata does not match prediction dimensions: {folder}')
    return record, arrays


def accuracy(logits, labels):
    return float(100. * (np.asarray(logits).argmax(1) == labels).mean())


def harmonic_mean(base, novel):
    return float(2. * base * novel / (base + novel)) if base + novel else 0.


def _seed_metrics(per_split):
    base, novel = per_split['base'], per_split['novel']
    return {
        'base_accuracy': base['accuracy'],
        'novel_accuracy': novel['accuracy'],
        'harmonic_mean': harmonic_mean(base['accuracy'], novel['accuracy']),
        'confidence': {
            'base_mean': base['confidence_mean'],
            'novel_mean': novel['confidence_mean'],
        },
    }


def _aggregate(seeds):
    base = [item['base_accuracy'] for item in seeds]
    novel = [item['novel_accuracy'] for item in seeds]
    hm = [item['harmonic_mean'] for item in seeds]
    return {
        'base_mean': float(np.mean(base)),
        'novel_mean': float(np.mean(novel)),
        'seed_harmonic_means': hm,
        'mean_seed_harmonic_mean': float(np.mean(hm)),
        'harmonic_mean_of_mean_accuracies': harmonic_mean(float(np.mean(base)), float(np.mean(novel))),
    }


def analyze(input_root):
    input_root = Path(input_root).resolve()
    raw, records = {}, {}
    for method in METHODS:
        raw[method], records[method] = {}, {}
        for seed in SEEDS:
            raw[method][seed], records[method][seed] = {}, {}
            for split, folder in SPLITS.items():
                record, arrays = load_export(input_root / method / f'种子{seed}' / folder)
                raw[method][seed][split] = arrays
                records[method][seed][split] = record

    # Comparisons are paired only after proving identical sample/class order.
    for seed in SEEDS:
        for split in SPLITS:
            anchor_record = records[METHODS[0]][seed][split]
            anchor_labels = raw[METHODS[0]][seed][split]['labels']
            for method in METHODS[1:]:
                record = records[method][seed][split]
                if record['rows'] != anchor_record['rows'] or record['class_order'] != anchor_record['class_order']:
                    raise ValueError(f'Paired export metadata mismatch: seed={seed}, split={split}, method={method}')
                np.testing.assert_array_equal(raw[method][seed][split]['labels'], anchor_labels)

    metrics = {}
    for method in METHODS:
        metrics[method] = {}
        for branch in ('simple_fusion', 'ungated_residual', 'confidence_gated'):
            seed_rows = []
            for seed in SEEDS:
                split_rows = {}
                for split in SPLITS:
                    arrays = raw[method][seed][split]
                    computed = confidence_gated_fusion(arrays['S'], arrays['M'], arrays['R'])
                    logits = {'simple_fusion': computed['simple_fusion'],
                              'ungated_residual': arrays['F'] + arrays['R'],
                              'confidence_gated': computed['gated_fusion']}[branch]
                    split_rows[split] = {
                        'accuracy': accuracy(logits, arrays['labels']),
                        'correct': int((logits.argmax(1) == arrays['labels']).sum()),
                        'total': int(len(arrays['labels'])),
                        'confidence_mean': float(computed['confidence'].mean()),
                    }
                seed_rows.append(_seed_metrics(split_rows))
            metrics[method][branch] = {'seeds': seed_rows, 'aggregate': _aggregate(seed_rows)}

    def comparison(left_method, left_branch, right_method, right_branch):
        left = metrics[left_method][left_branch]
        right = metrics[right_method][right_branch]
        hm_gains = [a['harmonic_mean'] - b['harmonic_mean']
                    for a, b in zip(left['seeds'], right['seeds'])]
        return {
            'left': f'{left_method}/{left_branch}', 'right': f'{right_method}/{right_branch}',
            'base_mean_gain_pp': left['aggregate']['base_mean'] - right['aggregate']['base_mean'],
            'novel_mean_gain_pp': left['aggregate']['novel_mean'] - right['aggregate']['novel_mean'],
            'seed_harmonic_mean_gains_pp': hm_gains,
            'mean_seed_harmonic_mean_gain_pp': float(np.mean(hm_gains)),
            'positive_harmonic_mean_seeds': int(sum(value > 0 for value in hm_gains)),
        }

    comparisons = {
        '图片加权门控_对_图片加权简单融合': comparison(
            '图片加权', 'confidence_gated', '图片加权', 'simple_fusion'),
        '图片加权门控_对_均值描述门控': comparison(
            '图片加权', 'confidence_gated', '均值描述', 'confidence_gated'),
        '图片加权门控_对_类别名门控': comparison(
            '图片加权', 'confidence_gated', '类别名', 'confidence_gated'),
    }
    return {
        'status': 'exploratory_complete',
        'dataset': 'FGVCAircraft',
        'evidence_role': 'post_hoc_discovery_only',
        'warning': 'The formula was selected after Aircraft test outputs were available; these numbers are not independent confirmation evidence.',
        'formula': FORMULA,
        'input_root': str(input_root),
        'methods': metrics,
        'comparisons': comparisons,
        'next_confirmation': 'Run the unchanged formula first on DTD and then UCF101; do not tune on either dataset.',
    }


def write_outputs(result, output_dir):
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'Use a fresh output directory: {output_dir}')
    output_dir.mkdir(parents=True)
    (output_dir / '固定公式.json').write_text(
        json.dumps(result['formula'], ensure_ascii=False, indent=2) + '\n')
    (output_dir / '探索分析.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    attentive = result['methods']['图片加权']['confidence_gated']['aggregate']
    fusion = result['methods']['图片加权']['simple_fusion']['aggregate']
    main = result['comparisons']['图片加权门控_对_图片加权简单融合']
    uniform = result['comparisons']['图片加权门控_对_均值描述门控']
    lines = [
        '# 置信度门控探索结果', '',
        '## 固定公式', '',
        '- `F = 0.5 × (S + M)`',
        '- `q = 1 - H(softmax(S)) / log(C)`',
        '- `G = F + q × R`',
        '- 没有搜索温度、阈值或融合系数。', '',
        '## Aircraft 三种子结果', '',
        '| 方法 | Base | Novel | 种子 HM 均值 |',
        '|---|---:|---:|---:|',
        f"| 简单融合 F | {fusion['base_mean']:.2f} | {fusion['novel_mean']:.2f} | {fusion['mean_seed_harmonic_mean']:.2f} |",
        f"| 图片加权门控 G | {attentive['base_mean']:.2f} | {attentive['novel_mean']:.2f} | {attentive['mean_seed_harmonic_mean']:.2f} |",
        '',
        f"相对简单融合，HM 三个种子的增益分别为 {', '.join(f'{x:+.2f}' for x in main['seed_harmonic_mean_gains_pp'])} 个百分点，均值 {main['mean_seed_harmonic_mean_gain_pp']:+.2f}。",
        f"相对均值描述门控，HM 均值增益 {uniform['mean_seed_harmonic_mean_gain_pp']:+.2f}，{uniform['positive_harmonic_mean_seeds']}/3 个种子为正。",
        '', '## 证据边界', '',
        '该公式是在查看 Aircraft 测试输出后确定的，因此这组结果只能用于提出假设，不能作为论文中的独立验证。下一步保持公式不变，先在 DTD、再在 UCF101 验证。',
    ]
    (output_dir / '探索报告.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    result = analyze(args.input_root)
    write_outputs(result, args.output_dir)
    print(json.dumps({'status': result['status'], 'output_dir': str(Path(args.output_dir).resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
