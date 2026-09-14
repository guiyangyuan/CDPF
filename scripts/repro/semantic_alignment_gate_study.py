#!/usr/bin/env python3
"""Screen the fixed semantic-alignment gate using authenticated saved logits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


FORMULA = {
    'simple_fusion': 'F=0.5*(S+M)',
    'semantic_direction': 'D=center(M-S)',
    'residual_direction': 'R0=center(R_hat)',
    'gate': 'q=clamp(cos(R0,D),0,1)',
    'corrected': 'G=F+q*R_hat',
    'fitted_hyperparameters': [],
}
REQUIRED_ARRAYS = {'labels', 'S', 'M', 'R', 'Z', 'F'}
ROOT = Path(__file__).resolve().parents[2]
DEPENDENCIES = (
    'trainers/coop_text_detail.py',
    'configs/repro/text_detail/semantic_gated_radius_attentive.yaml',
    'scripts/repro/semantic_alignment_gate_study.py',
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


def _logits(name, value):
    array = np.asarray(value)
    if (array.ndim != 2 or array.shape[1] < 2
            or not np.issubdtype(array.dtype, np.floating)
            or not np.isfinite(array).all()):
        raise ValueError(f'{name} must contain finite two-dimensional logits')
    return array


def compute_gated_logits(student, description, residual, eps=1e-8):
    student = _logits('student', student)
    description = _logits('description', description)
    residual = _logits('residual', residual)
    if student.shape != description.shape or student.shape != residual.shape:
        raise ValueError('student, description, and residual shapes must match')
    dtype = np.result_type(student.dtype, description.dtype, residual.dtype)
    student = student.astype(dtype, copy=False)
    description = description.astype(dtype, copy=False)
    residual = residual.astype(dtype, copy=False)
    semantic = description - student
    semantic -= semantic.mean(axis=1, keepdims=True)
    centered_residual = residual - residual.mean(axis=1, keepdims=True)
    numerator = np.sum(centered_residual * semantic, axis=1)
    denominator = np.maximum(np.linalg.norm(centered_residual, axis=1), eps)
    denominator *= np.maximum(np.linalg.norm(semantic, axis=1), eps)
    gate = np.clip(numerator / denominator, 0., 1.).astype(dtype, copy=False)
    fusion = .5 * (student + description)
    corrected = fusion + gate[:, None] * residual
    return {'F': fusion, 'Q': gate, 'G': corrected}


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
    for name in REQUIRED_ARRAYS - {'labels'}:
        value = _logits(name, arrays[name])
        if len(value) != len(labels):
            raise ValueError(f'Logit row count differs from labels: {archive_path}')
    np.testing.assert_array_equal(arrays['F'], .5 * (arrays['S'] + arrays['M']))
    np.testing.assert_array_equal(arrays['Z'], arrays['F'] + arrays['R'])
    if (len(record.get('rows', [])) != len(labels)
            or len(record.get('class_order', [])) != arrays['S'].shape[1]):
        raise ValueError(f'Export metadata does not match predictions: {folder}')
    return record, arrays


def _accuracy(logits, labels):
    return float(100. * (logits.argmax(1) == labels).mean())


def _harmonic_mean(base, novel):
    return float(2. * base * novel / (base + novel)) if base + novel else 0.


def development_decision(hm_gain, novel_gain):
    values = [float(hm_gain), float(novel_gain)]
    if not all(np.isfinite(values)):
        raise ValueError('Development decision requires finite gains')
    criteria = {
        'HM增益为正': values[0] > 0.,
        'Novel下降不超过1个百分点': values[1] >= -1.,
    }
    return {
        'passes': all(criteria.values()),
        'criteria': criteria,
        'HM增益': values[0],
        'Novel增益': values[1],
    }


def _split_analysis(arrays):
    computed = compute_gated_logits(arrays['S'], arrays['M'], arrays['R'])
    labels = arrays['labels']
    fusion_predictions = computed['F'].argmax(1)
    gated_predictions = computed['G'].argmax(1)
    fusion_correct = fusion_predictions == labels
    gated_correct = gated_predictions == labels
    return {
        'total': int(len(labels)),
        'F_accuracy': _accuracy(computed['F'], labels),
        'G_accuracy': _accuracy(computed['G'], labels),
        'radius_accuracy': _accuracy(arrays['Z'], labels),
        'gate_mean': float(computed['Q'].mean()),
        'gate_zero_fraction': float((computed['Q'] == 0.).mean()),
        'predictions_changed_vs_F': int((gated_predictions != fusion_predictions).sum()),
        'corrections_vs_F': int((~fusion_correct & gated_correct).sum()),
        'harms_vs_F': int((fusion_correct & ~gated_correct).sum()),
        'Q': computed['Q'],
        'G': computed['G'],
    }


def analyze_exports(base_folder, novel_folder):
    base_record, base_arrays = load_export(base_folder)
    novel_record, novel_arrays = load_export(novel_folder)
    split_rows = {
        'base': _split_analysis(base_arrays),
        'novel': _split_analysis(novel_arrays),
    }
    hm_f = _harmonic_mean(split_rows['base']['F_accuracy'], split_rows['novel']['F_accuracy'])
    hm_g = _harmonic_mean(split_rows['base']['G_accuracy'], split_rows['novel']['G_accuracy'])
    hm_radius = _harmonic_mean(
        split_rows['base']['radius_accuracy'], split_rows['novel']['radius_accuracy'])
    for row in split_rows.values():
        row.pop('Q'); row.pop('G')
    hm_gain = hm_g - hm_f
    novel_gain = split_rows['novel']['G_accuracy'] - split_rows['novel']['F_accuracy']
    return {
        'status': 'development_screen_complete',
        'dataset': 'DTD',
        'evidence_role': 'development_only',
        'new_trainings': 0,
        'formula': FORMULA,
        'source_exports': {
            'base': {'folder': str(Path(base_folder).resolve()),
                     'predictions_sha256': base_record['predictions_sha256']},
            'novel': {'folder': str(Path(novel_folder).resolve()),
                      'predictions_sha256': novel_record['predictions_sha256']},
        },
        'splits': split_rows,
        'harmonic_means': {'F': hm_f, 'G': hm_g, 'radius': hm_radius},
        'hm_gain': hm_gain,
        'novel_gain': novel_gain,
        'decision': development_decision(hm_gain, novel_gain),
    }


def _dependency_hashes():
    return {name: sha256_file(ROOT / name) for name in DEPENDENCIES}


def _source_hashes(folder):
    folder = Path(folder).resolve()
    return {name: sha256_file(folder / name) for name in ('导出.json', '预测.npz')}


def build_protocol(base_folder, novel_folder, output_root, repository_commit):
    base_folder, novel_folder = Path(base_folder).resolve(), Path(novel_folder).resolve()
    return {
        'protocol': '语义一致性门控-v1',
        'created_before_gate_results': True,
        'repository_commit': repository_commit,
        'formula': FORMULA,
        'source': {
            'dataset': 'DTD',
            'role': 'development_only',
            'base_folder': str(base_folder),
            'novel_folder': str(novel_folder),
            'base_sha256': _source_hashes(base_folder),
            'novel_sha256': _source_hashes(novel_folder),
            'source_model': 'old fusion-aligned seed1 checkpoint reinterpreted with description-radius bound',
        },
        'output_root': str(Path(output_root).resolve()),
        'maximum_new_trainings': 0,
        'stopping_rule': 'stop unless G>F in HM and Novel gain >= -1pp',
        'next_evidence': 'UCF101 remains untouched unless the DTD development gate passes',
        'files_sha256': _dependency_hashes(),
    }


def verify_protocol(protocol):
    if (protocol.get('protocol') != '语义一致性门控-v1'
            or protocol.get('formula') != FORMULA
            or protocol.get('maximum_new_trainings') != 0
            or protocol.get('stopping_rule') != 'stop unless G>F in HM and Novel gain >= -1pp'):
        raise ValueError('Unknown or modified semantic-alignment protocol')
    if protocol.get('files_sha256') != _dependency_hashes():
        raise ValueError('Frozen source or configuration changed')
    source = protocol['source']
    for split in ('base', 'novel'):
        if source.get(f'{split}_sha256') != _source_hashes(source[f'{split}_folder']):
            raise ValueError(f'Frozen {split} export changed')


def _write_gate_archive(source_folder, target):
    _, arrays = load_export(source_folder)
    computed = compute_gated_logits(arrays['S'], arrays['M'], arrays['R'])
    np.savez_compressed(target, labels=arrays['labels'], Q=computed['Q'], G=computed['G'])
    return sha256_file(target)


def write_report(path, analysis):
    base, novel = analysis['splits']['base'], analysis['splits']['novel']
    hm = analysis['harmonic_means']
    decision = analysis['decision']
    lines = [
        '# 语义一致性门控 DTD 零训练筛选', '',
        f"结论：{'通过开发筛选' if decision['passes'] else '未通过开发筛选，停止本候选'}。", '',
        '| 方法 | Base | Novel | HM |',
        '|---|---:|---:|---:|',
        f"| 简单融合 F | {base['F_accuracy']:.2f} | {novel['F_accuracy']:.2f} | {hm['F']:.2f} |",
        f"| 半径约束残差 | {base['radius_accuracy']:.2f} | {novel['radius_accuracy']:.2f} | {hm['radius']:.2f} |",
        f"| 语义一致性门控 G | {base['G_accuracy']:.2f} | {novel['G_accuracy']:.2f} | {hm['G']:.2f} |",
        '',
        f"G 相对 F：Base {base['G_accuracy'] - base['F_accuracy']:+.2f}pp，"
        f"Novel {analysis['novel_gain']:+.2f}pp，HM {analysis['hm_gain']:+.2f}pp。",
        f"平均门控值：Base {base['gate_mean']:.4f}，Novel {novel['gate_mean']:.4f}。",
        f"逐样本变化：Base 纠正 {base['corrections_vs_F']}、改错 {base['harms_vs_F']}；"
        f"Novel 纠正 {novel['corrections_vs_F']}、改错 {novel['harms_vs_F']}。", '',
        '本筛选复用已归档 logits，新增训练为 0。公式、输入哈希和停止条件均在计算结果前冻结。',
        'DTD 是开发证据；只有两项门槛全部通过，才允许在未查看的 UCF101 上做独立确认。',
    ]
    Path(path).write_text('\n'.join(lines) + '\n')


def write_protocol(args):
    target = Path(args.output).resolve()
    if target.exists():
        raise FileExistsError(f'Protocol already exists: {target}')
    protocol = build_protocol(args.base_export, args.novel_export,
                              args.output_root, args.repository_commit)
    atomic_json(target, protocol)
    return {'status': 'prepared', 'protocol': str(target), 'maximum_new_trainings': 0}


def run_protocol(protocol_path):
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    verify_protocol(protocol)
    output_root = Path(protocol['output_root'])
    if output_root.exists():
        raise FileExistsError(f'Use a fresh output directory: {output_root}')
    output_root.mkdir(parents=True)
    started = time.time()
    source = protocol['source']
    analysis = analyze_exports(source['base_folder'], source['novel_folder'])
    atomic_json(output_root / '结果分析.json', analysis)
    write_report(output_root / '结果报告.md', analysis)
    archives = {
        '基础测试集门控预测.npz': _write_gate_archive(
            source['base_folder'], output_root / '基础测试集门控预测.npz'),
        '新类别测试集门控预测.npz': _write_gate_archive(
            source['novel_folder'], output_root / '新类别测试集门控预测.npz'),
    }
    status = {
        'state': ('passed_development_gate' if analysis['decision']['passes']
                  else 'stopped_by_preregistered_rule'),
        'protocol': str(protocol_path),
        'started': started,
        'ended': time.time(),
        'new_trainings_completed': 0,
        'UCF101_output_created': False,
        'gate_archives_sha256': archives,
        'conclusion': ('proceed to frozen UCF101 confirmation' if analysis['decision']['passes']
                       else 'DTD development gate failed; do not use UCF101'),
    }
    atomic_json(output_root / '执行状态.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    plan = commands.add_parser('plan')
    plan.add_argument('--base-export', required=True)
    plan.add_argument('--novel-export', required=True)
    plan.add_argument('--output-root', required=True)
    plan.add_argument('--repository-commit', required=True)
    plan.add_argument('--output', required=True)
    run = commands.add_parser('run')
    run.add_argument('--protocol', required=True)
    args = parser.parse_args()
    result = write_protocol(args) if args.command == 'plan' else run_protocol(args.protocol)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
