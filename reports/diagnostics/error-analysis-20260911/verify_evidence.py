"""Cross-check derived diagnostics against previously audited counts and inputs."""
import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = args.workspace.resolve()
    repo = root / 'work/github-publish/textconsistency_v2_tune'
    report = repo / 'reports/diagnostics/error-analysis-20260911'
    evidence = report / 'evidence'
    analysis_path = repo / 'reports/experiments/attribution-study-20260911-42527/analysis.json'
    analysis = json.loads(analysis_path.read_text())
    scores = {(r['dataset'], r['seed'], r['branch']): r for r in analysis['fusion_seed_branch_rows']}
    tables = {}
    for path in evidence.glob('*.csv'):
        with path.open() as stream:
            tables[path.stem] = list(csv.DictReader(stream))
    assert {k: len(v) for k, v in tables.items()} == {
        'transitions': 126, 'fusion-partitions': 54, 'class-transitions': 2232, 'transition-summaries': 42}
    indexed, grouped = {}, defaultdict(list)
    for row in tables['transitions']:
        d, s, split, ref, target = (row[k] for k in ('dataset', 'seed', 'split', 'reference', 'target'))
        key = d, s, split, ref, target
        assert key not in indexed
        indexed[key] = row
        before = scores[d, int(s), ref]['counts'][split]
        after = scores[d, int(s), target]['counts'][split]
        assert int(row['n']) == before['total'] == after['total']
        assert int(row['reference_correct']) == before['correct']
        assert int(row['target_correct']) == after['correct']
        assert int(row['rescued']) - int(row['harmed']) == after['correct'] - before['correct'] == int(row['net_correct'])
        assert sum(int(row[k]) for k in ('both_correct', 'rescued', 'harmed', 'neither_correct')) == int(row['n'])
        assert math.isclose(float(row['net_pp']), 100 * (after['correct'] - before['correct']) / before['total'], abs_tol=1e-12)
        grouped[d, split, ref, target].append(row)
    for row in tables['transition-summaries']:
        rows = grouped[tuple(row[k] for k in ('dataset', 'split', 'reference', 'target'))]
        assert sorted(r['seed'] for r in rows) == ['1', '2', '3']
        for summary, source in [('rescued_mean', 'rescued'), ('harmed_mean', 'harmed'), ('net_pp_mean', 'net_pp')]:
            assert math.isclose(float(row[summary]), mean(float(r[source]) for r in rows), abs_tol=1e-12)
        assert math.isclose(float(row['net_pp_sd']), stdev(float(r['net_pp']) for r in rows), abs_tol=1e-12)
    class_groups = defaultdict(list)
    for row in tables['class-transitions']:
        class_groups[tuple(row[k] for k in ('dataset', 'seed', 'split', 'fusion'))].append(row)
    for (d, s, split, fusion), rows in class_groups.items():
        assert len({r['label'] for r in rows}) == len(rows)
        total = indexed[d, s, split, 'student', fusion]
        for field in ('n', 'reference_correct', 'target_correct', 'rescued', 'harmed', 'net_correct'):
            assert sum(int(r[field]) for r in rows) == int(total[field])
    assert len(class_groups) == 54
    for row in tables['fusion-partitions']:
        d, s, split, fusion = (row[k] for k in ('dataset', 'seed', 'split', 'fusion'))
        groups = ('both_correct', 'student_only', 'description_only', 'both_wrong')
        expected = scores[d, int(s), fusion]['counts'][split]
        assert sum(int(row[k + '_n']) for k in groups) == expected['total']
        assert sum(int(row[k + '_fusion_correct']) for k in groups) == expected['correct']
        assert all(0 <= int(row[k + '_fusion_correct']) <= int(row[k + '_n']) for k in groups)
    diag = json.loads((evidence / 'diagnostic.json').read_text())
    assert diag['passed'] and diag['test_prediction_records'] == 26424 and diag['unique_test_images'] == 8808
    study_root = root / 'outputs/textconsistency-attribution-study-20260911-42527'
    study = study_root / 'unpacked/attribution-study-20260911-v1'
    receipt_path = study_root / 'local-backup-receipt.json'
    assert sha(receipt_path) == diag['input_receipt_sha256']
    assert len(diag['input_hashes']) == 38
    assert all(sha(study / path) == value for path, value in diag['input_hashes'].items())
    tensor = json.loads((evidence / 'kd-tensor-comparison.json').read_text())
    assert tensor['passed'] and len(tensor['pairs']) == tensor['teacher_exact_equal_pairs'] == tensor['student_changed_pairs'] == 6
    for pair in tensor['pairs']:
        teacher, student = pair['groups']['view_aggregator'], pair['groups']['prompt_learner']
        assert teacher['all_tensors_exact_equal'] and teacher['numel'] == 787970 and teacher['max_abs_difference'] == 0
        assert not student['all_tensors_exact_equal'] and student['numel'] == 2048
    inputs = json.loads((evidence / 'proposed-probe-inputs.json').read_text())
    assert inputs['state'] == 'local_input_inventory_verified_not_gpu_ready'
    assert len(inputs['cases']) == inputs['validation_cases'] == 9
    model_paths = []
    for case in inputs['cases']:
        assert case['split'] == 'val' and case['candidate_classes'] == 'base'
        for item in case['checkpoints'].values():
            path = root / item['workspace_relative_checkpoint']
            assert sha(path) == item['sha256']
            assert sha(path.parent.parent / 'run_manifest.json') == item['manifest_sha256']
            model_paths.append(str(path.resolve()))
    assert len(set(model_paths)) == inputs['model_count'] == 27
    for path in report.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    local_links = 0
    for path in report.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if not target.startswith(('https://', 'http://', '#')):
                resolved = (path.parent / target.split('#')[0]).resolve()
                assert resolved.exists() or resolved == args.output.resolve(), (path.name, target)
                local_links += 1
    with tempfile.TemporaryDirectory(prefix='tc-diagnostic-rejection-') as directory:
        temp = Path(directory)
        receipt = json.loads(receipt_path.read_text())
        receipt['study_file_sha256']['status.json'] = '0' * 64
        (temp / 'receipt.json').write_text(json.dumps(receipt))
        rejected = subprocess.run([sys.executable, str(report / 'analyze_predictions.py'),
            '--study', str(study), '--receipt', str(temp / 'receipt.json'), '--output', str(temp / 'out')],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert rejected.returncode != 0 and 'AssertionError: status.json' in rejected.stderr
        assert not (temp / 'out').exists()
    artifacts = sorted([*report.glob('*.py'), *evidence.glob('*.csv'),
                        *(p for p in evidence.glob('*.json') if p.name != 'verification.json')])
    result = {'passed': True, 'checks': {'audited_count_comparisons': 126, 'class_group_sums': 54,
        'class_rows': 2232, 'fusion_partition_totals': 54, 'seed_summaries': 42,
        'input_file_sha256_rechecked': 38, 'checkpoint_and_manifest_sha256_rechecked': 27,
        'teacher_identical_student_changed_pairs': 6, 'local_markdown_targets': local_links,
        'corrupted_receipt_rejected_before_output': True, 'scripts_compile': True},
        'audited_source_analysis_sha256': sha(analysis_path),
        'artifacts_sha256': {str(p.relative_to(report)): sha(p) for p in artifacts},
        'scope': 'Local evidence consistency only; no new model inference or GPU readiness assertion'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['checks'], indent=2))


if __name__ == '__main__':
    main()
