"""Verify downloaded CPU evidence and compare fixed inputs with local preparation."""
from pathlib import Path
import csv
import hashlib
import json
import tarfile

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
REPO = ROOT / 'work/github-publish/textconsistency_v2_tune'
LOCAL = ROOT / 'work/validation-probe-20260911/prepared-ce31832'
UNPACKED = WORK / 'unpacked'
PREFIX = 'validation-prep-20260911-42527'
REMOTE_LOGS = Path('/root/autodl-tmp/textconsistency/setup-logs')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def mapped(path):
    return UNPACKED / Path(path).relative_to(REMOTE_LOGS)


def main():
    receipt = read(WORK / 'backup-receipt.json')
    archive = WORK / Path(receipt['archive']).name
    assert sha(archive) == receipt['sha256'] and archive.stat().st_size == receipt['bytes']
    expected_files = {x['path']: x for x in receipt['files']}
    assert len(expected_files) == len(receipt['files'])
    if UNPACKED.exists():
        raise FileExistsError(UNPACKED)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert len(members) == len(expected_files)
        assert {x.name for x in members} == set(expected_files)
        assert all(x.isfile() and not Path(x.name).is_absolute() and '..' not in Path(x.name).parts for x in members)
        tar.extractall(UNPACKED, filter='data')
    for rel, x in expected_files.items():
        path = UNPACKED / rel
        assert sha(path) == x['sha256'] and path.stat().st_size == x['bytes'], rel
    first, v2, retry, cache, final = [UNPACKED / (PREFIX + s) for s in ('', '-v2', '-resume', '-cache', '-final')]
    ready, execution = read(final / 'ready.json'), read(final / 'execution.json')
    assert ready['state'] == execution['state'] == 'server_cpu_preparation_complete_gpu_pending'
    assert len(ready['cases']) == 9 and len(execution['records']) == 11
    assert not ready['gpu_export_run'] and ready['new_training_runs'] == 0
    assert ready['full_test_count'] == 128 and ready['models'] == 27
    for item in execution['records']:
        assert sha(mapped(item['log'])) == item['log_sha256']
        assert item['returncode'] == (1 if Path(item['log']).name == 'no-cuda.log' else 0)
    tests = mapped(ready['tests']['log']).read_text()
    assert 'Ran 128 tests' in tests and '\nOK\n' in tests and '\nFAILED' not in tests
    assert read(first / 'inputs-before.json') == read(v2 / 'inputs-before.json') == read(final / 'inputs-after.json')
    assert sha(first / 'inputs-before.json') == ready['before_sha256'] == ready['after_sha256']
    assert (first / 'failure.json').exists()
    assert read(v2 / 'execution.json')['records'][-1]['returncode'] == -9
    assert read(retry / 'isolated-retry.json')['returncode'] == -9
    assert read(cache / 'isolated-retry.json')['returncode'] == 0
    assert 'CUDA is required' in read(final / 'no-cuda-rejection/failure.json')['error']
    for name, digest in ready['tools_sha256'].items():
        assert sha(REPO / 'scripts/repro' / name) == digest
    local_images = read(WORK / 'local-validation-image-hashes.json')
    remote_images = read(final / 'validation-image-hashes.json')
    assert local_images == remote_images
    assert len(remote_images) == 1280 and sum(x['bytes'] for x in remote_images.values()) == 179110246
    expected_cases = {(d, s) for d in ('ucf101', 'dtd', 'fgvc_aircraft') for s in (1, 2, 3)}
    assert {(c['dataset'], c['seed']) for c in ready['cases']} == expected_cases
    rows = []
    inventory_sha = sha(REPO / 'reports/diagnostics/error-analysis-20260911/evidence/proposed-probe-inputs.json')
    for c in ready['cases']:
        d, s = c['dataset'], c['seed']
        path = mapped(c['prepared'])
        assert sha(path) == c['sha256']
        a, b = read(path), read(LOCAL / f'{d}-seed{s}/prepared.json')
        assert a['status'] == 'prepared' and (a['dataset_key'], a['seed']) == (d, s)
        assert a['protocol'] == b['protocol'] and a['sample_hashes'] == b['sample_hashes']
        assert a['inventory']['sha256'] == b['inventory']['sha256'] == inventory_sha
        assert a['inventory']['case'] == b['inventory']['case']
        assert a['source_sha256'] == b['source_sha256']
        assert a['samples']['classes'] == b['samples']['classes']
        for split in ('train', 'val', 'test'):
            triples = lambda items: [(i['sample_id'], i['label'], i['classname']) for i in items]
            assert triples(a['samples'][split]) == triples(b['samples'][split])
        for k in ('clip_sha256', 'template_sha256'):
            assert a['assets'][k] == b['assets'][k]
        for role in ('coop', 'kd0', 'kd1'):
            for k in ('checkpoint_sha256', 'sidecar_sha256', 'manifest_sha256', 'effective_config_sha256'):
                assert a['checkpoints'][role][k] == b['checkpoints'][role][k]
        assert a['teacher_tensor_exact_equal'] and a['checkpoints']['teacher_tensor_exact_equal']
        assert a['transform']['repr'] == b['transform']['repr']
        for k in ('count', 'tensor_shape', 'tensor_dtype'):
            assert a['transform']['decoded_val'][k] == b['transform']['decoded_val'][k]
        rows.append({'dataset': d, 'seed': s, 'classes': len(a['samples']['classes']),
                     'train_count': len(a['samples']['train']), 'validation_count': len(a['samples']['val']),
                     'teacher_exact_equal': True, 'prepared_sha256': c['sha256']})
    assert sum(x['validation_count'] for x in rows) == 1500
    plan = read(final / 'gpu-plan.json')
    assert plan['state'] == 'not_executed' and plan['new_training_runs'] == 0
    assert {(x['dataset'], x['seed']) for x in plan['cases']} == expected_cases
    outputs = []
    for x in plan['cases']:
        c = next(c for c in ready['cases'] if (c['dataset'], c['seed']) == (x['dataset'], x['seed']))
        assert x['export'][x['export'].index('--prepared') + 1] == c['prepared']
        outputs.append(x['export'][x['export'].index('--output-dir') + 1])
    assert len(set(outputs)) == 9 and len(plan['aggregate'][plan['aggregate'].index('--analyses') + 1:plan['aggregate'].index('--output-dir')]) == 9
    audit = {'state': 'server_cpu_preparation_and_local_backup_verified', 'verified_date': '2026-09-12',
             'code_commit': ready['code_commit'], 'archive_sha256': receipt['sha256'],
             'archive_bytes': receipt['bytes'], 'files_verified': len(expected_files),
             'full_tests': 128, 'cases': rows, 'models_verified': 27, 'validation_records': 1500,
             'unique_validation_images': 1280, 'validation_image_bytes': 179110246,
             'validation_image_bytes_equal_local': True, 'sample_order_and_labels_equal_local': True,
             'source_clip_template_checkpoint_config_hashes_equal_local': True,
             'all_inputs_snapshots_unchanged': True, 'teacher_equal_pairs': 9,
             'test_portability_failure_and_two_sigkills_preserved': True,
             'gpu_plan_verified_not_executed': True, 'gpu_export_run': False, 'new_training_runs': 0,
             'tools_sha256': ready['tools_sha256'],
             'limitations': ['No new method score or real CUDA forward verification',
                 'Cross-machine ordered transform digest includes absolute paths and is not asserted equal; image bytes, transform definition, shape and dtype are compared',
                 'All-dataset input snapshots use path/size/mtime for images; content hashing covers all validation images only']}
    (WORK / 'local-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    with (WORK / 'cases.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in audit.items() if k not in ('cases', 'tools_sha256')}, indent=2))


if __name__ == '__main__':
    main()
