"""Hash validation images and archive completed CPU preparation evidence only."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path('/root/autodl-tmp/textconsistency')
FIRST = ROOT / 'setup-logs/validation-prep-20260911-42527'
V2 = ROOT / 'setup-logs/validation-prep-20260911-42527-v2'
RETRY = ROOT / 'setup-logs/validation-prep-20260911-42527-resume'
CACHE = ROOT / 'setup-logs/validation-prep-20260911-42527-cache'
PREP = ROOT / 'setup-logs/validation-prep-20260911-42527-final'
ARCHIVE = ROOT / 'backups/validation-server-prep-20260911-de57efd.tar.gz'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if ARCHIVE.exists() or (PREP / 'validation-image-hashes.json').exists():
        raise FileExistsError('Final evidence already exists; do not overwrite')
    ready = json.loads((PREP / 'ready.json').read_text())
    execution = json.loads((PREP / 'execution.json').read_text())
    assert ready['state'] == execution['state'] == 'server_cpu_preparation_complete_gpu_pending'
    assert ready['full_test_count'] == 128 and ready['models'] == 27
    assert not ready['gpu_export_run'] and ready['new_training_runs'] == 0
    assert len(ready['cases']) == 9 and len(execution['records']) == 11
    assert not (PREP / 'failure.json').exists()
    assert (FIRST / 'failure.json').exists()
    for row in execution['records']:
        assert sha(row['log']) == row['log_sha256']
        assert row['returncode'] == (1 if Path(row['log']).name == 'no-cuda.log' else 0)
    before = json.loads((V2 / 'inputs-before.json').read_text())
    assert before == json.loads((PREP / 'inputs-after.json').read_text())
    assert before == json.loads((FIRST / 'inputs-before.json').read_text())
    assert sha(FIRST / 'inputs-before.json') == ready['before_sha256']
    assert sha(PREP / 'inputs-after.json') == ready['after_sha256']
    assert json.loads((RETRY / 'isolated-retry.json').read_text())['returncode'] == -9
    assert json.loads((CACHE / 'isolated-retry.json').read_text())['returncode'] == 0
    assert json.loads((V2 / 'execution.json').read_text())['records'][-1]['returncode'] == -9
    images = {}
    records = 0
    for case in ready['cases']:
        path = Path(case['prepared'])
        assert sha(path) == case['sha256']
        prepared = json.loads(path.read_text())
        data_root = Path(prepared['data_root']).resolve()
        assert prepared['checkpoints']['teacher_tensor_exact_equal']
        decoded = prepared['transform']['decoded_val']
        assert decoded['count'] == len(prepared['samples']['val'])
        assert decoded['tensor_dtype'] == 'torch.float32' and decoded['tensor_shape'] == [3, 224, 224]
        for row in prepared['samples']['val']:
            records += 1
            image = (data_root / row['sample_id']).resolve()
            image.relative_to(data_root)
            if row['sample_id'] not in images:
                images[row['sample_id']] = {'sha256': sha(image), 'bytes': image.stat().st_size}
    assert records == 1500 and len(images) == 1280
    assert not (ROOT / 'runs/validation-probe-20260911-v1').exists()
    (PREP / 'validation-image-hashes.json').write_text(json.dumps(images, indent=2) + '\n')
    audit = {'state': 'server_evidence_checked_local_comparison_pending',
             'validation_records': records, 'unique_validation_images_hashed': len(images),
             'validation_image_bytes': sum(x['bytes'] for x in images.values()),
             'all_log_hashes_checked': True, 'all_prepared_hashes_checked': True,
             'inputs_unchanged_since_first_attempt': True,
             'memory_events': Path('/sys/fs/cgroup/memory.events').read_text(),
             'gpu_export_run': False, 'new_training_runs': 0}
    (PREP / 'server-evidence-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    members = []
    with tarfile.open(ARCHIVE, 'w:gz') as tar:
        for folder in (FIRST, V2, RETRY, CACHE, PREP):
            for path in sorted(folder.rglob('*')):
                if not path.is_file():
                    continue
                assert not path.is_symlink() and path.suffix in ('.json', '.log'), path
                relative = str(Path(folder.name) / path.relative_to(folder))
                tar.add(path, arcname=relative, recursive=False)
                members.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    receipt = {'archive': str(ARCHIVE), 'sha256': sha(ARCHIVE), 'bytes': ARCHIVE.stat().st_size,
               'files': members, 'contains': 'CPU receipts/logs/input snapshots and image hashes; no images or weights',
               'first_failed_attempt_retained': True, 'both_sigkill_attempts_retained': True,
               'gpu_export_run': False, 'new_training_runs': 0}
    (PREP / 'backup-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'files'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
