"""Verify the downloaded archive and every backed-up study file."""
from pathlib import Path
import hashlib
import json
import tarfile
import time

destination = Path('outputs/textconsistency-attribution-study-20260911-42527').resolve()
receipt = json.loads((destination / 'backup-receipt.json').read_text())
archive = destination / Path(receipt['archive']).name

def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

assert archive.stat().st_size == receipt['archive_bytes']
assert sha(archive) == receipt['archive_sha256']
unpacked = destination / 'unpacked'
unpacked.mkdir(exist_ok=False)
with tarfile.open(archive, 'r:gz') as tar:
    for member in tar.getmembers():
        target = (unpacked / member.name).resolve()
        assert unpacked == target or unpacked in target.parents
        assert member.isdir() or member.isfile(), (member.name, member.type)
    tar.extractall(unpacked)
study = unpacked / Path(receipt['study_root']).name
actual_files = {str(p.relative_to(study)): sha(p) for p in study.rglob('*') if p.is_file()}
assert actual_files == receipt['study_file_sha256']
assert len(actual_files) == receipt['study_file_count']
for row in receipt['checkpoints']:
    checkpoint = study / row['relative_path']
    assert checkpoint.stat().st_size == row['bytes']
    assert sha(checkpoint) == row['sha256']
    assert checkpoint.with_name(checkpoint.name + '.sha256').read_text().strip() == row['sha256']
status = json.loads((study / 'status.json').read_text())
analysis = json.loads((study / 'analysis.json').read_text())
assert status['state'] == 'passed' and analysis['passed']
assert len(status['commands']) == 136 and all(c['exit_code'] == 0 for c in status['commands'])
assert len(status['completed']) == 12
assert len(analysis['full_seed_branch_rows']) == 36
assert len(analysis['fusion_seed_branch_rows']) == 54
receipt.update({
    'state': 'local_complete_backup_verified',
    'local_archive': str(archive),
    'local_study': str(study),
    'verified_study_file_count': len(actual_files),
    'verified_checkpoint_count': len(receipt['checkpoints']),
    'local_verified_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
})
with (destination / 'local-backup-receipt.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps({k: v for k, v in receipt.items() if k not in ('study_file_sha256', 'checkpoints')}, indent=2))
