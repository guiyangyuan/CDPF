"""Archive a fully audited study only; no mutation of experiment inputs/results."""
from pathlib import Path
import hashlib
import json
import os
import tarfile
import time

root = Path('/root/autodl-tmp/textconsistency')
study = root / 'runs/attribution-study-20260911-v1'
setup = root / 'setup-logs/attribution-gpu-20260911-42527'
prepared = root / 'setup-logs/attribution-prep-20260911-v1/prepared-43e537c'
status = json.loads((study / 'status.json').read_text())
analysis = json.loads((study / 'analysis.json').read_text())
assert status['state'] == 'passed' and analysis['passed']
assert len(status['commands']) == 136
assert all(row['exit_code'] == 0 and row['ended'] is not None for row in status['commands'])
assert len(status['completed']) == 12
assert len(analysis['full_seed_branch_rows']) == 36
assert len(analysis['fusion_seed_branch_rows']) == 54
archive = root / 'backups/attribution-study-20260911-42527.tar.gz'
receipt_path = setup / 'backup-receipt.json'
assert not archive.exists() and not receipt_path.exists()

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

files_before = {str(p.relative_to(study)): sha(p) for p in study.rglob('*') if p.is_file()}
checkpoints = []
for path in sorted(study.rglob('model.pth.tar-*')):
    if path.name.endswith('.sha256'):
        continue
    actual = sha(path)
    assert path.with_name(path.name + '.sha256').read_text().strip() == actual
    checkpoints.append({'relative_path': str(path.relative_to(study)), 'sha256': actual, 'bytes': path.stat().st_size})
assert len(checkpoints) == 10
assert sum(row['relative_path'].endswith('model.pth.tar-1') for row in checkpoints) == 2
assert sum(row['relative_path'].endswith('model.pth.tar-20') for row in checkpoints) == 8
with tarfile.open(archive, 'x:gz') as tar:
    tar.add(study, arcname=study.name)
    tar.add(setup, arcname='start-evidence')
    tar.add(prepared, arcname='preparation')
    for name in ['ready.json', 'run_gpu_study.sh', 'deployment-43e537c.json']:
        tar.add(prepared.parent / name, arcname='preparation/' + name)
files_after = {str(p.relative_to(study)): sha(p) for p in study.rglob('*') if p.is_file()}
assert files_before == files_after
launch = json.loads((setup / 'launch.json').read_text())
receipt = {
    'state': 'server_complete_backup_created', 'archive': str(archive),
    'archive_bytes': archive.stat().st_size, 'archive_sha256': sha(archive),
    'study_root': str(study), 'study_file_count': len(files_before),
    'study_file_sha256': files_before, 'checkpoints': checkpoints,
    'queue_command_count': 136, 'raw_command_count': 124,
    'queue_wall_seconds': status['commands'][-1]['ended'] - launch['started_epoch'],
    'new_checkpoint_count': 10, 'formal_or_anchor_checkpoints': 8, 'smoke_checkpoints': 2,
    'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'excludes': ['datasets', 'frozen CLIP weights', '18 historical reference weights already backed up separately'],
    'code': 'Operational code 43e537c and immutable model sources are preserved in Git and deployment archives',
}
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps({key: value for key, value in receipt.items() if key not in ['study_file_sha256', 'checkpoints']}, indent=2))
