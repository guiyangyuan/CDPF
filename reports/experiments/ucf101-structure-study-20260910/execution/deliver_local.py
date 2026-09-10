"""Verify the received completed backup before publishing lightweight evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

workspace = Path.cwd()
output = workspace / 'outputs/textconsistency-structure-study-20260910-27946'
remote = json.loads((output/'server-backup-receipt.json').read_text())
archive = output/Path(remote['path']).name
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
assert archive.stat().st_size == remote['size'] and sha(archive) == remote['sha256']
unpacked = output/'unpacked'; unpacked.mkdir(exist_ok=False)
with tarfile.open(archive) as tar:
    for member in tar.getmembers():
        path = Path(member.name)
        assert not path.is_absolute() and '..' not in path.parts
        assert member.isfile() or member.isdir()
    tar.extractall(unpacked, filter='data')
study = unpacked/'structure-study-20260910-27946-v2'
analysis = json.loads((study/'analysis.json').read_text())
assert analysis['passed']
checks=[]
for phase, items in [('formal', analysis['verified_checkpoints']), ('smoke', analysis['smoke_audit']['models'])]:
    for row in items:
        epoch = 20 if phase=='formal' else 1
        checkpoint = study/phase/row['variant']/f'seed{row["seed"]}'/'textconsistency'/f'model.pth.tar-{epoch}'
        assert sha(checkpoint)==row['sha256']
        checks.append({'phase':phase,'variant':row['variant'],'seed':row['seed'],'sha256':row['sha256']})
assert len(checks)==20
receipt = {**remote, 'local_archive':str(archive.relative_to(workspace)),
           'local_extracted_study':str(study.relative_to(workspace)), 'local_archive_hash_verified':True,
           'all_20_checkpoint_hashes_verified':True, 'checkpoints':checks}
(output/'backup-receipt.json').write_text(json.dumps(receipt,indent=2))
repo=workspace/'work/github-publish/textconsistency_v2_tune'
report=repo/'reports/experiments/ucf101-structure-study-20260910'
evidence=report/'evidence'; evidence.mkdir(exist_ok=False)
for name in ['analysis.json','runtime.json','smoke-audit.json','formal-audit.json','reference-replay.json']:
    shutil.copy2(study/name,report/name)
shutil.copy2(study/'status.json',report/'completed-status.json')
shutil.copy2(output/'backup-receipt.json',report/'backup-receipt.json')
for path in (study/'formal').rglob('*'):
    if path.is_file() and (path.name in ('protocol.json','data_preflight.json','run_manifest.json',
                                        'effective_config.yaml','metrics.json','results.json','results.csv')):
        target=evidence/path.relative_to(study/'formal'); target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
execution=report/'execution'
for name in ['analyze_structure.py','prepare_delivery.py','progress.py','structure_queue.py']:
    shutil.copy2(unpacked/'structure-gpu-20260910-27946/execution-v2'/name,execution/name)
print(json.dumps({'verified':True,'models':len(checks),'archive':str(archive),'sha256':remote['sha256'],
                  'lightweight_evidence_files':len(list(evidence.rglob('*')))},indent=2))
