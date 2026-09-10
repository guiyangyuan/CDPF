"""Package completed study evidence and models without datasets or base CLIP weights."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--study', type=Path, required=True)
    a = p.parse_args()
    root, study = a.root.resolve(), a.study.resolve()
    assert study.parent == root / 'runs'
    assert json.loads((study / 'status.json').read_text())['state'] == 'completed'
    analysis = json.loads((study / 'analysis.json').read_text())
    assert analysis['passed'] and len(analysis['verified_checkpoints']) == 15
    assert len(analysis['smoke_audit']['models']) == 5
    directory = root / 'backups'
    directory.mkdir(exist_ok=True)
    archive = directory / (study.name + '.tar.gz')
    assert not archive.exists()
    temporary = archive.with_suffix('.tar.gz.tmp')
    assert not temporary.exists()
    with tarfile.open(temporary, 'w:gz') as tar:
        tar.add(study, arcname=study.name)
        logs = root / 'setup-logs/structure-gpu-20260910-27946'
        tar.add(logs, arcname='structure-gpu-20260910-27946')
    temporary.replace(archive)
    with archive.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    receipt = {'created_unix': time.time(), 'path': str(archive), 'size': archive.stat().st_size,
               'sha256': digest, 'study_complete': True, 'includes': '15 formal checkpoints plus 5 smoke checkpoints, configs, logs, protocol, analysis and startup/recovery evidence',
               'excludes': 'datasets and frozen base CLIP weights'}
    archive.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
