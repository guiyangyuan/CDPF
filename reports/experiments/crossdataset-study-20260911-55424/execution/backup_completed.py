"""Package this fixed study only after its final audit passes."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

ROOT = Path('/root/autodl-tmp/textconsistency')
STUDY = ROOT / 'runs/crossdataset-study-20260910-v1'
START = ROOT / 'setup-logs/crossdataset-gpu-start-20260911-55424'
PREP = ROOT / 'setup-logs/crossdataset-prep-20260911-v2'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    status = json.loads((STUDY / 'status.json').read_text())
    analysis = json.loads((STUDY / 'analysis.json').read_text())
    assert status['state'] == 'passed' and analysis['passed']
    assert len(analysis['seed_branch_rows']) == 42 and len(analysis['summary_rows']) == 14
    checkpoints = []
    for phase, seeds, epoch in [('smoke', [1], 1), ('formal', [1, 2, 3], 20)]:
        for dataset in ['dtd', 'fgvc_aircraft']:
            assert json.loads((STUDY / 'audit' / f'{phase}_{dataset}.json').read_text())['passed']
            for variant in ['baseline', 'hierarchical_adapter', 'flat_dual_adapter']:
                for seed in seeds:
                    path = STUDY / phase / dataset / variant / f'seed{seed}' / 'textconsistency' / f'model.pth.tar-{epoch}'
                    digest = sha(path)
                    assert digest == path.with_name(path.name + '.sha256').read_text().strip()
                    checkpoints.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                                        'sha256': digest, 'phase': phase})
    assert len(checkpoints) == 24
    archive = ROOT / 'backups/crossdataset-study-20260911-55424.tar.gz'
    temporary = archive.with_name(archive.name + '.partial')
    assert not archive.exists() and not temporary.exists()
    archive.parent.mkdir(exist_ok=True)
    subprocess.run(['tar', '-czf', str(temporary), '-C', str(ROOT),
                    str(STUDY.relative_to(ROOT)), str(START.relative_to(ROOT)), str(PREP.relative_to(ROOT))], check=True)
    temporary.replace(archive)
    receipt = {'created_unix': time.time(), 'archive': str(archive), 'bytes': archive.stat().st_size,
               'sha256': sha(archive), 'checkpoints': checkpoints,
               'scope': '24 checkpoints, all study logs/configs/metrics/audits, GPU launch evidence and CPU preparation including operational source archive; excludes datasets and frozen CLIP weights'}
    target = START / 'backup-receipt.json'
    with target.open('x') as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps({key: value for key, value in receipt.items() if key != 'checkpoints'}, indent=2))


if __name__ == '__main__':
    main()
