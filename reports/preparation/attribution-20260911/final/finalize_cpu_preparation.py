"""Record actual CPU readiness after reviewed deployment; never starts GPU work."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('/root/autodl-tmp/textconsistency')
setup = root / 'setup-logs/attribution-prep-20260911-v1'
commit = sys.argv[1]
release = root / ('releases/attribution-ops-20260911-' + commit)
prepared_dir = setup / ('prepared-' + commit)
prepared_path = prepared_dir / 'preparation.json'
prepared = json.loads(prepared_path.read_text())
protocol = json.loads((prepared_dir / 'protocol.json').read_text())
assert protocol == prepared['protocol']
assert protocol['command_count'] == 124 and len(protocol['stages']) == 11
assert sum(map(len, prepared['expected_effective_configs'].values())) == 124
study = Path(protocol['output_root'])
assert not study.exists()
sha = lambda path: hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()
for name in ('attribution_study.py', 'attribution_audit.py', 'evaluate_frozen_fusion.py'):
    path = release / 'scripts/repro' / name
    assert prepared['files_sha256'][str(path)] == sha(path)
validation = json.loads((setup / ('validation-' + commit + '.json')).read_text())
assert len(validation) == 3 and all(r['exit_code'] == 0 for r in validation)
launcher = setup / 'run_gpu_study.sh'
assert str(release.name) in launcher.read_text()
assert str(prepared_dir.name) in launcher.read_text()
subprocess.run(['bash', '-n', str(launcher)], check=True)
env = os.environ.copy()
env.update({key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')})
env['PYTHONDONTWRITEBYTECODE'] = '1'
env.pop('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', None)
import torch
cuda_available = torch.cuda.is_available()
assert not cuda_available, 'GPU became available; leave execution to the reviewed controller launch'
started = time.time()
guard = subprocess.run(['bash', str(launcher)], cwd=release, env=env, capture_output=True, text=True)
(setup / ('no-gpu-guard-' + commit + '.log')).write_text(guard.stdout + guard.stderr)
assert guard.returncode != 0
assert 'CUDA with at least one visible GPU is required for execution' in guard.stderr
assert not study.exists()
receipt = {
    'state': 'cpu_preparation_complete', 'operational_commit': commit,
    'release': str(release), 'preparation_dir': str(prepared_dir),
    'preparation_sha256': sha(prepared_path),
    'preparation_bytes': prepared_path.stat().st_size,
    'protocol_sha256': sha(prepared_dir / 'protocol.json'),
    'launcher': str(launcher), 'launcher_sha256': sha(launcher),
    'study_root': str(study), 'study_exists': study.exists(),
    'raw_command_count': 124, 'stage_count': 11,
    'frozen_dependency_files': len(prepared['files_sha256']),
    'frozen_historical_files': len(prepared['input_files_sha256']),
    'reference_models': 18, 'full_historical_models': len(prepared['historical']),
    'test_log': str(setup / ('tests-' + commit + '.log')),
    'validation_receipt': str(setup / ('validation-' + commit + '.json')),
    'no_gpu_guard': {'exit_code': guard.returncode, 'expected_rejection': True,
                     'seconds': time.time() - started},
    'cuda_available': cuda_available, 'gpu_smoke_completed': False,
    'new_gpu_evaluation_completed': False, 'formal_training_completed': False,
    'cgroup_memory_max': Path('/sys/fs/cgroup/memory.max').read_text().strip(),
    'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
}
ready = setup / 'ready.json'
assert not ready.exists(), 'Do not overwrite readiness evidence'
ready.write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
