"""One-time deployment/CPU verification of the reviewed 13b3497 source archive."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

root = Path('/root/autodl-tmp/textconsistency')
archive = root / 'inbox/structure-13b3497.tar.gz'
release = root / 'releases/structure-20260910-v1'
evidence = root / 'setup-logs/structure-prep-20260910'
expected_archive = '56479bfe2f73a8e3278464f2ffda7ca9d57d6772d943ce14234493c55cc1c1b5'
expected_source = '9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba'
python = root / '.venv/bin/python'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

assert sha(archive) == expected_archive, 'Archive hash mismatch'
assert not release.exists() and not evidence.exists(), 'Use a fresh release/evidence directory'
with tarfile.open(archive) as tar:
    for item in tar.getmembers():
        path = Path(item.name)
        assert not path.is_absolute() and '..' not in path.parts
        assert item.isfile() or item.isdir(), 'Unsupported archive member'
    release.mkdir(parents=True)
    tar.extractall(release, filter='data')
evidence.mkdir(parents=True)
(release / 'clip/ViT-B-16.pt').symlink_to(root / 'assets/ViT-B-16.pt')
env = os.environ.copy()
env.update(PYTHONPATH=f'{release}:{release}/Dassl.pytorch', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
env.pop('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', None)
records = []

def run(name, args, expected=0):
    print(f'Checking {name}', flush=True)
    started = time.monotonic()
    result = subprocess.run([str(python), '-B', *map(str, args)], cwd=release, env=env,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (evidence / f'{name}.txt').write_text(result.stdout)
    records.append({'name': name, 'returncode': result.returncode,
                    'seconds': round(time.monotonic()-started, 3), 'log': f'{name}.txt'})
    assert result.returncode == expected, f'{name}: exit {result.returncode}; see {evidence}'
    return result.stdout

runtime = json.loads(run('runtime', ['-c', 'import json,torch,platform; from trainers.textconsistency_runtime import source_digest; print(json.dumps({"torch":torch.__version__,"python":platform.python_version(),"cuda":torch.cuda.is_available(),"code_sha256":source_digest()}))']))
assert runtime['cuda'] is False and runtime['code_sha256'] == expected_source
memory_limit = Path('/sys/fs/cgroup/memory.max').read_text().strip()
tests = run('server-tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
assert 'Ran 46 tests' in tests and tests.rstrip().endswith('OK')
preflights = {}
for dataset in ('ucf101', 'dtd', 'fgvc_aircraft'):
    target = evidence / f'preflight-{dataset}.json'
    run(f'preflight-{dataset}', ['scripts/repro/preflight.py', '--data-root', root / 'data',
                               '--dataset', dataset, '--output', target])
    preflights[dataset] = json.loads(target.read_text())
plans = {}
variants = ['hierarchical_adapter', 'flat_dual_adapter', 'hierarchical_interleaved_adapter',
            'hierarchical_shuffle17_adapter', 'hierarchical_shuffle29_adapter']
for name, output, opts in [
        ('formal-plan', 'ucf101-structure-v1', ['--structure-study', '--seeds', '1', '2', '3']),
        ('smoke-plan', 'structure-smoke-v1', ['--variants', *variants, '--seeds', '1', '--epochs', '1'])]:
    plan = json.loads(run(name, ['scripts/repro/run_pilot.py', '--data-root', root / 'data',
                                '--output-root', root / 'runs' / output, *opts, '--plan-only']))
    (evidence / f'{name}.json').write_text(json.dumps(plan, indent=2))
    plans[name] = len(plan['commands'])
    assert not (root / 'runs' / output).exists()
guard_output = root / 'runs/cpu-guard-structure-v1'
guard = run('cuda-guard', ['scripts/repro/run_pilot.py', '--data-root', root / 'data',
                         '--output-root', guard_output, '--structure-study', '--seeds', '1', '2', '3'], expected=1)
assert 'CUDA is unavailable' in guard and not guard_output.exists()
run('train-help', ['train.py', '--help'])
clip_hash = sha(root / 'assets/ViT-B-16.pt')
assert clip_hash == '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'
activation = root / 'activate-structure-v1.sh'
assert not activation.exists()
activation.write_text(f'''# Source this file to use the prepared structure release.
source {shlex.quote(str(root / '.venv/bin/activate'))}
export PYTHONPATH={shlex.quote(str(release) + ':' + str(release / 'Dassl.pytorch'))}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
cd {shlex.quote(str(release))}
''')
files = ['train.py', 'trainers/textconsistency_common.py', 'trainers/textconsistency_runtime.py',
         'scripts/repro/run_pilot.py', 'scripts/repro/summarize.py', 'scripts/repro/structure_spec.py',
         'tests/repro/test_structure.py', 'tests/repro/test_trainers.py', 'docs/STRUCTURE_CONTROLS.md']
ready = {'state': 'cpu_preparation_complete', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'code_commit': '13b34978eeb8b23532b53abd1066af4719a0f909', 'source_sha256': expected_source,
         'archive_sha256': expected_archive, 'release': str(release), 'activation': str(activation),
         'runtime': runtime, 'memory_limit_bytes': int(memory_limit), 'tests_passed': 46,
         'checks': records, 'datasets': preflights, 'plan_commands': plans,
         'clip_sha256': clip_hash, 'files': {name: sha(release / name) for name in files},
         'formal_training_started': False, 'new_gpu_smoke_passed': False,
         'next_action': 'User enables GPU; verify runtime then run isolated smoke before formal study'}
(evidence / 'ready.json').write_text(json.dumps(ready, indent=2))
print(json.dumps({'state': ready['state'], 'evidence': str(evidence), 'tests': 46,
                  'source_sha256': expected_source, 'plans': plans}), flush=True)
