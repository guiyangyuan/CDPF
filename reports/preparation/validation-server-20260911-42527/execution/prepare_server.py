"""One-off no-GPU deployment and real CLI verification. Never trains a model."""
from pathlib import Path
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path('/root/autodl-tmp/textconsistency')
INBOX = ROOT / 'inbox/validation-prep-20260911-42527'
RELEASE = ROOT / 'releases/validation-ops-20260911-6945bf7'
INPUTS = ROOT / 'reused-inputs/validation-probe-20260911'
PREP = ROOT / 'setup-logs/validation-prep-20260911-42527'
STUDY = ROOT / 'runs/validation-probe-20260911-v1'
PYTHON = ROOT / '.venv/bin/python'
INVENTORY = RELEASE / 'reports/diagnostics/error-analysis-20260911/evidence/proposed-probe-inputs.json'
SOURCES = {'ucf101': ROOT / 'src', 'dtd': ROOT / 'releases/structure-20260910-v1',
           'fgvc_aircraft': ROOT / 'releases/structure-20260910-v1'}
ENV = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '',
       'PYTHONPATH': f'{RELEASE}:{RELEASE}/Dassl.pytorch', 'OMP_NUM_THREADS': '1',
       'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def snapshot(match):
    groups = {}
    directories = {'source_ucf101': SOURCES['ucf101'], 'source_structure': SOURCES['dtd']}
    directories.update({f'data_{d}': ROOT / 'data' / d for d in SOURCES})
    for name, directory in directories.items():
        rows = []
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or '.git' in path.parts:
                continue
            stat = path.stat()
            row = {'path': str(path.relative_to(directory)), 'bytes': stat.st_size,
                   'mtime_ns': stat.st_mtime_ns}
            if path.is_symlink():
                row['resolved'] = str(path.resolve())
            if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                row['sha256'] = sha(path)
            rows.append(row)
        groups[name] = {'file_count': len(rows), 'entries': rows}
    checkpoint_files = {}
    for item in match['matched']:
        old = Path(item['source']['path'])
        new = INPUTS / item['workspace_relative_checkpoint']
        for checkpoint in (old, new):
            for path in (checkpoint, checkpoint.with_name(checkpoint.name + '.sha256'),
                         checkpoint.parent.parent / 'run_manifest.json',
                         checkpoint.parent.parent / 'effective_config.yaml'):
                checkpoint_files[str(path)] = sha(path)
    groups['checkpoint_files'] = checkpoint_files
    return groups


def run(command, log_name, expected=0):
    log = PREP / log_name
    if log.exists():
        raise FileExistsError(log)
    started = time.monotonic()
    with log.open('w') as stream:
        process = subprocess.run(command, cwd=RELEASE, env=ENV, stdout=stream,
                                 stderr=subprocess.STDOUT)
    record = {'command': command, 'returncode': process.returncode,
              'seconds': time.monotonic() - started, 'log': str(log), 'log_sha256': sha(log)}
    state['records'].append(record)
    write(PREP / 'execution.json', state)
    print(json.dumps({'log': log_name, 'returncode': process.returncode,
                      'seconds': record['seconds']}), flush=True)
    if process.returncode != expected:
        raise RuntimeError(f'{log_name}: exit {process.returncode}, expected {expected}')
    return record


def execute():
    # Release the framework probe process before launching tests/model checks:
    # this host has only 2 GiB in no-GPU mode.
    resource_code = ('import json,torch,torchvision,numpy,PIL,yaml; '
        'print(json.dumps({"torch":torch.__version__,"torchvision":torchvision.__version__, '
        '"numpy":numpy.__version__,"pillow":PIL.__version__,"yaml":yaml.__version__, '
        '"cuda_available":torch.cuda.is_available()}))')
    framework = json.loads(subprocess.check_output(
        [str(PYTHON), '-B', '-c', resource_code], text=True,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1'}))
    if framework['cuda_available']:
        raise RuntimeError('Expected no-GPU mode for this preparation')
    if STUDY.exists():
        raise FileExistsError(f'Planned future study already exists: {STUDY}')
    for directory in (RELEASE, INPUTS, PREP):
        if directory.exists():
            raise FileExistsError(directory)
    PREP.mkdir()
    global state
    state = {'state': 'deploying', 'records': [], 'gpu_export_run': False,
             'new_training_runs': 0, 'code_commit': '6945bf721f71988a8b4e8b4db3eab2b6858a6bd3'}
    write(PREP / 'execution.json', state)
    resource = {**framework, 'hostname': platform.node(), 'python': platform.python_version(),
                'memory_max':
                Path('/sys/fs/cgroup/memory.max').read_text().strip(),
                'disk_free_bytes': shutil.disk_usage(ROOT).free}
    write(PREP / 'environment.json', resource)
    package = json.loads((INBOX / 'ops-package.json').read_text())
    archive = INBOX / Path(package['archive']).name
    if sha(archive) != package['sha256']:
        raise ValueError('Operations archive hash mismatch')
    RELEASE.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(RELEASE, filter='data')
    if sha(INVENTORY) != '858bfbf1f64d71e93c5ca89f706bd757ca69b1f6250b2cadb55a521dcda4d185':
        raise ValueError('Fixed inventory hash mismatch')
    inventory = json.loads(INVENTORY.read_text())
    expected = {(c['dataset'], c['seed'], role): entry for c in inventory['cases']
                for role, entry in c['checkpoints'].items()}
    match = json.loads((INBOX / 'input-match.json').read_text())
    if match['missing'] or len(match['matched']) != 27:
        raise ValueError('Incomplete model mapping')
    if {(r['dataset'], r['seed'], r['role']) for r in match['matched']} != set(expected):
        raise ValueError('Model mapping cases mismatch')
    INPUTS.mkdir()
    for row in match['matched']:
        entry = expected[(row['dataset'], row['seed'], row['role'])]
        if row['workspace_relative_checkpoint'] != entry['workspace_relative_checkpoint']:
            raise ValueError('Checkpoint target differs from inventory')
        old = Path(row['source']['path']).resolve()
        old.relative_to(ROOT)
        new = INPUTS / entry['workspace_relative_checkpoint']
        new.resolve().relative_to(INPUTS)
        pairs = [(old, new, entry['sha256']),
                 (old.with_name(old.name + '.sha256'), new.with_name(new.name + '.sha256'),
                  row['source']['sidecar_sha256']),
                 (old.parent.parent / 'run_manifest.json', new.parent.parent / 'run_manifest.json',
                  entry['manifest_sha256']),
                 (old.parent.parent / 'effective_config.yaml', new.parent.parent / 'effective_config.yaml',
                  row['source']['config_sha256'])]
        for source, target, digest in pairs:
            if sha(source) != digest or target.exists():
                raise ValueError(f'Changed source or existing destination: {source}')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if sha(target) != digest:
                raise ValueError(f'Copy hash mismatch: {target}')
    write(PREP / 'deployment.json', {'release': str(RELEASE), 'workspace_root': str(INPUTS),
          'source_roots': {k: str(v) for k, v in SOURCES.items()}, 'data_root': str(ROOT / 'data'),
          'package': package, 'copied_model_bundles': 27, 'input_match': match})
    before = snapshot(match)
    write(PREP / 'inputs-before.json', before)
    tests = run([str(PYTHON), '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'], 'tests.log')
    if not re.search(r'Ran 128 tests in ', (PREP / 'tests.log').read_text()):
        raise ValueError('Expected full 128-test suite')
    cases = []
    future = []
    for dataset in SOURCES:
        for seed in (1, 2, 3):
            name = f'{dataset}-seed{seed}'
            output = PREP / 'prepared' / name
            command = [str(PYTHON), '-B', str(RELEASE / 'scripts/repro/validation_probe.py'),
                       'prepare', '--inventory', str(INVENTORY), '--workspace-root', str(INPUTS),
                       '--source-root', str(SOURCES[dataset]), '--data-root', str(ROOT / 'data'),
                       '--dataset', dataset, '--seed', str(seed), '--output-dir', str(output)]
            state['state'] = 'preparing_' + name
            run(command, name + '.log')
            receipt = json.loads((output / 'prepared.json').read_text())
            cases.append({'dataset': dataset, 'seed': seed, 'prepared': str(output / 'prepared.json'),
                          'sha256': sha(output / 'prepared.json'),
                          'validation_count': len(receipt['samples']['val']),
                          'sample_hashes': receipt['sample_hashes'],
                          'decoded_val': receipt['transform']['decoded_val']})
            future.append({'dataset': dataset, 'seed': seed,
                'export': [str(PYTHON), '-B', str(RELEASE / 'scripts/repro/validation_probe.py'),
                           'export', '--prepared', str(output / 'prepared.json'), '--output-dir',
                           str(STUDY / 'exports' / name)],
                'analyze': [str(PYTHON), '-B', str(RELEASE / 'scripts/repro/validation_analysis.py'),
                            'analyze', '--export', str(STUDY / 'exports' / name / 'export.json'),
                            '--output-dir', str(STUDY / 'analysis' / name)]})
    rejection = PREP / 'no-cuda-rejection'
    run([str(PYTHON), '-B', str(RELEASE / 'scripts/repro/validation_probe.py'), 'export',
         '--prepared', cases[0]['prepared'], '--output-dir', str(rejection)], 'no-cuda.log', expected=1)
    failure = json.loads((rejection / 'failure.json').read_text())
    if 'CUDA is required' not in failure['error'] or list(rejection.glob('*.npz')) or (rejection / 'export.json').exists():
        raise ValueError('Expected explicit no-CUDA refusal without logits/success')
    after = snapshot(match)
    write(PREP / 'inputs-after.json', after)
    if before != after:
        raise ValueError('Immutable source/data/model inputs changed')
    if STUDY.exists():
        raise ValueError('CPU preparation unexpectedly created the future GPU study')
    write(PREP / 'gpu-plan.json', {'state': 'not_executed', 'new_training_runs': 0,
        'cwd': str(RELEASE), 'pythonpath': f'{RELEASE}:{RELEASE}/Dassl.pytorch', 'cases': future,
        'aggregate': [str(PYTHON), '-B', str(RELEASE / 'scripts/repro/validation_analysis.py'),
                      'aggregate', '--analyses', *[str(STUDY / 'analysis' / f'{x["dataset"]}-seed{x["seed"]}' / 'analysis.json') for x in cases],
                      '--output-dir', str(STUDY / 'aggregate')]})
    write(PREP / 'ready.json', {'state': 'server_cpu_preparation_complete_gpu_pending',
        'code_commit': state['code_commit'], 'environment': resource, 'cases': cases,
        'tests': tests, 'full_test_count': 128, 'models': 27,
        'validation_records': sum(c['validation_count'] for c in cases),
        'inputs_unchanged': True, 'before_sha256': sha(PREP / 'inputs-before.json'),
        'after_sha256': sha(PREP / 'inputs-after.json'), 'cuda_unavailable_refusal_passed': True,
        'gpu_study_exists': False, 'gpu_export_run': False, 'new_training_runs': 0,
        'tools_sha256': {p.name: sha(p) for p in sorted((RELEASE / 'scripts/repro').glob('validation_*.py'))},
        'limitations': ['Actual CUDA loading/forward/export is unexecuted',
                        'No new method accuracy or novelty evidence',
                        'Image immutability snapshot uses path/size/mtime; transformed validation tensors separately recorded']})
    state['state'] = 'server_cpu_preparation_complete_gpu_pending'
    write(PREP / 'execution.json', state)
    print(json.dumps({'state': state['state'], 'cases': len(cases), 'models': 27,
                      'validation_records': sum(c['validation_count'] for c in cases)}), flush=True)


if __name__ == '__main__':
    try:
        execute()
    except BaseException as exc:
        if PREP.is_dir():
            write(PREP / 'failure.json', {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
        raise
