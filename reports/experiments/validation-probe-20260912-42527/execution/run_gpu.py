"""Execute the frozen nine-case inference plan, audit inputs, preserve all outputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
import traceback

ROOT = Path('/root/autodl-tmp/textconsistency')
PREP = ROOT / 'setup-logs/validation-prep-20260911-42527-final'
RELEASE = ROOT / 'releases/validation-ops-20260911-de57efd'
PYTHON = ROOT / '.venv/bin/python'
STUDY = ROOT / 'runs/validation-probe-20260911-v1'
INBOX = ROOT / 'inbox/validation-gpu-20260912-42527'
BACKUP = ROOT / 'backups/validation-gpu-20260912-42527.tar.gz'
ENV = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '0',
       'PYTHONPATH': f'{RELEASE}:{RELEASE}/Dassl.pytorch', 'OMP_NUM_THREADS': '1',
       'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
state = {}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def save():
    state['updated_at'] = now()
    write(STUDY / 'status.json', state)


def gpu_query():
    args = ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version', '--format=csv,noheader']
    return subprocess.check_output(args, text=True).strip()


def run(command, name):
    log = STUDY / 'logs' / (name + '.log')
    record = {'name': name, 'command': command, 'started_at': now(), 'log': str(log)}
    state['active'] = record
    save()
    began = time.monotonic()
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=RELEASE, env=ENV, stdout=stream, stderr=subprocess.STDOUT)
        record['pid'] = process.pid
        save()
        code = process.wait()
    record.update(returncode=code, seconds=time.monotonic() - began, ended_at=now(), log_sha256=sha(log))
    state['records'].append(record)
    state['active'] = None
    save()
    print(json.dumps({'name': name, 'returncode': code, 'seconds': record['seconds']}), flush=True)
    if code != 0:
        raise RuntimeError(f'{name}: exit {code}; see preserved log')


def image_hashes(expected):
    actual = {}
    for sample in sorted(expected):
        image = (ROOT / 'data' / sample).resolve()
        image.relative_to(ROOT / 'data')
        actual[sample] = {'sha256': sha(image), 'bytes': image.stat().st_size}
    if actual != expected:
        raise ValueError('Validation image bytes changed from CPU preparation')
    return actual


def main():
    global state
    if STUDY.exists() or BACKUP.exists():
        raise FileExistsError('Fresh study and backup required; inspect prior state, do not overwrite')
    expected = read(INBOX / 'expected-inputs.json')
    for name, digest in expected.items():
        assert sha(PREP / name) == digest, name
    ready, plan = read(PREP / 'ready.json'), read(PREP / 'gpu-plan.json')
    assert ready['state'] == 'server_cpu_preparation_complete_gpu_pending'
    assert plan['state'] == 'not_executed' and len(plan['cases']) == 9
    assert plan['cwd'] == str(RELEASE) and plan['pythonpath'] == ENV['PYTHONPATH']
    for name, digest in ready['tools_sha256'].items():
        assert sha(RELEASE / 'scripts/repro' / name) == digest
    for c in ready['cases']:
        assert sha(c['prepared']) == c['sha256']
    for c in plan['cases']:
        for kind in ('export', 'analyze'):
            command = c[kind]
            assert command[:2] == [str(PYTHON), '-B']
            assert command[3] == kind
            Path(command[2]).relative_to(RELEASE / 'scripts/repro')
            out = Path(command[command.index('--output-dir') + 1])
            out.relative_to(STUDY)
            assert not out.exists()
    active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('GPU already has compute processes; inspect before proceeding')
    STUDY.mkdir()
    (STUDY / 'logs').mkdir()
    state = {'state': 'verifying_inputs', 'started_at': now(), 'pid': os.getpid(),
             'code_commit': ready['code_commit'], 'tools_sha256': ready['tools_sha256'],
             'runner_sha256': sha(__file__), 'expected_preparation_sha256': expected,
             'records': [], 'active': None, 'new_training_runs': 0}
    save()
    started = time.monotonic()
    try:
        spec = importlib.util.spec_from_file_location('snapshot_support', ROOT / 'inbox/validation-prep-20260911-42527-v2/prepare_server_v2.py')
        support = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(support)
        matched = read(ROOT / 'inbox/validation-prep-20260911-42527-v2/input-match.json')
        before = support.snapshot(matched)
        assert before == read(PREP / 'inputs-after.json'), 'Inputs changed since CPU preparation'
        write(STUDY / 'inputs-before.json', before)
        images = image_hashes(read(PREP / 'validation-image-hashes.json'))
        write(STUDY / 'validation-image-hashes.json', images)
        probe = 'import json,torch,numpy; assert torch.cuda.is_available(); print(json.dumps(dict(torch=torch.__version__,numpy=numpy.__version__,cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(0),cuda_available=True)))'
        environment = json.loads(subprocess.check_output([str(PYTHON), '-B', '-c', probe], env=ENV, text=True))
        environment.update(python=platform.python_version(), hostname=platform.node(), gpu_query=gpu_query(),
                           memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip())
        write(STUDY / 'environment.json', environment)
        write(STUDY / 'executed-plan.json', plan)
        state['state'] = 'running_frozen_validation_inference'
        save()
        for c in plan['cases']:
            name = f"{c['dataset']}-seed{c['seed']}"
            run(c['export'], name + '-export')
            exported = read(STUDY / 'exports' / name / 'export.json')
            assert exported['status'] == 'passed' and exported['environment']['device'] == 'cuda'
            assert exported['teacher_tensor_exact_equal'] and exported['teacher_logits_exact_equal']
            run(c['analyze'], name + '-analyze')
        run(plan['aggregate'], 'aggregate')
        aggregate = read(STUDY / 'aggregate/aggregate.json')
        assert aggregate['status'] == 'passed' and aggregate['case_count'] == 9
        state['state'] = 'checking_inputs_after_inference'
        save()
        after = support.snapshot(matched)
        write(STUDY / 'inputs-after.json', after)
        assert before == after
        assert image_hashes(images) == images
        assert len(state['records']) == 19 and all(x['returncode'] == 0 for x in state['records'])
        for c in ready['cases']:
            assert sha(c['prepared']) == c['sha256']
        for name, digest in ready['tools_sha256'].items():
            assert sha(RELEASE / 'scripts/repro' / name) == digest
        state.update(state='gpu_execution_and_audit_passed_local_backup_pending', ended_at=now(),
                     wall_seconds=time.monotonic() - started, inputs_unchanged=True,
                     validation_image_hashes_unchanged=True, fixed_candidate=aggregate['fixed_candidate'],
                     guards=aggregate['guards'], deltas_pp=aggregate['deltas_pp'], gpu_final=gpu_query())
        save()
        files = []
        with tarfile.open(BACKUP, 'x:gz') as tar:
            for p in sorted(STUDY.rglob('*')):
                if not p.is_file():
                    continue
                assert not p.is_symlink() and p.suffix in ('.json', '.log', '.npz')
                name = str(Path(STUDY.name) / p.relative_to(STUDY))
                tar.add(p, arcname=name, recursive=False)
                files.append({'path': name, 'bytes': p.stat().st_size, 'sha256': sha(p)})
        write(STUDY / 'backup-receipt.json', {'archive': str(BACKUP), 'sha256': sha(BACKUP),
              'bytes': BACKUP.stat().st_size, 'files': files, 'new_training_runs': 0,
              'contains': 'all nine six-branch logits, all analyses, aggregate, commands/logs and input audit; no images or model weights'})
        print(json.dumps({'state': state['state'], 'guards': state['guards'], 'backup': str(BACKUP)}), flush=True)
    except Exception as exc:
        state.update(state='failed', ended_at=now(), error=f'{type(exc).__name__}: {exc}')
        save()
        write(STUDY / 'failure.json', {'error': state['error'], 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
