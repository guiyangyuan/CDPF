"""Complete the bounded CPU preparation without launching GPU experiments."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path('/root/autodl-tmp/textconsistency')
OPS = ROOT / 'releases/crossdataset-ops-20260911-6ce9c6e'
SOURCE = ROOT / 'releases/structure-20260910-v1'
LOGS = ROOT / 'setup-logs/crossdataset-prep-20260911-v2'
PREPARED = LOGS / 'prepared-v2'
STUDY = ROOT / 'runs/crossdataset-study-20260910-v1'
PYTHON = str(ROOT / '.venv/bin/python')
ENV = os.environ.copy()
ENV.update(PYTHONPATH=f'{OPS}:{OPS}/Dassl.pytorch', OMP_NUM_THREADS='4',
           MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4',
           CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
ENV.pop('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', None)


def main():
    status_path = LOGS / 'driver-status.json'
    assert not status_path.exists(), 'Do not restart this driver over existing evidence'
    state = {'state': 'running', 'pid': os.getpid(), 'started_unix': time.time(), 'steps': [],
             'mode': 'cpu_preparation_only', 'gpu_training_started': False}

    def save():
        temp = status_path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(state, indent=2)); temp.replace(status_path)

    def run(name, command, expected=0, env=None):
        record = {'name': name, 'command': command, 'started_unix': time.time(),
                  'expected_returncode': expected, 'log': str(LOGS / (name + '.log'))}
        state['steps'].append(record); state['stage'] = name; save()
        print(json.dumps({'stage': name, 'state': 'running'}), flush=True)
        with Path(record['log']).open('x') as log:
            result = subprocess.run(command, cwd=OPS, env=env or ENV, stdout=log, stderr=subprocess.STDOUT)
        record.update(returncode=result.returncode, seconds=time.time()-record['started_unix']); save()
        if result.returncode != expected:
            raise RuntimeError(f'{name} exited {result.returncode}; inspect {record["log"]}')
        return Path(record['log']).read_text()

    save()
    try:
        assert not STUDY.exists() and not PREPARED.exists()
        common = [PYTHON, '-B', str(OPS / 'scripts/repro/cross_dataset_study.py'),
                  '--source-root', str(SOURCE), '--data-root', str(ROOT / 'data'),
                  '--output-root', str(STUDY), '--preparation-dir', str(PREPARED),
                  '--clip-path', str(ROOT / 'assets/ViT-B-16.pt')]
        planned = json.loads(run('plan-only', common + ['--plan-only']))
        assert [len(s['commands']) for s in planned['stages']] == [17, 17, 51, 51]
        assert not STUDY.exists() and not PREPARED.exists()
        (LOGS / 'plan.json').write_text(json.dumps(planned, indent=2))
        modules = ['test_assets', 'test_components', 'test_preflight', 'test_regressions',
                   'test_runner', 'test_runtime_compatibility', 'test_structure', 'test_trainers',
                   'test_cross_dataset']
        test_log = run('server-tests', [PYTHON, '-B', '-m', 'unittest',
                                       *['tests.repro.' + n for n in modules], '-v'])
        match = re.search(r'Ran (\d+) tests?', test_log)
        assert match and int(match.group(1)) == 59 and '\nOK\n' in test_log
        run('prepare-only', common + ['--prepare-only'])
        prepared = json.loads((PREPARED / 'preparation.json').read_text())
        assert prepared['protocol'] == planned and not prepared['environment']['cuda_available']
        guard_env = dict(ENV, CUDA_VISIBLE_DEVICES='')
        guard_log = run('no-cuda-guard', common + ['--execute'], expected=1, env=guard_env)
        assert 'CUDA with at least one visible GPU is required' in guard_log
        assert not STUDY.exists(), 'No-GPU guard unexpectedly created a training study'
        ready = {'state': 'cpu_preparation_complete', 'created_unix': time.time(),
                 'code_commit': '6ce9c6e2770c6359dcd1f8dfea326e3755aaa79c',
                 'model_code_commit': '13b34978eeb8b23532b53abd1066af4719a0f909',
                 'model_source_sha256': prepared['model_source_sha256'],
                 'operational_release': str(OPS), 'model_source_release': str(SOURCE),
                 'preparation_directory': str(PREPARED), 'planned_study_directory': str(STUDY),
                 'environment': prepared['environment'], 'tests_passed': 59,
                 'step_records': state['steps'], 'no_gpu_guard_passed': True,
                 'gpu_smoke_executed': False, 'gpu_formal_executed': False,
                 'phase_command_counts': [17, 17, 51, 51],
                 'smoke_training_count': 6, 'smoke_evaluation_count': 28,
                 'formal_training_count': 18, 'formal_evaluation_count': 84,
                 'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'image_evidence_scope': 'path/size/mtime inventory; not per-image content SHA256'}
        (LOGS / 'ready.json').write_text(json.dumps(ready, indent=2))
        state.update(state='completed', stage='awaiting_user_gpu', finished_unix=time.time())
        print(json.dumps({'state': state['state'], 'ready': str(LOGS / 'ready.json')}), flush=True)
    except BaseException as exc:
        state.update(state='failed', error=repr(exc), finished_unix=time.time())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
