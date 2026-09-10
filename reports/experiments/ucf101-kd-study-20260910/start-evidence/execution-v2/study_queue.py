"""Bounded study queue: official matched20, then two controlled KD jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--resume-checkpoint-sha')
    a = p.parse_args()
    root, study = a.root.resolve(), a.study.resolve()
    if a.resume_checkpoint_sha:
        old_state = json.loads((study / 'status.json').read_text())
        assert old_state['state'] == 'failed' and old_state['stage'] == 'official_matched20'
        assert not (study / 'kd0').exists() and not (study / 'kd1').exists()
        assert not (study / 'status-first-failure.json').exists()
        (study / 'status-first-failure.json').write_text(json.dumps(old_state, indent=2))
        (study / 'official.log').rename(study / 'official-first-attempt.log')
    else:
        study.mkdir(parents=True, exist_ok=False)
    source = root / 'releases/cpu-prep-20260910'
    env = os.environ.copy()
    env.update(PYTHONPATH=f'{source}:{source}/Dassl.pytorch', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    state = {'state': 'running', 'pid': os.getpid(), 'started_unix': time.time(), 'stage': 'official_matched20',
             'project_commit': (source / 'DEPLOYED_COMMIT').read_text().strip(), 'completed': []}
    def save():
        tmp = study / 'status.json.tmp'
        tmp.write_text(json.dumps(state, indent=2)); tmp.replace(study / 'status.json')
    save()
    try:
        command = [sys.executable, '-u', str(Path(__file__).with_name('official_runner.py')),
                   '--root', str(root), '--output', str(study / 'official_matched20')]
        if a.resume_checkpoint_sha:
            command += ['--resume-checkpoint-sha', a.resume_checkpoint_sha]
        with (study / 'official.log').open('w') as log:
            subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        state['completed'].append('official_matched20'); state['stage'] = 'kd0_and_kd1'; save()
        jobs = []
        for kd in (0, 1):
            command = [sys.executable, '-u', str(source / 'scripts/repro/run_pilot.py'), '--dataset', 'ucf101',
                       '--data-root', str(root / 'data'), '--output-root', str(study / f'kd{kd}'), '--seeds', '1', '2', '3',
                       '--variants', 'hierarchical', 'hierarchical_adapter', '--kd-weight', str(kd)]
            log = (study / f'kd{kd}.log').open('w')
            jobs.append((kd, subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        state['jobs'] = {f'kd{kd}': process.pid for kd, process, _ in jobs}
        save()
        failures = []
        for kd, process, log in jobs:
            code = process.wait(); log.close()
            if code: failures.append(f'kd{kd} exit {code}')
            else: state['completed'].append(f'kd{kd}')
            save()
        if failures: raise RuntimeError('; '.join(failures))
        state.update(state='completed', stage='awaiting_analysis', finished_unix=time.time())
    except BaseException as exc:
        state.update(state='failed', error=repr(exc), finished_unix=time.time())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
