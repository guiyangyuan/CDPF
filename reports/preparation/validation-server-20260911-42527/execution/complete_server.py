"""Finish the remaining case; retain passed cases and all interrupted attempts."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess

ROOT = Path('/root/autodl-tmp/textconsistency')
FIRST = ROOT / 'setup-logs/validation-prep-20260911-42527'
V2 = ROOT / 'setup-logs/validation-prep-20260911-42527-v2'
RETRY = ROOT / 'setup-logs/validation-prep-20260911-42527-resume'
CACHE = ROOT / 'setup-logs/validation-prep-20260911-42527-cache'
FINAL = ROOT / 'setup-logs/validation-prep-20260911-42527-final'
HELPER = ROOT / 'inbox/validation-prep-20260911-42527-v2/prepare_server_v2.py'


def main():
    spec = importlib.util.spec_from_file_location('preparation_support', HELPER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if FINAL.exists() or m.STUDY.exists():
        raise FileExistsError('Fresh final preparation and absent GPU study required')
    v2 = json.loads((V2 / 'execution.json').read_text())
    resumed = json.loads((CACHE / 'isolated-retry.json').read_text())
    assert v2['records'][0]['returncode'] == 0 and resumed['returncode'] == 0
    assert len([r for r in v2['records'] if r['returncode'] == 0]) == 8
    FINAL.mkdir()
    m.PREP = FINAL
    m.ENV.update(MALLOC_ARENA_MAX='2', MALLOC_TRIM_THRESHOLD_='131072')
    m.state = {'state': 'finishing_last_case', 'records': [], 'gpu_export_run': False,
               'new_training_runs': 0, 'code_commit': 'de57efd7452e4feab3fd361ef9acac27a6dee4ef'}
    match = json.loads((m.INBOX / 'input-match.json').read_text())
    files = set()
    for row in match['matched']:
        files.add(Path(row['source']['path']).resolve())
        files.add((m.INPUTS / row['workspace_relative_checkpoint']).resolve())
    for source in set(m.SOURCES.values()):
        files.add((source / 'clip/ViT-B-16.pt').resolve())
    before = int(Path('/sys/fs/cgroup/memory.current').read_text())
    for path in sorted(files):
        path.relative_to(ROOT)
        with path.open('rb') as stream:
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    m.write(FINAL / 'cache-advice.json', {'operation': 'POSIX_FADV_DONTNEED via read-only descriptors',
        'files': [str(p) for p in sorted(files)], 'memory_before': before,
        'memory_after': int(Path('/sys/fs/cgroup/memory.current').read_text()),
        'allocator_environment': {'MALLOC_ARENA_MAX': '2', 'MALLOC_TRIM_THRESHOLD_': '131072'}})
    command = list(v2['records'][-1]['command'])
    command.insert(2, '-u')
    command[command.index('--seed') + 1] = '3'
    command[-1] = str(FINAL / 'prepared/fgvc_aircraft-seed3')
    m.run(command, 'fgvc_aircraft-seed3.log')
    cases = []
    future = []
    for dataset in m.SOURCES:
        for seed in (1, 2, 3):
            base = CACHE if (dataset, seed) == ('fgvc_aircraft', 2) else (
                FINAL if (dataset, seed) == ('fgvc_aircraft', 3) else V2)
            name = f'{dataset}-seed{seed}'
            path = base / 'prepared' / name / 'prepared.json'
            receipt = json.loads(path.read_text())
            assert receipt['status'] == 'prepared' and receipt['dataset_key'] == dataset and receipt['seed'] == seed
            cases.append({'dataset': dataset, 'seed': seed, 'prepared': str(path), 'sha256': m.sha(path),
                          'validation_count': len(receipt['samples']['val']),
                          'sample_hashes': receipt['sample_hashes'],
                          'decoded_val': receipt['transform']['decoded_val']})
            future.append({'dataset': dataset, 'seed': seed,
                'export': [str(m.PYTHON), '-B', str(m.RELEASE / 'scripts/repro/validation_probe.py'),
                           'export', '--prepared', str(path), '--output-dir', str(m.STUDY / 'exports' / name)],
                'analyze': [str(m.PYTHON), '-B', str(m.RELEASE / 'scripts/repro/validation_analysis.py'),
                            'analyze', '--export', str(m.STUDY / 'exports' / name / 'export.json'),
                            '--output-dir', str(m.STUDY / 'analysis' / name)]})
    rejection = FINAL / 'no-cuda-rejection'
    m.run([str(m.PYTHON), '-B', str(m.RELEASE / 'scripts/repro/validation_probe.py'),
           'export', '--prepared', cases[0]['prepared'], '--output-dir', str(rejection)], 'no-cuda.log', expected=1)
    failure = json.loads((rejection / 'failure.json').read_text())
    assert 'CUDA is required' in failure['error'] and not list(rejection.glob('*.npz'))
    assert not (rejection / 'export.json').exists()
    after = m.snapshot(match)
    m.write(FINAL / 'inputs-after.json', after)
    assert after == json.loads((FIRST / 'inputs-before.json').read_text())
    assert after == json.loads((V2 / 'inputs-before.json').read_text())
    assert not m.STUDY.exists()
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                         capture_output=True, text=True)
    assert gpu.returncode != 0 and 'No devices were found' in gpu.stdout + gpu.stderr
    m.write(FINAL / 'gpu-plan.json', {'state': 'not_executed', 'new_training_runs': 0,
        'cwd': str(m.RELEASE), 'pythonpath': f'{m.RELEASE}:{m.RELEASE}/Dassl.pytorch', 'cases': future,
        'aggregate': [str(m.PYTHON), '-B', str(m.RELEASE / 'scripts/repro/validation_analysis.py'),
                      'aggregate', '--analyses', *[str(m.STUDY / 'analysis' / f'{x["dataset"]}-seed{x["seed"]}' / 'analysis.json') for x in cases],
                      '--output-dir', str(m.STUDY / 'aggregate')]})
    successful = [r for r in v2['records'] if r['returncode'] == 0]
    successful.append({k: resumed[k] for k in ('command', 'returncode', 'seconds', 'log', 'log_sha256')})
    successful.extend(m.state['records'])
    assert len(successful) == 11
    m.write(FINAL / 'ready.json', {'state': 'server_cpu_preparation_complete_gpu_pending',
        'code_commit': m.state['code_commit'], 'environment': json.loads((V2 / 'environment.json').read_text()),
        'cases': cases, 'tests': v2['records'][0], 'full_test_count': 128, 'models': 27,
        'validation_records': sum(c['validation_count'] for c in cases), 'inputs_unchanged': True,
        'before_snapshot': str(FIRST / 'inputs-before.json'), 'before_sha256': m.sha(FIRST / 'inputs-before.json'),
        'after_sha256': m.sha(FINAL / 'inputs-after.json'), 'cuda_unavailable_refusal_passed': True,
        'gpu_query': {'returncode': gpu.returncode, 'stdout': gpu.stdout, 'stderr': gpu.stderr},
        'gpu_study_exists': False, 'gpu_export_run': False, 'new_training_runs': 0,
        'tools_sha256': {p.name: m.sha(p) for p in sorted((m.RELEASE / 'scripts/repro').glob('validation_*.py'))},
        'interrupted_attempts': [str(V2 / 'failure.json'), str(RETRY / 'isolated-retry.json')],
        'test_path_fix_attempt': str(FIRST / 'failure.json'),
        'limitations': ['Actual CUDA loading/forward/export unexecuted', 'No new method score or novelty evidence',
                        'SIGKILL attribution uncertain; cgroup OOM counters stayed zero',
                        'Images snapshot records path/size/mtime; validation image bytes are checked separately']})
    m.state['records'] = successful
    m.state['state'] = 'server_cpu_preparation_complete_gpu_pending'
    m.write(FINAL / 'execution.json', m.state)
    print(json.dumps({'state': m.state['state'], 'cases': 9, 'models': 27, 'validation_records': 1500}), flush=True)


if __name__ == '__main__':
    main()
