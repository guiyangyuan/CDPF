"""Verify remote archive and independently recompute every case from raw logits."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tarfile
import time
import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
REPO = ROOT / 'work/github-publish/textconsistency_v2_tune'
UNPACKED = WORK / 'unpacked'
STUDY = UNPACKED / 'validation-probe-20260911-v1'
REANALYSIS = WORK / 'reanalysis'
REMOTE = Path('/root/autodl-tmp/textconsistency/runs/validation-probe-20260911-v1')
CPU = ROOT / 'work/validation-server-20260911-42527'
ROUNDOFF = []


def compare_cross_machine(local, remote, path='root'):
    if isinstance(local, dict) and isinstance(remote, dict):
        assert local.keys() == remote.keys(), path
        for key in local:
            compare_cross_machine(local[key], remote[key], path + '.' + key)
    elif isinstance(local, list) and isinstance(remote, list):
        assert len(local) == len(remote), path
        for index, (a, b) in enumerate(zip(local, remote)):
            compare_cross_machine(a, b, f'{path}[{index}]')
    elif local != remote:
        assert type(local) is float and type(remote) is float, (path, local, remote)
        assert any(field in path.split('.') for field in ('nll','multiclass_brier','probability_top1_top2_margin')), (path, local, remote)
        assert math.isfinite(local) and math.isfinite(remote) and abs(local-remote) <= 1e-12, (path, local, remote)
        ROUNDOFF.append({'path':path,'local':local,'remote':remote,'absolute_difference':abs(local-remote)})


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


def mapped(path):
    return STUDY / Path(path).relative_to(REMOTE)


def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def main():
    receipt = read(WORK / 'backup-receipt.json')
    archive = WORK / Path(receipt['archive']).name
    assert sha(archive) == receipt['sha256'] and archive.stat().st_size == receipt['bytes']
    expected = {x['path']: x for x in receipt['files']}
    assert len(expected) == len(receipt['files'])
    if (WORK / 'local-audit.json').exists():
        raise FileExistsError('Completed audit must not be overwritten')
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert len(members) == len(expected) and {x.name for x in members} == set(expected)
        assert all(x.isfile() and not Path(x.name).is_absolute() and '..' not in Path(x.name).parts for x in members)
        if not UNPACKED.exists():
            tar.extractall(UNPACKED, filter='data')
    assert {str(p.relative_to(UNPACKED)) for p in UNPACKED.rglob('*') if p.is_file()} == set(expected)
    for rel, item in expected.items():
        p = UNPACKED / rel
        assert sha(p) == item['sha256'] and p.stat().st_size == item['bytes'], rel
    state = read(STUDY / 'status.json')
    assert state['state'] == 'gpu_execution_and_audit_passed_local_backup_pending'
    assert state['new_training_runs'] == 0 and len(state['records']) == 19
    assert state['runner_sha256'] == sha(WORK / 'run_gpu.py')
    for record in state['records']:
        assert record['returncode'] == 0 and sha(mapped(record['log'])) == record['log_sha256']
    assert read(STUDY / 'inputs-before.json') == read(STUDY / 'inputs-after.json')
    assert read(STUDY / 'inputs-before.json') == read(CPU / 'unpacked/validation-prep-20260911-42527-final/inputs-after.json')
    assert read(STUDY / 'validation-image-hashes.json') == read(CPU / 'local-validation-image-hashes.json')
    assert not list(STUDY.rglob('failure.json'))
    for name, digest in state['tools_sha256'].items():
        assert sha(REPO / 'scripts/repro' / name) == digest
    ready = read(CPU / 'unpacked/validation-prep-20260911-42527-final/ready.json')
    expected_cases = {(d, s) for d in ('ucf101', 'dtd', 'fgvc_aircraft') for s in (1, 2, 3)}
    exports = sorted((STUDY / 'exports').glob('*/export.json'))
    assert len(exports) == 9
    assert {(read(x)['dataset_key'], read(x)['seed']) for x in exports} == expected_cases
    REANALYSIS.mkdir(exist_ok=True)
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': '',
           'PYTHONPATH': f'{REPO}:{REPO}/Dassl.pytorch', 'OMP_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    execution = read(REANALYSIS / 'execution.json')['records'] if (REANALYSIS / 'execution.json').exists() else []
    assert len({x['log'] for x in execution}) == len(execution)
    for record in execution:
        assert record['returncode'] == 0 and sha(record['log']) == record['log_sha256']
    case_rows, all_rows, analyses = [], [], []
    def run(command, name):
        log = REANALYSIS / (name + '.log')
        previous = [x for x in execution if x['log'] == str(log)]
        if previous:
            assert len(previous) == 1 and previous[0]['command'] == command
            assert (Path(command[-1]) / ('aggregate.json' if name == 'aggregate' else 'analysis.json')).is_file()
            return
        start = time.monotonic()
        with log.open('x') as f:
            result = subprocess.run(command, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
        record = {'command': command, 'returncode': result.returncode, 'seconds': time.monotonic()-start,
                  'log': str(log), 'log_sha256': sha(log)}
        execution.append(record)
        write(REANALYSIS / 'execution.json', {'records': execution})
        assert result.returncode == 0, log
    for path in exports:
        x = read(path)
        d, s = x['dataset_key'], x['seed']
        case = next(c for c in ready['cases'] if (c['dataset'], c['seed']) == (d, s))
        prepared = read(CPU / 'unpacked' / Path(case['prepared']).relative_to('/root/autodl-tmp/textconsistency/setup-logs'))
        assert x['prepared_sha256'] == case['sha256']
        assert x['tool']['validation_probe_sha256'] == state['tools_sha256']['validation_probe.py']
        assert x['environment']['device'] == 'cuda' and x['environment']['gpu'] == 'NVIDIA GeForce RTX 5090'
        assert x['evaluated_transform'] == prepared['transform']['decoded_val']
        assert x['teacher_tensor_exact_equal'] and x['teacher_logits_exact_equal']
        triples = lambda rows: [(r['sample_id'], r['label'], r['classname']) for r in rows]
        assert triples(x['rows']) == triples(prepared['samples']['val']) and x['class_order'] == prepared['samples']['classes']
        assert sha(path.parent / 'logits.npz') == x['npz_sha256']
        with np.load(path.parent / 'logits.npz', allow_pickle=False) as data:
            assert set(data.files) == {'C','S0','S1','M','T0','T1','labels'}
            labels = data['labels']
            for b in ('C','S0','S1','M','T0','T1'):
                assert data[b].dtype == np.float32 and data[b].shape == (len(labels), len(x['class_order']))
                assert np.isfinite(data[b]).all()
            assert np.array_equal(data['T0'], data['T1'])
            acc = lambda logits: 100.0 * int(np.sum(np.argmax(logits, axis=1) == labels)) / len(labels)
            softmax = lambda a: np.exp(a-a.max(axis=1,keepdims=True)) / np.exp(a-a.max(axis=1,keepdims=True)).sum(axis=1,keepdims=True)
            selected = {name: acc((data[student].astype(np.float64)+data['M'].astype(np.float64))*.5)
                        for name,student in [('S1+M_logits','S1'),('S0+M_logits','S0'),('C+M_logits','C')]}
            selected['C+M_probability'] = acc((softmax(data['C'].astype(np.float64)) + softmax(data['M'].astype(np.float64)))*.5)
        name = f'{d}-seed{s}'
        run([sys.executable,'-B',str(REPO/'scripts/repro/validation_analysis.py'),'analyze','--export',str(path),'--output-dir',str(REANALYSIS/name)],name)
        local_path = REANALYSIS / name / 'analysis.json'
        local, remote = read(local_path), read(STUDY / 'analysis' / name / 'analysis.json')
        compare_cross_machine({k:v for k,v in local.items() if k!='export'}, {k:v for k,v in remote.items() if k!='export'}, name)
        for method, accuracy in selected.items():
            assert accuracy == remote['combinations'][method]['accuracy_pct']
        case_rows.append({'dataset':d,'seed':s,'validation_count':len(x['rows']),
                          'forward_precision':json.dumps(x['forward_precision']),
                          'teacher_logits_equal':True,**selected})
        for method, metrics in {**local['branches'], **local['combinations']}.items():
            all_rows.append({'dataset':d,'seed':s,'method':method,**metrics})
        analyses.append(local_path)
    run([sys.executable,'-B',str(REPO/'scripts/repro/validation_analysis.py'),'aggregate','--analyses',*[str(p) for p in analyses],'--output-dir',str(REANALYSIS/'aggregate')],'aggregate')
    local, remote = read(REANALYSIS/'aggregate/aggregate.json'), read(STUDY/'aggregate/aggregate.json')
    compare_cross_machine({k:v for k,v in local.items() if k not in ('inputs','trusted_inventory')}, {k:v for k,v in remote.items() if k not in ('inputs','trusted_inventory')}, 'aggregate')
    assert local['trusted_inventory']['sha256'] == remote['trusted_inventory']['sha256']
    means = {d:{m:statistics.mean(r[m] for r in case_rows if r['dataset']==d) for m in selected} for d in ('ucf101','dtd','fgvc_aircraft')}
    delta = lambda m: {d:means[d]['S1+M_logits']-means[d][m] for d in means}
    guards = {'S1+M_vs_C+M_logits':statistics.mean(delta('C+M_logits').values())>=0 and min(delta('C+M_logits').values())>=-1,
              'S1+M_vs_C+M_probability':statistics.mean(delta('C+M_probability').values())>=0 and min(delta('C+M_probability').values())>=-1,
              'S1+M_vs_S0+M_logits':statistics.mean(delta('S0+M_logits').values())>0}
    guards['all_passed'] = all(guards.values())
    assert guards == remote['guards']
    assert len(all_rows)==216 and sum(r['validation_count'] for r in case_rows)==1500
    csv_write(WORK/'cases.csv',case_rows)
    csv_write(WORK/'all-methods-by-seed.csv',all_rows)
    audit = {'state':'gpu_validation_results_and_local_backup_verified','verified_date':'2026-09-12',
             'archive_sha256':receipt['sha256'],'archive_bytes':receipt['bytes'],'files_verified':len(expected),
             'remote_commands_passed':19,'local_reanalysis_commands_passed':10,'cases':9,'models_reused':27,
             'validation_records':1500,'unique_images':1280,'methods_per_case':24,'method_seed_rows':216,
             'all_inputs_unchanged':True,'all_validation_image_hashes_equal_cpu':True,
             'all_export_identities_and_tensors_verified':True,'raw_logits_independent_accuracy_check_passed':True,
             'all_analysis_fields_and_aggregate_match_with_metric_roundoff_tolerance':True,
             'accuracy_predictions_counts_and_guards_exact_equal_remote':True,
             'cross_machine_derived_metric_absolute_tolerance':1e-12,
             'cross_machine_roundoff_count':len(ROUNDOFF),
             'cross_machine_roundoff_max_absolute':max((r['absolute_difference'] for r in ROUNDOFF), default=0),
             'fixed_guards_independently_recomputed':True,'guards':guards,'deltas_pp':remote['deltas_pp'],
             'teacher_weights_and_logits_equal_pairs':9,'gpu_export_run':True,'new_training_runs':0,
             'limitations':['Base validation only, no new Novel/HM scores','Screening rules are not significance or publication criteria','No independent confirmation dataset in this batch']}
    write(WORK/'local-audit.json',audit)
    write(WORK/'cross-machine-roundoff.json',{'allowed_fields':['nll','multiclass_brier','probability_top1_top2_margin'],'absolute_tolerance':1e-12,'differences':ROUNDOFF})
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
