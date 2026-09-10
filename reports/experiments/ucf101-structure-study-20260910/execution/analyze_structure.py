"""Audit all fixed structure outputs before reporting any experimental conclusion."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, stdev
import sys


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    source, study = args.source.resolve(), args.study.resolve()
    sys.path[:0] = [str(source), str(source/'Dassl.pytorch')]
    import torch
    from scripts.repro.summarize import collect
    from scripts.repro.structure_spec import STRUCTURE_VARIANTS
    state = json.loads((study/'status.json').read_text())
    assert state['state'] == 'completed' and state['completed'] == ['smoke', 'formal']
    summary = collect(study/'formal')
    assert len(summary) == 15
    rows, verified, log_audits, replay = [], [], [], []
    history = json.loads((source/'reports/experiments/ucf101-kd-study-20260910/analysis.json').read_text())
    old = {(r['seed'], r['branch']): r for r in history['kd_per_seed']
           if r['kd'] == 1 and r['variant'] == 'hierarchical_adapter'}
    for variant in STRUCTURE_VARIANTS:
        for seed in (1, 2, 3):
            folder = study/'formal'/variant/f'seed{seed}'
            manifest = json.loads((folder/'run_manifest.json').read_text())
            checkpoint = folder/'textconsistency/model.pth.tar-20'
            digest = sha(checkpoint)
            assert checkpoint.with_name(checkpoint.name+'.sha256').read_text().strip() == digest
            # These are our own trusted experiment checkpoints, including scheduler state.
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert saved['epoch'] == 20 and saved['run_provenance'] == manifest
            assert saved['repro_signature'] == manifest['model_signature']
            parameters = {k: v for k, v in saved['state_dict'].items()
                          if not k.endswith(('token_prefix', 'token_suffix'))}
            assert sum(value.numel() for value in parameters.values()) == 790018
            assert all(torch.isfinite(value).all() for value in parameters.values())
            assert manifest['trainable_parameters'] == 790018
            verified.append({'variant': variant, 'seed': seed, 'sha256': digest,
                             'parameter_count_from_checkpoint': 790018, 'all_tensors_finite': True})
            for branch in ('student', 'teacher', 'ensemble'):
                scores = {}
                for split in ('base', 'new'):
                    directory = folder/f'eval_{split}_{branch}'
                    metric = json.loads((directory/'metrics.json').read_text())
                    assert metric['checkpoints'][0]['sha256'] == digest
                    scores[split] = float(metric['metrics']['accuracy'])
                    log = (directory/'log.txt').read_text().replace(',', '')
                    total = re.findall(r'\* total:\s*(\d+)', log)
                    correct = re.findall(r'\* correct:\s*(\d+)', log)
                    assert total and correct, f'Missing independent evaluator counts: {directory}'
                    expected = int(correct[-1])/int(total[-1])*100
                    assert abs(expected-scores[split]) < 1e-10
                    log_audits.append({'variant': variant, 'seed': seed, 'branch': branch, 'split': split,
                                       'correct': int(correct[-1]), 'total': int(total[-1]), 'accuracy': expected})
                base, novel = scores['base'], scores['new']
                row = {'variant': variant, 'seed': seed, 'branch': branch, 'base': base, 'novel': novel,
                       'hm': 2*base*novel/(base+novel) if base+novel else 0.}
                rows.append(row)
                if variant == 'hierarchical_adapter':
                    replay.append({'seed': seed, 'branch': branch,
                                   'base_difference': base-old[seed, branch]['base'],
                                   'novel_difference': novel-old[seed, branch]['novel'],
                                   'hm_difference': row['hm']-old[seed, branch]['hm']})
    paired = []
    for variant in STRUCTURE_VARIANTS[1:]:
        for branch in ('student', 'teacher', 'ensemble'):
            deltas = []
            for seed in (1, 2, 3):
                items = {r['variant']: r for r in rows if r['seed'] == seed and r['branch'] == branch}
                deltas.append({'seed': seed, **{key: items[variant][key]-items['hierarchical_adapter'][key]
                                               for key in ('base', 'novel', 'hm')}})
            paired.append({'variant': variant, 'reference': 'hierarchical_adapter', 'branch': branch,
                           'direction': 'control minus original hierarchical model', 'per_seed': deltas,
                           'mean_delta_hm': mean(r['hm'] for r in deltas),
                           'std_delta_hm': stdev(r['hm'] for r in deltas),
                           'positive_seeds': sum(r['hm']>0 for r in deltas)})
    samples = []
    for line in (study/'gpu-samples.jsonl').read_text().splitlines():
        item = json.loads(line)
        if item['returncode'] == 0:
            parts = item['sample'].split(',')
            try: samples.append(float(parts[0].strip()))
            except ValueError: pass
    result = {'passed': True, 'scope': 'UCF101 fixed 16-shot, 20 epochs, all 5 structures and 3 seeds',
              'summary': summary, 'per_seed': rows, 'paired_vs_original': paired,
              'original_vs_previous_results': replay,
              'original_exactly_reproduced': all(r[k] == 0 for r in replay for k in ('base_difference', 'novel_difference')),
              'verified_checkpoints': verified, 'independent_eval_count_checks': log_audits,
              'smoke_audit': json.loads((study/'smoke-audit.json').read_text()),
              'formal_seconds': state['runs']['formal']['seconds'], 'smoke_seconds': state['runs']['smoke']['seconds'],
              'observed_device_peak_MiB': max(samples) if samples else None,
              'memory_measurement_note': '5-second whole-device sampling includes persistent queue CUDA context; not per-model peak allocation or a speed benchmark.',
              'limits': ['One dataset and three training seeds cannot establish broad novelty or publication readiness.',
                         'Two fixed random grouping permutations are both reported, without selecting the better one.',
                         'Equal parameter count does not imply equal FLOPs or identical optimization dynamics.',
                         'These are follow-up diagnostic controls after observing prior UCF101 results.',
                         'DTD/Aircraft and closest-method comparisons remain unrun.']}
    assert len(rows) == 45 and len(verified) == 15 and len(log_audits) == 90
    (study/'analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'passed': True, 'main': [r for r in summary if r['branch']=='ensemble'],
                      'paired': [r for r in paired if r['branch']=='ensemble'],
                      'original_exactly_reproduced': result['original_exactly_reproduced']}, indent=2))


if __name__ == '__main__':
    main()
