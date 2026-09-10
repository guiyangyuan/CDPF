"""Audit complete outputs before computing paired KD effects; read only model files."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
import sys


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--study', type=Path, required=True)
    a = p.parse_args()
    source, study = a.source.resolve(), a.study.resolve()
    sys.path[:0] = [str(source), str(source / 'Dassl.pytorch')]
    import torch
    import yaml
    from scripts.repro.summarize import collect
    assert json.loads((study / 'status.json').read_text())['state'] == 'completed'
    summaries = {kd: collect(study / f'kd{kd}') for kd in (0, 1)}
    history = json.loads((source / 'reports/experiments/ucf101-s123-20260910/three-seed-analysis.json').read_text())
    historical = {(r['variant'], r['seed'], r['branch']): r for r in history['per_seed']}
    configs, rows, checkpoints, isolation, replay = [], [], [], [], []
    for variant in ('hierarchical', 'hierarchical_adapter'):
        for seed in (1, 2, 3):
            models = {}
            pair_configs, manifests = [], []
            for kd in (0, 1):
                folder = study / f'kd{kd}' / variant / f'seed{seed}'
                cfg = yaml.safe_load((folder / 'effective_config.yaml').read_text())
                assert cfg['TRAINER']['TEXTCONSISTENCY']['KD_WEIGHT'] == kd
                normalized = copy.deepcopy(cfg)
                normalized.pop('OUTPUT_DIR')
                normalized['TRAINER']['TEXTCONSISTENCY'].pop('KD_WEIGHT')
                pair_configs.append(normalized)
                manifest = json.loads((folder / 'run_manifest.json').read_text())
                manifests.append(manifest)
                cp = folder / 'textconsistency/model.pth.tar-20'
                digest = sha(cp)
                assert cp.with_name(cp.name+'.sha256').read_text().strip() == digest
                saved = torch.load(cp, map_location='cpu', weights_only=False)
                assert saved['epoch'] == 20 and saved['run_provenance'] == manifest
                models[kd] = saved['state_dict']
                checkpoints.append({'kd':kd,'variant':variant,'seed':seed,'sha256':digest,'path':str(cp)})
                for branch in ('student', 'teacher', 'ensemble'):
                    accuracies = {}
                    for split in ('base', 'new'):
                        metrics = json.loads((folder / f'eval_{split}_{branch}/metrics.json').read_text())
                        assert metrics['checkpoints'][0]['sha256'] == digest
                        accuracies[split] = float(metrics['metrics']['accuracy'])
                    b,n = accuracies['base'],accuracies['new']
                    row = {'kd':kd,'variant':variant,'seed':seed,'branch':branch,'base':b,'novel':n,'hm':2*b*n/(b+n)}
                    rows.append(row)
                    if kd == 1:
                        old = historical[variant,seed,branch]
                        replay.append({'variant':variant,'seed':seed,'branch':branch,
                                       'base_difference':b-old['base'],'novel_difference':n-old['novel']})
            assert pair_configs[0] == pair_configs[1], 'KD pair differs beyond weight/output'
            for key in ('code_sha256','train_samples_sha256','val_samples_sha256','test_samples_sha256','model_signature','trainable_parameters'):
                assert manifests[0][key] == manifests[1][key], (variant,seed,key)
            configs.append({'variant':variant,'seed':seed,'only_kd_and_output_changed':True,
                            'train_samples_sha256':manifests[0]['train_samples_sha256']})
            assert models[0].keys() == models[1].keys()
            distances = {}
            for prefix in ('prompt_learner.', 'view_aggregator.'):
                values = [float((models[1][key]-models[0][key]).abs().max()) for key in models[0] if key.startswith(prefix) and models[0][key].numel()]
                assert values
                distances[prefix] = max(values)
            isolation.append({'variant':variant,'seed':seed,'max_absolute_weight_differences':distances})
    paired = []
    for variant in ('hierarchical','hierarchical_adapter'):
        for branch in ('student','teacher','ensemble'):
            values = []
            for seed in (1,2,3):
                pair = {r['kd']:r for r in rows if (r['variant'],r['branch'],r['seed'])==(variant,branch,seed)}
                values.append({'seed':seed,**{key:pair[1][key]-pair[0][key] for key in ('base','novel','hm')}})
            paired.append({'variant':variant,'branch':branch,'per_seed_differences':values,
                           'mean_delta_hm':mean(x['hm'] for x in values),'std_delta_hm':stdev(x['hm'] for x in values),
                           'positive_seeds':sum(x['hm']>0 for x in values)})
    official = json.loads((study/'official_matched20/summary.json').read_text())
    assert len(official['results']) == 6 and [r['seed'] for r in official['seeds']]==[1,2,3]
    baseline_replay=[]
    for row in official['seeds']:
        old=historical['baseline',row['seed'],'student']
        baseline_replay.append({'seed':row['seed'],'base_difference':row['base']-old['base'],'novel_difference':row['new']-old['novel']})
    for seed in (1,2,3):
        cp=study/f'official_matched20/seed{seed}/prompt_learner/model.pth.tar-20'
        expected=json.loads((study/f'official_matched20/evidence/seed{seed}-checkpoint.json').read_text())['sha256']
        assert sha(cp)==expected
        assert all(r['checkpoint_sha256']==expected for r in official['results'] if r['seed']==seed)
    result={'passed':True,'scope':'UCF101 only; fixed 20 epochs, 16-shot, all 3 seeds; no test-based selection',
            'kd_summary':summaries,'kd_per_seed':rows,'paired_kd_effects':paired,'paired_protocol_audit':configs,
            'parameter_isolation':isolation,'verified_tc_checkpoints':checkpoints,'official_matched20':official,
            'kd1_vs_previous_results':replay,'official_vs_previous_baseline':baseline_replay,
            'limitations':['3 seeds on one dataset do not establish cross-dataset novelty or publication readiness.',
                           'The 200-epoch upstream reference has not been run in this batch.',
                           'Official entry keeps upstream cuDNN/RNG behavior; see provenance for environment adaptations.',
                           'Concurrent KD jobs make elapsed time unsuitable for method efficiency comparisons.']}
    (study/'analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':True,'official':official['means'],'official_hm':official['hm_of_means'],
                      'kd_summary':summaries,'paired':paired},indent=2))


if __name__ == '__main__':
    main()
