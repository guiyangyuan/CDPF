#!/usr/bin/env python3
"""Reject missing or mislabelled runs; aggregate matched checkpoint evaluations."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


def read_metric(path, seed, split, branch, epochs):
    data = json.loads(path.read_text())
    if (data['seed'], data['subsample'], data['branch'], data['dataset']) != (seed, split, branch, 'UCF101'):
        raise ValueError(f'Actual evaluation protocol differs from its directory label: {path}')
    if data.get('shots') != 16:
        raise ValueError(f'Expected the fixed 16-shot protocol: {path}')
    if data['state'] != 'loaded_checkpoint' or len(data['checkpoints']) != 1:
        raise ValueError(f'Evaluation did not load exactly one complete checkpoint: {path}')
    if data['checkpoints'][0]['epoch'] != epochs:
        raise ValueError(f'Wrong checkpoint epoch: {path}')
    accuracy = float(data['metrics']['accuracy'])
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 100:
        raise ValueError(f'Invalid accuracy in {path}')
    return data


def collect(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    rows = []
    for variant in protocol['variants']:
        branches = ['student'] if variant == 'baseline' else ['student', 'teacher', 'ensemble']
        for branch in branches:
            bases, novels, hms, train_hashes = [], [], [], []
            for seed in protocol['seeds']:
                directory = root / variant / f'seed{seed}'
                base = read_metric(directory / f'eval_base_{branch}/metrics.json', seed, 'base', branch, protocol['epochs'])
                novel = read_metric(directory / f'eval_new_{branch}/metrics.json', seed, 'new', branch, protocol['epochs'])
                if base['checkpoints'] != novel['checkpoints']:
                    raise ValueError('Base and novel results came from different checkpoints')
                if base['model_signature'] != novel['model_signature'] or base['code_sha256'] != novel['code_sha256']:
                    raise ValueError('Base/novel model configuration or source mismatch')
                b, n = float(base['metrics']['accuracy']), float(novel['metrics']['accuracy'])
                bases.append(b); novels.append(n); hms.append(2*b*n/(b+n) if b+n else 0.)
                train_hashes.append(base['train_samples_sha256'])
            if len(set(train_hashes)) != len(train_hashes):
                raise ValueError('Different seed labels reused the same few-shot training split')
            b, n = mean(bases), mean(novels)
            rows.append({'variant': variant, 'branch': branch, 'seeds': len(bases),
                         'base_mean': b, 'novel_mean': n, 'hm_of_means': 2*b*n/(b+n) if b+n else 0.,
                         'base_std': stdev(bases) if len(bases)>1 else None,
                         'novel_std': stdev(novels) if len(novels)>1 else None,
                         'mean_seed_hm': mean(hms), 'seed_hm_std': stdev(hms) if len(hms)>1 else None})
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); rows = collect(args.root)
    with (args.root / 'results.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.root / 'results.json').write_text(json.dumps(rows, indent=2))
    print(f'Validated {len(rows)} variant/branch rows. Single-seed rows are pilot results only.')

if __name__ == '__main__': main()
