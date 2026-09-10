#!/usr/bin/env python3
"""Verify actual UCF101 loader splits, text coverage, and independent seeds."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'Dassl.pytorch')]
from dassl.config import get_cfg_default
from dassl.utils import set_random_seed
from datasets.ucf101 import UCF101
from train import extend_cfg
from trainers.textconsistency_common import description_rows, sha256_file
from trainers.textconsistency_runtime import sample_digest


def check(data_root):
    data_root = Path(data_root).resolve()
    source = data_root / 'ucf101/split_zhou_UCF101.json'
    split = json.loads(source.read_text())
    missing = [row[0] for rows in split.values() for row in rows
               if not (data_root / 'ucf101/UCF-101-midframes' / row[0]).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} split images are missing; first: {missing[0]}')
    paths = {name: {row[0] for row in rows} for name, rows in split.items()}
    if any(paths[a] & paths[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('Image paths overlap between train/validation/test splits')
    descriptions = json.loads((ROOT / 'templates/ucf_prompts_full.json').read_text())
    results = []
    for seed in (1, 2, 3):
        cfg = get_cfg_default(); extend_cfg(cfg)
        cfg.SEED = seed; cfg.DATASET.ROOT = str(data_root); cfg.DATASET.NUM_SHOTS = 16
        cfg.DATASET.SUBSAMPLE_CLASSES = 'base'; cfg.DATASET.NAME = 'UCF101'
        set_random_seed(seed)
        base = UCF101(cfg)
        base_names = set(base.classnames)
        description_rows(descriptions, base.classnames, 20)
        cfg.DATASET.SUBSAMPLE_CLASSES = 'new'
        novel = UCF101(cfg)
        description_rows(descriptions, novel.classnames, 20)
        if base_names & set(novel.classnames):
            raise ValueError('Base and novel class names overlap')
        results.append({'seed': seed, 'base_classes': len(base_names), 'novel_classes': len(novel.classnames),
                        'train_samples': len(base.train_x),
                        'train_sha256': sample_digest(base.train_x, data_root)})
    if len({r['train_sha256'] for r in results}) != 3:
        raise ValueError('Different seeds produced the same training split; inspect cached few-shot files')
    return {'dataset': 'UCF101', 'split_sha256': sha256_file(source),
            'image_counts': {name: len(rows) for name, rows in split.items()},
            'all_images_present': True, 'splits_disjoint': True, 'seeds': results}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); result = check(args.data_root)
    text = json.dumps(result, indent=2); print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text)

if __name__ == '__main__': main()
