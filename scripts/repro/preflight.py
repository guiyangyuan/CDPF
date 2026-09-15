#!/usr/bin/env python3
"""Verify fixed dataset splits, description coverage, and independent few-shot seeds."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'Dassl.pytorch')]
from dassl.config import get_cfg_default
from dassl.utils import set_random_seed
from datasets.ucf101 import UCF101
from datasets.dtd import DescribableTextures
from datasets.fgvc_aircraft import FGVCAircraft
from datasets.caltech101 import Caltech101
from datasets.oxford_flowers import OxfordFlowers
from datasets.eurosat import EuroSAT
from train import extend_cfg
from trainers.textconsistency_common import TEMPLATE_FILES, description_rows, sha256_file
from trainers.textconsistency_runtime import sample_digest

DATASETS = {
    'ucf101': {'name': 'UCF101', 'loader': UCF101, 'dir': 'ucf101', 'image_dir': 'UCF-101-midframes', 'split': 'split_zhou_UCF101.json'},
    'dtd': {'name': 'DescribableTextures', 'loader': DescribableTextures, 'dir': 'dtd', 'image_dir': 'images', 'split': 'split_zhou_DescribableTextures.json'},
    'fgvc_aircraft': {'name': 'FGVCAircraft', 'loader': FGVCAircraft, 'dir': 'fgvc_aircraft', 'image_dir': 'images'},
    'caltech101': {'name': 'Caltech101', 'loader': Caltech101, 'dir': 'caltech-101',
                   'image_dir': '101_ObjectCategories', 'split': 'split_zhou_Caltech101.json'},
    'oxford_flowers': {'name': 'OxfordFlowers', 'loader': OxfordFlowers, 'dir': 'oxford_flowers',
                       'image_dir': 'jpg', 'split': 'split_zhou_OxfordFlowers.json'},
    'eurosat': {'name': 'EuroSAT', 'loader': EuroSAT, 'dir': 'eurosat',
                'image_dir': '2750', 'split': 'split_zhou_EuroSAT.json'},
}


def _validate_base_shots(train_x, classnames, shots=16):
    counts = Counter(item.label for item in train_x)
    expected_labels = set(range(len(classnames)))
    if set(counts) != expected_labels or any(counts[label] != shots for label in expected_labels):
        raise ValueError('The base training split is not exactly 16-shot per class')


def _fixed_split(data_root, key, spec):
    directory = data_root / spec['dir']
    if key != 'fgvc_aircraft':
        source = directory / spec['split']
        if not source.is_file():
            raise FileNotFoundError(f'Required fixed Zhou split is missing: {source}')
        split = json.loads(source.read_text())
        rows = {name: [(row[0], row[2]) for row in split[name]] for name in ('train', 'val', 'test')}
        files = [source]
    else:
        files = [directory / 'variants.txt'] + [directory / f'images_variant_{name}.txt' for name in ('train', 'val', 'test')]
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f'Required official Aircraft metadata is missing: {missing[0]}')
        rows = {}
        for name, path in zip(('train', 'val', 'test'), files[1:]):
            rows[name] = [(line.strip().split(' ', 1)[0] + '.jpg', line.strip().split(' ', 1)[1])
                          for line in path.read_text().splitlines()]
    image_dir = directory / spec['image_dir']
    missing_images = [path for values in rows.values() for path, _ in values if not (image_dir / path).is_file()]
    if missing_images:
        raise FileNotFoundError(f'{len(missing_images)} split images are missing; first: {missing_images[0]}')
    paths = {name: {path for path, _ in values} for name, values in rows.items()}
    if any(paths[a] & paths[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('Image paths overlap between train/validation/test splits')
    digest = sha256_file(files[0]) if len(files) == 1 else {path.name: sha256_file(path) for path in files}
    return rows, digest


def check(data_root, dataset='ucf101'):
    if dataset not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset}')
    data_root = Path(data_root).resolve(); spec = DATASETS[dataset]
    rows, split_digest = _fixed_split(data_root, dataset, spec)
    descriptions = json.loads((ROOT / TEMPLATE_FILES[spec['name']]).read_text())
    results = []
    for seed in (1, 2, 3):
        cfg = get_cfg_default(); extend_cfg(cfg)
        cfg.SEED = seed; cfg.DATASET.ROOT = str(data_root); cfg.DATASET.NUM_SHOTS = 16
        cfg.DATASET.SUBSAMPLE_CLASSES = 'base'; cfg.DATASET.NAME = spec['name']
        set_random_seed(seed); base = spec['loader'](cfg)
        base_names = set(base.classnames); description_rows(descriptions, base.classnames, 20)
        _validate_base_shots(base.train_x, base.classnames)
        cfg.DATASET.SUBSAMPLE_CLASSES = 'new'; novel = spec['loader'](cfg)
        description_rows(descriptions, novel.classnames, 20)
        if base_names & set(novel.classnames):
            raise ValueError('Base and novel class names overlap')
        results.append({'seed': seed, 'base_classes': len(base_names), 'novel_classes': len(novel.classnames),
                        'train_samples': len(base.train_x), 'train_sha256': sample_digest(base.train_x, data_root)})
    if len({r['train_sha256'] for r in results}) != 3:
        raise ValueError('Different seeds produced the same training split; inspect cached few-shot files')
    return {'dataset': spec['name'], 'split_sha256': split_digest,
            'image_counts': {name: len(values) for name, values in rows.items()},
            'all_images_present': True, 'splits_disjoint': True, 'seeds': results}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--dataset', choices=list(DATASETS), default='ucf101'); parser.add_argument('--output', type=Path)
    args = parser.parse_args(); result = check(args.data_root, args.dataset)
    text = json.dumps(result, indent=2); print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text)

if __name__ == '__main__': main()
