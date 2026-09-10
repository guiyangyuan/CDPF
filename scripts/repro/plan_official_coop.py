#!/usr/bin/env python3
"""Print pinned upstream CoOp training/evaluation commands; never executes them."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PIN = 'ff61507c790454bce7c5052c3ac39e60772f1f89'


def plan(upstream, data, output, recipe, seeds, python):
    if output.exists():
        raise FileExistsError('Use a new output directory')
    commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != PIN or subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Expected an unmodified, pinned official CoOp checkout')
    source_manifest = json.loads((ROOT / 'reports/preparation/official-coop-source.json').read_text())
    for name, digest in source_manifest['files'].items():
        if hashlib.sha256((upstream / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Official source checksum mismatch: {name}')
    epochs = 20 if recipe == 'matched20' else 200
    cfg = 'vit_b16.yaml' if recipe == 'matched20' else 'vit_b16_ctxv1.yaml'
    commands = []
    for seed in seeds:
        train = output / f'seed{seed}'
        common = [python, str(upstream / 'train.py'), '--root', str(data), '--seed', str(seed),
                  '--trainer', 'CoOp', '--dataset-config-file', str(upstream / 'configs/datasets/ucf101.yaml'),
                  '--config-file', str(upstream / 'configs/trainers/CoOp' / cfg)]
        opts = ['DATASET.NUM_SHOTS', '16', 'TRAINER.COOP.N_CTX', '4', 'TRAINER.COOP.CSC', 'False',
                'TRAINER.COOP.CLASS_TOKEN_POSITION', 'end', 'TRAINER.COOP.PREC', 'fp32',
                'OPTIM.MAX_EPOCH', str(epochs), 'TEST.FINAL_MODEL', 'last_step']
        if recipe == 'matched20':
            opts += ['TRAINER.COOP.CTX_INIT', '', 'DATALOADER.TRAIN_X.BATCH_SIZE', '16',
                     'DATALOADER.TEST.BATCH_SIZE', '16', 'DATALOADER.NUM_WORKERS', '4']
        commands.append(common + ['--output-dir', str(train)] + opts + ['DATASET.SUBSAMPLE_CLASSES', 'base'])
        for split in ('base', 'new'):
            commands.append(common + ['--output-dir', str(train / ('eval_' + split)), '--model-dir', str(train),
                                     '--load-epoch', str(epochs), '--eval-only'] + opts + ['DATASET.SUBSAMPLE_CLASSES', split])
    return {'purpose': 'official_coop_preparation_only', 'executed': False, 'recipe': recipe,
            'upstream_commit': commit, 'dataset': 'UCF101', 'seeds': seeds, 'epochs': epochs,
            'checkpoint_rule': 'specified final epoch, identical checkpoint for Base and Novel',
            'cwd': str(upstream), 'env': {'PYTHONPATH': str(ROOT / 'Dassl.pytorch'),
                                        'CUBLAS_WORKSPACE_CONFIG': ':4096:8'},
            'requirements_before_execution': [
                'User has enabled GPU; verify torch.cuda.is_available() before any command.',
                'Run data preflight; record few-shot sample, source, dependency, CLIP and checkpoint hashes.',
                'Place the verified CLIP weight at ~/.cache/clip/ViT-B-16.pt; do not replace mismatching files.',
                'Preserve logs, effective configs and loaded checkpoint evidence; upstream does not emit our TC manifests.',
                'FP32 and project-vendored Dassl are documented adaptations, not the original software environment.'],
            'commands': commands}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--recipe', choices=['matched20', 'reference200'], required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error('Seeds must be distinct nonnegative integers')
    print(json.dumps(plan(args.upstream_root.resolve(), args.data_root.resolve(), args.output_root.resolve(),
                          args.recipe, args.seeds, sys.executable), indent=2))


if __name__ == '__main__':
    main()
