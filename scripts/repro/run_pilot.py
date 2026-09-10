#!/usr/bin/env python3
"""Matched aggregation comparisons; commands are reviewable with --plan-only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    'baseline': ('none', False), 'mean': ('mean', False), 'flat': ('flat', False),
    'flat_adapter': ('flat', True), 'hierarchical': ('hierarchical', False),
    'hierarchical_adapter': ('hierarchical', True),
}


def build_commands(data_root, output_root, seeds, epochs, variants, device='cuda', python=sys.executable):
    commands = []
    for variant in variants:
        if variant not in VARIANTS: raise ValueError(f'Unknown variant: {variant}')
        mode, adapter = VARIANTS[variant]
        baseline = variant == 'baseline'
        trainer = 'CoOp_TextConsistency_Ablation' if baseline else 'CoOp_TextConsistency_V2_FG'
        for seed in seeds:
            train_dir = Path(output_root) / variant / f'seed{seed}'
            common = [python, str(ROOT / 'train.py'), '--root', str(Path(data_root).resolve()),
                      '--seed', str(seed), '--trainer', trainer,
                      '--dataset-config-file', str(ROOT / 'configs/datasets/ucf101.yaml'),
                      '--config-file', str(ROOT / 'configs/repro/ucf101.yaml')]
            opts = ['DATASET.NUM_SHOTS', '16', 'TRAINER.TEXTCONSISTENCY.DEVICE', device,
                    'TRAINER.TEXTCONSISTENCY.AGGREGATION', 'mean' if baseline else mode,
                    'TRAINER.TEXTCONSISTENCY.USE_FEATURE_ADAPTER', str(adapter),
                    'TRAINER.TEXTCONSISTENCY.ABLATION_GROUP', 'baseline' if baseline else 'full',
                    'TRAINER.TEXTCONSISTENCY.INFERENCE_BRANCH', 'student' if baseline else 'ensemble',
                    'OPTIM.MAX_EPOCH', str(epochs)]
            commands.append(common + ['--output-dir', str(train_dir)] + opts + ['DATASET.SUBSAMPLE_CLASSES', 'base'])
            for split in ('base', 'new'):
                for branch in (('student',) if baseline else ('student', 'teacher', 'ensemble')):
                    eval_opts = opts.copy()
                    eval_opts[eval_opts.index('TRAINER.TEXTCONSISTENCY.INFERENCE_BRANCH') + 1] = branch
                    commands.append(common + ['--output-dir', str(train_dir / f'eval_{split}_{branch}'),
                                              '--model-dir', str(train_dir), '--load-epoch', str(epochs), '--eval-only']
                                    + eval_opts + ['DATASET.SUBSAMPLE_CLASSES', split])
    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error('Epochs must be positive and seeds must be distinct nonnegative integers')
    output = args.output_root.resolve()
    commands = build_commands(args.data_root, output, args.seeds, args.epochs, args.variants, args.device)
    plan = {'purpose': 'controlled_pilot', 'dataset': 'UCF101', 'seeds': args.seeds, 'epochs': args.epochs,
            'variants': args.variants, 'primary_branch': 'ensemble (baseline: student)',
            'checkpoint_rule': 'specified final epoch; no test-based selection', 'commands': commands}
    if args.plan_only:
        print(json.dumps(plan, indent=2)); return
    if output.exists(): raise FileExistsError(f'Choose a fresh output root: {output}')
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use a GPU server for the full pilot')
    from preflight import check
    preflight = check(args.data_root)
    output.mkdir(parents=True)
    (output / 'protocol.json').write_text(json.dumps(plan, indent=2))
    (output / 'data_preflight.json').write_text(json.dumps(preflight, indent=2))
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / 'Dassl.pytorch'), env.get('PYTHONPATH', '')])
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    for index, command in enumerate(commands):
        print(f'Running {index+1}/{len(commands)}: {command}', flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/repro/summarize.py'), '--root', str(output)], check=True)

if __name__ == '__main__': main()
