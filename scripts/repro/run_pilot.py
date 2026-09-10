#!/usr/bin/env python3
"""Matched aggregation comparisons; commands are reviewable with --plan-only."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.repro.structure_spec import EXTRA_VARIANTS, STRUCTURE_VARIANTS, order_settings
VARIANTS = {
    'baseline': ('none', False), 'mean': ('mean', False), 'flat': ('flat', False),
    'flat_adapter': ('flat', True), 'hierarchical': ('hierarchical', False),
    'hierarchical_adapter': ('hierarchical', True),
}
DEFAULT_VARIANTS = tuple(VARIANTS)
VARIANTS.update(EXTRA_VARIANTS)
DATASETS = {
    'ucf101': {'name': 'UCF101'},
    'dtd': {'name': 'DescribableTextures'},
    'fgvc_aircraft': {'name': 'FGVCAircraft'},
}


def build_commands(data_root, output_root, seeds, epochs, variants, device='cuda', python=sys.executable,
                   dataset='ucf101', kd_weight=1.0):
    if dataset not in DATASETS:
        raise ValueError(f'Unknown dataset: {dataset}')
    if not math.isfinite(kd_weight) or kd_weight < 0:
        raise ValueError('KD weight must be a finite nonnegative number')
    commands = []
    for variant in variants:
        if variant not in VARIANTS: raise ValueError(f'Unknown variant: {variant}')
        mode, adapter = VARIANTS[variant]
        order = order_settings(variant)
        baseline = variant == 'baseline'
        trainer = 'CoOp_TextConsistency_Ablation' if baseline else 'CoOp_TextConsistency_V2_FG'
        for seed in seeds:
            train_dir = Path(output_root) / variant / f'seed{seed}'
            common = [python, str(ROOT / 'train.py'), '--root', str(Path(data_root).resolve()),
                      '--seed', str(seed), '--trainer', trainer,
                      '--dataset-config-file', str(ROOT / f'configs/datasets/{dataset}.yaml'),
                      '--config-file', str(ROOT / f'configs/repro/{dataset}.yaml')]
            opts = ['DATASET.NUM_SHOTS', '16', 'TRAINER.TEXTCONSISTENCY.DEVICE', device,
                    'TRAINER.TEXTCONSISTENCY.AGGREGATION', 'mean' if baseline else mode,
                    'TRAINER.TEXTCONSISTENCY.USE_FEATURE_ADAPTER', str(adapter),
                    'TRAINER.TEXTCONSISTENCY.VIEW_ORDER', order['policy'],
                    'TRAINER.TEXTCONSISTENCY.ORDER_SEED', str(order['seed']),
                    'TRAINER.TEXTCONSISTENCY.ABLATION_GROUP', 'baseline' if baseline else 'full',
                    'TRAINER.TEXTCONSISTENCY.INFERENCE_BRANCH', 'student' if baseline else 'ensemble',
                    'TRAINER.TEXTCONSISTENCY.KD_WEIGHT', str(0.0 if baseline else kd_weight),
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
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--variants', nargs='+', choices=list(VARIANTS))
    selection.add_argument('--structure-study', action='store_true',
                           help='Fixed UCF101 capacity/order controls; seeds 1 2 3, 20 epochs, KD=1')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--dataset', choices=list(DATASETS), default='ucf101')
    parser.add_argument('--kd-weight', type=float, default=1.0)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.structure_study and (args.seeds != [1, 2, 3] or args.epochs != 20 or
                                args.kd_weight != 1.0 or args.dataset != 'ucf101' or args.device != 'cuda'):
        parser.error('Structure study requires UCF101, CUDA, --seeds 1 2 3, 20 epochs and KD=1')
    args.variants = list(STRUCTURE_VARIANTS) if args.structure_study else (args.variants or list(DEFAULT_VARIANTS))
    if len(set(args.variants)) != len(args.variants):
        parser.error('Variants must be distinct')
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error('Epochs must be positive and seeds must be distinct nonnegative integers')
    output = args.output_root.resolve()
    if not math.isfinite(args.kd_weight) or args.kd_weight < 0:
        parser.error('KD weight must be a finite nonnegative number')
    commands = build_commands(args.data_root, output, args.seeds, args.epochs, args.variants, args.device,
                              dataset=args.dataset, kd_weight=args.kd_weight)
    plan = {'purpose': 'controlled_pilot', 'dataset': DATASETS[args.dataset]['name'],
            'dataset_key': args.dataset, 'seeds': args.seeds, 'epochs': args.epochs,
            'variants': args.variants, 'primary_branch': 'ensemble (baseline: student)',
            'kd_weight': args.kd_weight, 'shots': 16,
            'view_orders': {variant: order_settings(variant) for variant in args.variants},
            'structure_study': args.structure_study,
            'fixed_settings': {'n_ctx': 4, 'n_views': 20, 'n_groups': 4, 'kd_temp': 2.0,
                               'adaptive_ce_weight': 1.0, 'contrastive_weight': 0.0,
                               'diversity_weight': 0.0, 'ensemble_alpha': 0.5},
            'checkpoint_rule': 'specified final epoch; no test-based selection', 'commands': commands}
    if args.plan_only:
        print(json.dumps(plan, indent=2)); return
    if output.exists(): raise FileExistsError(f'Choose a fresh output root: {output}')
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use a GPU server for the full pilot')
    from preflight import check
    preflight = check(args.data_root, args.dataset)
    from trainers.textconsistency_runtime import source_digest
    plan['code_sha256'] = source_digest()
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
