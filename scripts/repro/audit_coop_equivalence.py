#!/usr/bin/env python3
"""Compare frozen CLIP + random-context CoOp on two real training images.

This checks logits, prompt gradients and one update, not benchmark accuracy.
Official/current implementations run in separate processes to isolate imports.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = 'ff61507c790454bce7c5052c3ac39e60772f1f89'
CLIP_SHA256 = '5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f'


def worker(args):
    source = args.upstream_root if args.worker == 'official' else ROOT
    sys.path[:0] = [str(source), str(ROOT / 'Dassl.pytorch')]
    import torch
    from PIL import Image
    from clip import clip
    from dassl.config import get_cfg_default
    from train import extend_cfg
    torch.set_num_threads(4)
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.MODEL.BACKBONE.NAME = 'ViT-B/16'
    weight = ROOT / 'clip/ViT-B-16.pt'
    with weight.open('rb') as handle:
        if hashlib.file_digest(handle, 'sha256').hexdigest() != CLIP_SHA256:
            raise ValueError('Unexpected CLIP checkpoint')
    if args.worker == 'official':
        from trainers.coop import CustomCLIP
        state = torch.jit.load(str(weight), map_location='cpu').state_dict()
        backbone = clip.build_model(state).float()
        cfg.TRAINER.COOP.N_CTX = 4
        cfg.TRAINER.COOP.CTX_INIT = ''
        cfg.TRAINER.COOP.CSC = False
        cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = 'end'
        cfg.TRAINER.COOP.PREC = 'fp32'
    else:
        from trainers.coop_textconsistency_ablation import CustomCLIP
        from trainers.textconsistency_common import load_clip_checkpoint
        cfg.TRAINER.TEXTCONSISTENCY.ABLATION_GROUP = 'baseline'
        cfg.TRAINER.TEXTCONSISTENCY.N_CTX = 4
        backbone = load_clip_checkpoint(cfg)
    rows = json.loads((args.data_root / 'ucf101/split_zhou_UCF101.json').read_text())['train']
    selected = {}
    for row in rows:
        selected.setdefault(row[2], row)
        if len(selected) == 2:
            break
    torch.manual_seed(1)  # Fix prompt initialization independently of backbone construction.
    model = CustomCLIP(cfg, list(selected), backbone).eval()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == 'prompt_learner.ctx')
    images = []
    for row in selected.values():
        with Image.open(args.data_root / 'ucf101/UCF-101-midframes' / row[0]) as image:
            images.append(clip._transform(224)(image))
    inputs = torch.stack(images)
    initial = model.prompt_learner.ctx.detach().clone()
    logits = model(inputs)
    if args.worker == 'current':
        logits = logits[0]
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    gradient = model.prompt_learner.ctx.grad.detach().clone()
    frozen = all(p.grad is None and not p.requires_grad for n, p in model.named_parameters()
                 if n != 'prompt_learner.ctx')
    optimizer = torch.optim.SGD([model.prompt_learner.ctx], lr=.002, momentum=.9, weight_decay=.0005)
    optimizer.step()
    torch.save({'initial_ctx': initial, 'logits': logits.detach(), 'gradient': gradient,
                'updated_ctx': model.prompt_learner.ctx.detach(), 'loss': loss.detach(),
                'frozen_parameters_without_grad': frozen,
                'images': [row[0] for row in selected.values()]}, args.output / (args.worker + '.pt'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=['official', 'current'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.upstream_root = args.upstream_root.resolve()
    args.data_root = args.data_root.resolve()
    args.output = args.output.resolve()
    if args.worker:
        worker(args)
        return
    commit = subprocess.check_output(['git', '-C', str(args.upstream_root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError('Official source does not match pinned audit commit')
    if subprocess.check_output(['git', '-C', str(args.upstream_root), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Official source must be unmodified')
    args.output.mkdir(parents=True, exist_ok=False)
    for side in ('official', 'current'):
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', CUDA_VISIBLE_DEVICES='')
        command = [sys.executable, str(Path(__file__).resolve()), '--upstream-root', str(args.upstream_root),
                   '--data-root', str(args.data_root), '--output', str(args.output), '--worker', side]
        with (args.output / (side + '.log')).open('w') as log:
            subprocess.run(command, check=True, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    import torch
    official = torch.load(args.output / 'official.pt', weights_only=True)
    current = torch.load(args.output / 'current.pt', weights_only=True)
    if official['images'] != current['images'] or not all(x['frozen_parameters_without_grad'] for x in (official, current)):
        raise AssertionError('Input or freezing mismatch')
    differences = {}
    for field in ('initial_ctx', 'logits', 'gradient', 'updated_ctx', 'loss'):
        torch.testing.assert_close(current[field], official[field], atol=1e-5, rtol=1e-4)
        differences[field] = float((current[field] - official[field]).abs().max())
    result = {'passed': True, 'benchmark_result': False, 'upstream_commit': commit,
              'shared_clip_sha256': CLIP_SHA256, 'device': 'cpu', 'images': official['images'],
              'scope': '2 training images, random 4-token initialization, FP32, one SGD update; not a full training reproduction',
              'frozen_parameters_without_grad': True, 'max_absolute_differences': differences}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
