#!/usr/bin/env python3
"""Two-class CPU smoke using official ViT-B/16 and actual UCF101 images.
This is a correctness check, not a benchmark accuracy experiment.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'Dassl.pytorch')]
import torch
from PIL import Image
from clip.clip import _transform
from dassl.config import get_cfg_default
from dassl.data.datasets import Datum
from dassl.engine.trainer import TrainerBase
from dassl.utils import set_random_seed
from train import extend_cfg
from trainers.coop_textconsistency_v2_fg import CoOp_TextConsistency_V2_FG


def build(data_root, output):
    set_random_seed(1); torch.set_num_threads(4)
    split = json.loads((data_root / 'ucf101/split_zhou_UCF101.json').read_text())
    chosen = {}
    for row in split['train']:
        chosen.setdefault(row[2], row)
        if len(chosen) == 2: break
    rows = list(chosen.values())
    samples = [Datum(impath=str(data_root / 'ucf101/UCF-101-midframes' / row[0]),
                     label=i, classname=row[2]) for i, row in enumerate(rows)]
    cfg = get_cfg_default(); extend_cfg(cfg)
    cfg.SEED = 1; cfg.USE_CUDA = False
    cfg.DATASET.NAME = 'UCF101'; cfg.DATASET.ROOT = str(data_root)
    cfg.DATASET.SUBSAMPLE_CLASSES = 'base'; cfg.DATASET.NUM_SHOTS = 1
    cfg.TRAINER.NAME = 'CoOp_TextConsistency_V2_FG'; cfg.OUTPUT_DIR = str(output)
    cfg.MODEL.BACKBONE.NAME = 'ViT-B/16'; cfg.OPTIM.MAX_EPOCH = 1; cfg.OPTIM.WARMUP_EPOCH = 0
    tc = cfg.TRAINER.TEXTCONSISTENCY
    tc.DEVICE = 'cpu'; tc.N_VIEWS = 5; tc.N_GROUPS = 2; tc.TEXT_BATCH_SIZE = 4
    tc.KD_WEIGHT = 1.; tc.INFERENCE_BRANCH = 'ensemble'
    trainer = CoOp_TextConsistency_V2_FG.__new__(CoOp_TextConsistency_V2_FG)
    TrainerBase.__init__(trainer); trainer.cfg = cfg
    trainer.dm = SimpleNamespace(dataset=SimpleNamespace(classnames=[r[2] for r in rows],
                                                          train_x=samples, val=[], test=samples))
    trainer.check_cfg(cfg); trainer.build_model()
    trainer.batch_idx = 0; trainer.num_batches = 1
    images = []
    for sample in samples:
        with Image.open(sample.impath) as image: images.append(_transform(224)(image))
    return trainer, torch.stack(images), [s.impath for s in samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=['train', 'reload'], required=True)
    args = parser.parse_args(); output = args.output.resolve()
    if args.stage == 'train' and output.exists(): raise FileExistsError(output)
    trainer, images, image_paths = build(args.data_root.resolve(), output / args.stage)
    if args.stage == 'train':
        prompt = trainer.model.prompt_learner.ctx.detach().clone()
        adapter = trainer.model.view_aggregator.fine_grained_proj[0].weight.detach().clone()
        losses = trainer.forward_backward({'img': images, 'label': torch.tensor([0, 1])})
        prompt_changed = not torch.equal(prompt, trainer.model.prompt_learner.ctx)
        adapter_changed = not torch.equal(adapter, trainer.model.view_aggregator.fine_grained_proj[0].weight)
        frozen = all(p.grad is None and not p.requires_grad for p in trainer.model.image_encoder.parameters())
        if not (prompt_changed and adapter_changed and frozen): raise AssertionError('Gradient smoke failed')
        trainer.save_model(0, output / 'train'); trainer.model.eval()
        with torch.no_grad(): expected = trainer.model_inference(images)
        torch.save(expected, output / 'expected_logits.pt')
        result = {'smoke_only': True, 'pretrained_clip': 'official ViT-B/16', 'device': 'cpu',
                  'classes': 2, 'images': image_paths, 'optimizer_steps': 1,
                  'prompt_updated': prompt_changed, 'feature_adapter_updated': adapter_changed,
                  'image_encoder_frozen': frozen, 'losses': losses}
        (output / 'train_check.json').write_text(json.dumps(result, indent=2))
    else:
        trainer.load_model(output / 'train', 1); trainer.model.eval()
        expected = torch.load(output / 'expected_logits.pt', map_location='cpu', weights_only=True)
        with torch.no_grad(): actual = trainer.model_inference(images)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        result = {'smoke_only': True, 'fresh_process_reload': True,
                  'max_logit_difference': float((actual - expected).abs().max()),
                  'checkpoints': trainer.loaded_checkpoints}
        (output / 'reload_check.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
