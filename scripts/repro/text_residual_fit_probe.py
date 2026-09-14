#!/usr/bin/env python3
"""Bounded training-only optimizer diagnostic; never scores validation/test.

Cache 4/16 Base training images per class with the existing frozen student.
Reset both residuals to the same fresh initialization, then fit 200 CPU steps.
These are training-fit diagnostics, not benchmark results or a new method score.
"""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'Dassl.pytorch')]
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dassl.config import get_cfg_default
from dassl.data.data_manager import DatasetWrapper
from dassl.data.transforms import build_transform
from dassl.engine import build_trainer
from dassl.utils import set_random_seed
import train
from trainers.coop_text_residual import SharedTextResidual
from trainers.textconsistency_common import sha256_file
from trainers.textconsistency_runtime import source_digest, sample_digest
from scripts.repro.text_residual_pilot import SAMPLE_HASHES, MODEL_SOURCE_SHA

SETTINGS = {
    'sgd': {'lr': .002, 'momentum': .9, 'weight_decay': .0005},
    'adamw': {'lr': .001, 'betas': (.9, .999), 'eps': 1e-8, 'weight_decay': .0005},
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def schedule(n, steps=200):
    gen = torch.Generator().manual_seed(1729)
    batches = []
    while len(batches) < steps:
        batches.extend(torch.randperm(n, generator=gen).split(16))
    return batches[:steps]


def logits(module, cache, indices=None):
    features, student = cache['features'], cache['S']
    if indices is not None:
        features, student = features[indices], student[indices]
    delta = module(cache['views'], cache['mask'])
    residual = cache['scale'] * features @ delta.T
    return student + residual, residual


@torch.no_grad()
def measure(module, cache):
    z, r = logits(module, cache)
    labels, s = cache['labels'], cache['S']
    correct, original = z.argmax(1).eq(labels), s.argmax(1).eq(labels)
    return {'ce': F.cross_entropy(z, labels).item(), 'correct': int(correct.sum()),
            'accuracy_pct': 100 * correct.float().mean().item(),
            'corrected': int((correct & ~original).sum()), 'harmed': int((~correct & original).sum()),
            'prediction_changes': int((z.argmax(1) != s.argmax(1)).sum()),
            'residual_row_std_mean': r.std(dim=1, unbiased=False).mean().item()}


def fit(module, cache, name, batches, output=None):
    if any(t.requires_grad for t in cache.values() if isinstance(t, torch.Tensor)):
        raise ValueError('All student/features/text caches must be detached')
    optimizer_cls = torch.optim.SGD if name == 'sgd' else torch.optim.AdamW
    optimizer = optimizer_cls(module.parameters(), **SETTINGS[name])
    initial = copy.deepcopy(module.state_dict())
    trajectory = [{'step': 0, **measure(module, cache)}]
    for step, indices in enumerate(batches, 1):
        optimizer.zero_grad(set_to_none=True)
        z, _ = logits(module, cache, indices)
        loss = F.cross_entropy(z, cache['labels'][indices])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in module.parameters()):
            raise FloatingPointError('Missing or nonfinite residual gradient')
        optimizer.step()
        if step % 20 == 0 or step == len(batches):
            trajectory.append({'step': step, **measure(module, cache)})
    with torch.no_grad():
        z, r = logits(module, cache)
    result = {'optimizer': name, 'settings': SETTINGS[name], 'steps': len(batches),
              'trajectory': trajectory, 'initial': trajectory[0], 'final': trajectory[-1],
              'ce_reduction': trajectory[0]['ce'] - trajectory[-1]['ce'],
              'parameter_update_l2': {k: (v - initial[k]).norm().item() for k, v in module.state_dict().items()}}
    # Predeclared training-fit signal, not a generalization/publication gate.
    result['train_fit_signal'] = result['ce_reduction'] >= .10 and result['final']['correct'] >= result['initial']['correct']
    if output is not None:
        output.mkdir()
        torch.save({'state_dict': module.state_dict(), 'optimizer': optimizer.state_dict(),
                    'steps': len(batches), 'settings': SETTINGS[name]}, output / 'residual.pt')
        np.savez_compressed(output / 'train_logits.npz', Z=z.numpy(), R=r.numpy(),
                            S=cache['S'].numpy(), labels=cache['labels'].numpy())
        write_json(output / 'result.json', result)
    return result


def run(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    out, anchor, data = map(lambda p: Path(p).resolve(), (args.output_root, args.model_dir, args.data_root))
    if out.exists():
        raise FileExistsError('Use a fresh diagnostic output directory')
    if source_digest() != MODEL_SOURCE_SHA:
        raise ValueError('Unexpected model source')
    cp = anchor / 'textconsistency/model.pth.tar-20'
    expected = json.loads((ROOT / 'reports/experiments/text-residual-20260914/contrast/export.json').read_text())
    if sha256_file(cp) != expected['checkpoint'][0]['sha256']:
        raise ValueError('Expected the archived contrast epoch20 checkpoint')
    cfg = get_cfg_default(); train.extend_cfg(cfg)
    cfg.merge_from_file(str(anchor / 'effective_config.yaml'))
    if (cfg.TRAINER.NAME != 'CoOp_TextResidual' or cfg.TRAINER.TEXTCONSISTENCY.TEXT_RESIDUAL_MODE != 'contrast'
            or cfg.SEED != 1 or cfg.DATASET.NAME != 'FGVCAircraft' or cfg.DATASET.SUBSAMPLE_CLASSES != 'base'
            or cfg.DATASET.NUM_SHOTS != 16):
        raise ValueError('Only the fixed Aircraft training-only diagnostic is supported')
    cfg.DATASET.ROOT = str(data); cfg.OUTPUT_DIR = str(out / 'runtime')
    cfg.TRAINER.TEXTCONSISTENCY.CLIP_PATH = str(Path(args.clip_path).resolve())
    cfg.USE_CUDA = True; cfg.TRAINER.TEXTCONSISTENCY.DEVICE = 'cuda'; cfg.freeze()
    set_random_seed(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    trainer = build_trainer(cfg)
    # Retain a single *fresh* residual initialization, before loading trained weights.
    initial = copy.deepcopy(trainer.model.text_residual.cpu().state_dict())
    trainer.model.text_residual.to(trainer.device)
    trainer.load_model(anchor, epoch=20)
    trainer.model.eval()
    if sample_digest(trainer.dm.dataset.train_x, data) != SAMPLE_HASHES['train_samples_sha256']:
        raise ValueError('Training split differs from fixed 16-shot split')
    rng, items = random.Random(1), []
    for label in range(50):
        group = sorted((x for x in trainer.dm.dataset.train_x if x.label == label), key=lambda x: x.impath)
        if len(group) != 16:
            raise ValueError('Expected 16 training samples per Base class')
        items.extend(rng.sample(group, 4))
    transform = build_transform(cfg, is_train=False)
    loader = DataLoader(DatasetWrapper(cfg, items, transform=transform, is_train=False),
                        batch_size=16, shuffle=False, num_workers=0)
    feats, scores, labels, rows = [], [], [], []
    m = trainer.model
    with torch.no_grad():
        student_text = F.normalize(m.text_encoder(m.prompt_learner(), m.tokenized_prompts), dim=-1)
        scale = m.logit_scale.exp()
        for batch in loader:
            image = batch['img'].to(trainer.device)
            features = F.normalize(m.image_encoder(image.to(m.dtype)), dim=-1)
            s = scale * features @ student_text.T
            if not feats:
                torch.testing.assert_close(s, m(image)['S'], rtol=0, atol=0)
            feats.append(features.cpu()); scores.append(s.cpu()); labels.append(batch['label'])
            rows.extend({'sample_id': str(Path(p).relative_to(data)), 'label': int(y)}
                        for p, y in zip(batch['impath'], batch['label']))
        cache = {'features': torch.cat(feats), 'S': torch.cat(scores), 'labels': torch.cat(labels),
                 'views': m.residual_views.cpu().clone(), 'mask': m.residual_mask.cpu().clone(),
                 'scale': scale.cpu().clone()}
    if cache['S'].shape != (200, 50) or len({x['sample_id'] for x in rows}) != 200:
        raise ValueError('Expected 200 unique Base training images')
    cache_seconds = time.monotonic() - started
    torch.save(cache, out / 'train_cache.pt'); torch.save(initial, out / 'initial_residual.pt')
    metadata = {'scope': 'fixed Base training subset; no validation/test inference',
                'source_sha256': source_digest(), 'script_sha256': sha256_file(Path(__file__)),
                'anchor_checkpoint_sha256': sha256_file(cp), 'anchor_student_trained_shots': 16,
                'residual_fit_shots': 4, 'rows': rows, 'class_order': list(trainer.dm.dataset.classnames),
                'subset_sha256': sample_digest(items, data), 'transform': str(transform),
                'cache_seconds': cache_seconds, 'cache_device': str(trainer.device), 'fit_device': 'cpu',
                'train_cache_sha256': sha256_file(out / 'train_cache.pt'),
                'initial_residual_sha256': sha256_file(out / 'initial_residual.pt'),
                'fresh_residual_shared_init': True, 'student_frozen': True,
                'fit_signal': 'full fixed training CE reduction >=0.10 and no accuracy decrease',
                'max_steps_per_setting': 200, 'batch_size': 16, 'schedule_seed': 1729,
                'lr_schedule': 'constant for both; no warmup; diagnostic differs from prior cosine training',
                'automatic_expansion': False}
    write_json(out / 'cache_manifest.json', metadata)
    trainer.model.cpu(); del m, trainer; torch.cuda.empty_cache()
    batches, results = schedule(200), {}
    for name in SETTINGS:
        module = SharedTextResidual(512, 64, .2)
        module.load_state_dict(initial)
        results[name] = fit(module, cache, name, batches, out / name)
        print(json.dumps({'stage': name, 'initial': results[name]['initial'], 'final': results[name]['final']}), flush=True)
    result = {'state': 'completed', 'scope': metadata['scope'], 'settings': results,
              'adamw_minus_sgd_ce': results['adamw']['final']['ce'] - results['sgd']['final']['ce'],
              'optimizer_setting_sensitivity_signal': results['adamw']['final']['ce'] <= results['sgd']['final']['ce'] - .1,
              'any_train_fit_signal': any(x['train_fit_signal'] for x in results.values()),
              'elapsed_seconds': time.monotonic() - started, 'gpu_optimization_steps': 0,
              'cpu_optimization_steps': 400, 'automatic_expansion': False}
    write_json(out / 'analysis.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir', 'data-root', 'clip-path', 'output-root'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))
