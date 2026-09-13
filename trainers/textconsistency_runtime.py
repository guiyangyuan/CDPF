"""Training, checkpoint provenance and machine-readable evaluation for TC."""
import hashlib
import json
import math
import platform
from pathlib import Path

import torch
from torch import nn
from dassl.engine import TrainerX
from dassl.optim import build_lr_scheduler
from dassl.metrics import compute_accuracy
from .textconsistency_common import (
    PROJECT_ROOT, load_clip_checkpoint, select_device, training_losses, sha256_file,
    description_permutation,
)
from .visual_adapter import validate_visual_adapter_config


def _loss_identity(mode, temperature, effective_weight):
    weighting = {
        'full_kl': 'none_full_distribution_kl',
        'weighted_nckd': 'teacher_probability_mass_outside_ground_truth',
    }
    if mode not in weighting:
        raise ValueError(f'Unknown KD loss mode: {mode}')
    return {
        'format': 1,
        'mode': mode,
        'temperature': float(temperature),
        'effective_weight': float(effective_weight),
        'active': effective_weight > 0,
        'non_target_weighting': weighting[mode],
    }


def sample_digest(items, root):
    records = []
    for item in items or []:
        path = Path(item.impath)
        try:
            path = path.relative_to(Path(root).resolve())
        except ValueError:
            raise ValueError(f'Dataset image is outside DATASET.ROOT: {path}')
        records.append((str(path), int(item.label), item.classname))
    return hashlib.sha256(json.dumps(records, ensure_ascii=False).encode()).hexdigest()


def source_digest():
    h = hashlib.sha256()
    files = [PROJECT_ROOT / 'train.py']
    for folder in ('trainers', 'datasets', 'clip', 'Dassl.pytorch/dassl'):
        files.extend(sorted((PROJECT_ROOT / folder).rglob('*.py')))
    for path in files:
        h.update(str(path.relative_to(PROJECT_ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


class ValidatedTrainer(TrainerX):
    def check_cfg(self, cfg):
        validate_visual_adapter_config(cfg)
        tc = cfg.TRAINER.TEXTCONSISTENCY
        if tc.PREC != 'fp32':
            raise ValueError('Use PREC fp32 for the validated reproduction; mixed precision is not yet validated')
        loss_scalars = (tc.KD_TEMP, tc.KD_WEIGHT, tc.ADAPTIVE_CE_WEIGHT,
                        tc.DIVERSITY_WEIGHT, tc.CONTRASTIVE_WEIGHT)
        if (not all(math.isfinite(float(value)) for value in loss_scalars)
                or tc.KD_TEMP <= 0
                or min(tc.KD_WEIGHT, tc.ADAPTIVE_CE_WEIGHT,
                       tc.DIVERSITY_WEIGHT, tc.CONTRASTIVE_WEIGHT) < 0):
            raise ValueError('Invalid loss configuration')
        if tc.CONTRASTIVE_WEIGHT != 0:
            raise ValueError('Disable the old frozen-feature contrastive loss (CONTRASTIVE_WEIGHT=0)')
        if cfg.SEED < 0:
            raise ValueError('A nonnegative explicit SEED is required')
        if tc.TEXT_BATCH_SIZE < 1 or tc.N_VIEWS < 1 or tc.N_GROUPS < 1:
            raise ValueError('Description batch size, view count and group count must be positive')
        if tc.KD_LOSS_MODE not in ('full_kl', 'weighted_nckd'):
            raise ValueError(f'Unknown KD loss mode: {tc.KD_LOSS_MODE}')
        description_permutation(tc.N_VIEWS, tc.VIEW_ORDER, tc.N_GROUPS, tc.ORDER_SEED)
        if tc.VIEW_ORDER != 'contiguous' and (tc.AGGREGATION != 'hierarchical' or tc.ABLATION_GROUP != 'full'):
            raise ValueError('Description order controls require the full hierarchical model')

    def build_model(self):
        cfg, tc = self.cfg, self.cfg.TRAINER.TEXTCONSISTENCY
        self.device = select_device(cfg)
        clip_model = load_clip_checkpoint(cfg)
        self.model = self.model_builder(cfg, self.dm.dataset.classnames, clip_model)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(name.startswith(('prompt_learner.', 'view_aggregator.', 'visual_adapter.')))
        self.model.to(self.device)
        self.model.image_encoder.eval()
        self.model.text_encoder.eval()
        self.trainable_modules = nn.ModuleDict({'prompt_learner': self.model.prompt_learner,
                                                'view_aggregator': self.model.view_aggregator})
        visual_adapter = getattr(self.model, 'visual_adapter', None)
        if visual_adapter is not None:
            self.trainable_modules['visual_adapter'] = visual_adapter
        parameters = [p for p in self.trainable_modules.parameters() if p.requires_grad]
        self.optim = torch.optim.SGD(parameters, lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM,
                                     weight_decay=cfg.OPTIM.WEIGHT_DECAY)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('textconsistency', self.trainable_modules, self.optim, self.sched)
        self.group = getattr(self.model, 'group', None)
        self.kd_weight = tc.KD_WEIGHT if self.group is None or self.group['use_kd'] else 0.
        self.adaptive_ce_weight = tc.ADAPTIVE_CE_WEIGHT
        self.diversity_weight = tc.DIVERSITY_WEIGHT
        if self.group is not None and not self.group['use_fg']:
            self.diversity_weight = 0.
        if any(p.requires_grad for p in self.model.view_aggregator.parameters()) and self.adaptive_ce_weight <= 0:
            raise ValueError('A trainable aggregator requires positive ADAPTIVE_CE_WEIGHT')
        encoder = self.model.multiview_encoder
        loss_identity = _loss_identity(tc.KD_LOSS_MODE, tc.KD_TEMP, self.kd_weight)
        self.repro_signature = {
            'format': 2, 'trainer': cfg.TRAINER.NAME, 'n_ctx': tc.N_CTX,
            'aggregation': getattr(self.model.view_aggregator, 'mode', 'none'),
            'feature_adapter': getattr(self.model.view_aggregator, 'fine_grained_proj', None) is not None,
            'n_groups': getattr(self.model.view_aggregator, 'n_groups', None),
            'n_views': tc.N_VIEWS, 'ablation_group': tc.ABLATION_GROUP if self.group is not None else None,
            'clip_sha256': clip_model._repro_sha256,
            'template_sha256': getattr(encoder, 'template_sha256', None),
            'view_order': tc.VIEW_ORDER, 'order_seed': tc.ORDER_SEED, 'order_version': 1,
            'loss_identity': dict(loss_identity),
        }
        if visual_adapter is not None:
            self.repro_signature.update(format=3,
                visual_adapter=visual_adapter.identity(self.adaptive_ce_weight))
        self.run_provenance = {
            'seed': cfg.SEED, 'dataset': cfg.DATASET.NAME, 'shots': cfg.DATASET.NUM_SHOTS,
            'subsample': cfg.DATASET.SUBSAMPLE_CLASSES, 'model_signature': self.repro_signature,
            'code_sha256': source_digest(), 'torch': torch.__version__, 'python': platform.python_version(),
            'device': str(self.device), 'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else None,
            'loss_identity': dict(loss_identity),
            'effective_losses': {'kd': self.kd_weight, 'adaptive_ce': self.adaptive_ce_weight,
                                 'diversity': self.diversity_weight, 'contrastive': 0.},
            'train_samples_sha256': sample_digest(self.dm.dataset.train_x, cfg.DATASET.ROOT),
            'val_samples_sha256': sample_digest(self.dm.dataset.val, cfg.DATASET.ROOT),
            'test_samples_sha256': sample_digest(self.dm.dataset.test, cfg.DATASET.ROOT),
            'trainable_parameters': sum(p.numel() for p in parameters),
            'description_counts': getattr(encoder, 'description_counts', []),
            'view_order_manifest': getattr(encoder, 'view_order_manifest', None),
        }
        directory = Path(cfg.OUTPUT_DIR); directory.mkdir(parents=True, exist_ok=True)
        (directory / 'run_manifest.json').write_text(json.dumps(self.run_provenance, indent=2))
        (directory / 'effective_config.yaml').write_text(cfg.dump())
        print('Effective protocol: ' + json.dumps(self.run_provenance, sort_keys=True))

    def set_model_mode(self, mode='train', names=None):
        super().set_model_mode(mode, names)
        if hasattr(self, 'model'):
            self.model.image_encoder.eval()
            self.model.text_encoder.eval()

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        logits, teacher, _, attention = self.model(image, return_attn=True)
        loss, pieces = training_losses(logits, teacher, label, attention, self.kd_weight,
                                       self.cfg.TRAINER.TEXTCONSISTENCY.KD_TEMP,
                                       self.adaptive_ce_weight, self.diversity_weight,
                                       kd_loss_mode=self.cfg.TRAINER.TEXTCONSISTENCY.KD_LOSS_MODE)
        self.model_backward_and_update(loss)
        if self.batch_idx + 1 == self.num_batches:
            self.update_lr()
        return {'loss': loss.item(), **{k: v.item() for k, v in pieces.items()},
                'acc': compute_accuracy(logits, label)[0].item()}

    def parse_batch_train(self, batch):
        return batch['img'].to(self.device), batch['label'].to(self.device)

    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=''):
        for name in self.get_model_names():
            folder = Path(directory) / name; folder.mkdir(parents=True, exist_ok=True)
            filename = model_name or ('model-best.pth.tar' if is_best else f'model.pth.tar-{epoch + 1}')
            path = folder / filename
            state = {'state_dict': self._models[name].state_dict(), 'epoch': epoch + 1,
                     'optimizer': self.optim.state_dict(), 'scheduler': self.sched.state_dict(),
                     'val_result': val_result, 'repro_signature': self.repro_signature,
                     'run_provenance': self.run_provenance}
            temporary = path.with_name(path.name + '.tmp')
            torch.save(state, temporary); temporary.replace(path)
            (folder / 'checkpoint').write_text(filename)
            (path.with_name(path.name + '.sha256')).write_text(sha256_file(path) + '\n')
            print(f'Checkpoint saved to {path}')

    @torch.no_grad()
    def test(self, split=None, is_final=False):
        self.set_model_mode('eval')
        split = split or self.cfg.TEST.SPLIT
        if split == 'val' and self.val_loader is not None:
            loader = self.val_loader
        else:
            split, loader = 'test', self.test_loader
        self.evaluator.reset()
        for batch in loader:
            image, label = self.parse_batch_test(batch)
            self.evaluator.process(self.model_inference(image), label)
        results = self.evaluator.evaluate()
        tc = self.cfg.TRAINER.TEXTCONSISTENCY
        payload = {**self.run_provenance, 'split': split, 'branch': tc.INFERENCE_BRANCH,
                   'ensemble_alpha': tc.ENSEMBLE_ALPHA,
                   'checkpoints': getattr(self, 'loaded_checkpoints', []),
                   'state': 'loaded_checkpoint' if hasattr(self, 'loaded_checkpoints') else 'training_in_memory',
                   'metrics': {k: float(v) for k, v in results.items()}}
        Path(self.cfg.OUTPUT_DIR, 'metrics.json').write_text(json.dumps(payload, indent=2))
        return float(results['accuracy'])
