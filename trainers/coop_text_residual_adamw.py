"""Keep the original student SGD while separately optimizing text residuals."""
import math

import torch
from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_lr_scheduler
from .coop_text_residual import CoOp_TextResidual


class SplitOptimizers:
    """The small optimizer interface used by TrainerBase and checkpoint loading.

    The scheduler receives the actual student SGD, never this container, so it
    cannot accidentally anneal the residual optimizer's constant learning rate.
    """
    def __init__(self, student, residual):
        self.student, self.residual = student, residual

    @property
    def param_groups(self):
        return self.student.param_groups + self.residual.param_groups

    def zero_grad(self):
        self.student.zero_grad(set_to_none=True)
        self.residual.zero_grad(set_to_none=True)

    def step(self):
        self.student.step()
        self.residual.step()

    def state_dict(self):
        return {'format': 'student_sgd_residual_adamw_v1',
                'student': self.student.state_dict(), 'residual': self.residual.state_dict()}

    def load_state_dict(self, state):
        if state.get('format') != 'student_sgd_residual_adamw_v1':
            raise ValueError('Expected a complete split-optimizer checkpoint')
        self.student.load_state_dict(state['student'])
        self.residual.load_state_dict(state['residual'])


@TRAINER_REGISTRY.register()
class CoOp_TextResidual_AdamW(CoOp_TextResidual):
    def check_cfg(self, cfg):
        super().check_cfg(cfg)
        lr = cfg.TRAINER.TEXTCONSISTENCY.TEXT_RESIDUAL_ADAMW_LR
        if not math.isfinite(float(lr)) or lr <= 0:
            raise ValueError('Residual AdamW LR must be finite and positive')

    def build_optimizers(self, parameters):
        cfg = self.cfg
        student = [p for p in self.model.prompt_learner.parameters() if p.requires_grad]
        residual = [p for p in self.model.text_residual.parameters() if p.requires_grad]
        student_ids, residual_ids = set(map(id, student)), set(map(id, residual))
        if student_ids & residual_ids or student_ids | residual_ids != set(map(id, parameters)):
            raise ValueError('The two optimizers must cover exactly the disjoint trainable parameters')
        sgd = torch.optim.SGD(student, lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM,
                              weight_decay=cfg.OPTIM.WEIGHT_DECAY)
        adamw = torch.optim.AdamW(residual, lr=cfg.TRAINER.TEXTCONSISTENCY.TEXT_RESIDUAL_ADAMW_LR,
                                  betas=(.9, .999), eps=1e-8, weight_decay=.0005)
        self.optimization_identity = {
            'format': 1, 'student': {'name': 'sgd', 'lr': cfg.OPTIM.LR,
                'momentum': cfg.OPTIM.MOMENTUM, 'weight_decay': cfg.OPTIM.WEIGHT_DECAY,
                'scheduler': cfg.OPTIM.LR_SCHEDULER, 'max_epoch': cfg.OPTIM.MAX_EPOCH,
                'warmup_epoch': cfg.OPTIM.WARMUP_EPOCH, 'warmup_type': cfg.OPTIM.WARMUP_TYPE,
                'warmup_cons_lr': cfg.OPTIM.WARMUP_CONS_LR},
            'residual': {'name': 'adamw', 'lr': cfg.TRAINER.TEXTCONSISTENCY.TEXT_RESIDUAL_ADAMW_LR,
                'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .0005,
                'scheduler': 'constant', 'warmup_epoch': 0},
            'student_gradient': 'student_ce_only', 'residual_gradient': 'correction_ce_only',
        }
        return SplitOptimizers(sgd, adamw), build_lr_scheduler(sgd, cfg.OPTIM)
