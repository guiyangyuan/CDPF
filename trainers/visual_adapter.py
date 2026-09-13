"""Shared residual adaptation of classification features (prototype, no KD)."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def validate_visual_adapter_config(cfg):
    tc = cfg.TRAINER.TEXTCONSISTENCY
    enabled = getattr(tc, 'USE_VISUAL_ADAPTER', False)
    if not isinstance(enabled, bool):
        raise ValueError('USE_VISUAL_ADAPTER must be a boolean')
    if not enabled:
        return
    if cfg.TRAINER.NAME != 'CoOp_TextConsistency_V2_FG':
        raise ValueError('Visual adaptation is only supported by CoOp_TextConsistency_V2_FG')
    width, scale = tc.VISUAL_ADAPTER_DIM, tc.VISUAL_ADAPTER_SCALE
    if type(width) is not int or width < 1:
        raise ValueError('VISUAL_ADAPTER_DIM must be a positive integer')
    if not math.isfinite(float(scale)) or not 0 < scale <= 1:
        raise ValueError('VISUAL_ADAPTER_SCALE must be finite and in (0, 1]')
    if (tc.AGGREGATION != 'mean' or tc.USE_FEATURE_ADAPTER
            or tc.ABLATION_GROUP != 'full' or tc.KD_WEIGHT != 0
            or tc.KD_LOSS_MODE != 'full_kl' or tc.DIVERSITY_WEIGHT != 0
            or tc.CONTRASTIVE_WEIGHT != 0):
        raise ValueError('Visual adaptation requires mean descriptions, no query adapter, '
                         'full group, and no KD/diversity/contrastive loss')


class ResidualVisualAdapter(nn.Module):
    """Unit CLIP feature -> shared bottleneck residual -> unit classifier feature.

    Zero up-projection starts at the original direction. Initialization must not
    advance the training RNG relative to the no-adapter control.
    """
    def __init__(self, dim, hidden_dim, scale):
        super().__init__()
        self.hidden_dim, self.scale = hidden_dim, float(scale)
        with torch.random.fork_rng(devices=[]):
            self.down = nn.Linear(dim, hidden_dim, bias=False)
            self.up = nn.Linear(hidden_dim, dim, bias=False)
            nn.init.zeros_(self.up.weight)

    def forward(self, features):
        return F.normalize(features + self.scale * self.up(F.relu(self.down(features))), dim=-1)

    def identity(self, description_ce_weight):
        return {'format': 1, 'kind': 'shared_residual_mlp',
                'hidden_dim': self.hidden_dim, 'scale': self.scale,
                'init': 'kaiming_down_zero_up', 'bias': False,
                'input': 'unit_clip_image', 'output': 'unit_residual_feature',
                'placement': 'student_and_description_logits',
                'description_ce_weight': float(description_ce_weight)}
