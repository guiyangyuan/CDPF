"""Class-shared text residuals with optional text-only neighbor centering.

The student receives only its own CE. No class-indexed parameter, image-neighbor
lookup, test labels, or validation-selected fusion coefficient is used.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from dassl.engine import TRAINER_REGISTRY
from clip import clip

from .coop_textconsistency_v2_fg import PromptLearner, TextEncoder
from .coop_textconsistency_correction import CoOp_TextConsistency_Correction
from .textconsistency_common import CachedMultiViewTextEncoder, build_view_aggregator, select_device
from .textconsistency_runtime import ValidatedTrainer
from .zsclip import CUSTOM_TEMPLATES


def neighbor_inputs(views, names, k):
    """Remove the mean of nearest *other* classname text embeddings.

    Stable sorting fixes ties by current class order. Novel inference rebuilds
    this geometry from the new candidate class texts only.
    """
    if len(names) < 2:
        raise ValueError('Neighbor centering needs at least two candidate classes')
    similarity = names @ names.T
    similarity.fill_diagonal_(-torch.inf)
    neighbors = similarity.argsort(dim=1, descending=True, stable=True)[:, :min(k, len(names) - 1)]
    reference = names[neighbors].mean(1)
    return views - reference[:, None, :], neighbors


class SharedTextResidual(nn.Module):
    def __init__(self, dim, hidden_dim, scale):
        super().__init__()
        self.scale = float(scale)
        with torch.random.fork_rng(devices=[]):
            self.down = nn.Linear(dim, hidden_dim, bias=False)
            self.up = nn.Linear(hidden_dim, dim, bias=False)
            nn.init.zeros_(self.up.weight)

    def forward(self, views, mask):
        hidden = F.relu(self.down(views))
        pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1)[:, None]
        # No normalization of this delta: preserve its exact zero starting point.
        return self.scale * self.up(pooled)


class TextResidualCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        tc = cfg.TRAINER.TEXTCONSISTENCY
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.multiview_encoder = CachedMultiViewTextEncoder(cfg, classnames, clip_model)
        self.view_aggregator = build_view_aggregator(cfg, clip_model.text_projection.shape[-1])
        self.image_encoder, self.text_encoder = clip_model.visual, TextEncoder(clip_model)
        self.logit_scale, self.dtype = clip_model.logit_scale, clip_model.dtype
        self.text_residual = SharedTextResidual(clip_model.text_projection.shape[-1],
                                               tc.TEXT_RESIDUAL_DIM, tc.TEXT_RESIDUAL_SCALE)
        template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        device, original = select_device(cfg), next(clip_model.parameters()).device
        clip_model.to(device).eval()
        tokens = clip.tokenize([template.format(name.replace('_', ' ')) for name in classnames])
        with torch.no_grad():
            names = torch.cat([F.normalize(clip_model.encode_text(chunk.to(device)).float(), dim=-1).cpu()
                               for chunk in tokens.split(tc.TEXT_BATCH_SIZE)])
        clip_model.to(original)
        views = self.multiview_encoder.get_view_features()
        mask = self.multiview_encoder.view_mask
        self.neighbor_indices = None
        if tc.TEXT_RESIDUAL_MODE == 'classname':
            views, mask = names[:, None, :], torch.ones(len(names), 1, dtype=torch.bool)
        elif tc.TEXT_RESIDUAL_MODE == 'contrast':
            views, neighbors = neighbor_inputs(views, names, tc.TEXT_RESIDUAL_K)
            self.neighbor_indices = neighbors.tolist()
        self.register_buffer('residual_views', views.clone(), persistent=False)
        self.register_buffer('residual_mask', mask.clone(), persistent=False)
        self.correction_identity = {
            'version': 2, 'enabled': True, 'kind': 'shared_text_residual',
            'mode': tc.TEXT_RESIDUAL_MODE, 'hidden_dim': tc.TEXT_RESIDUAL_DIM,
            'scale': tc.TEXT_RESIDUAL_SCALE, 'neighbor_k': tc.TEXT_RESIDUAL_K,
            'classname_template': template, 'init': 'kaiming_down_zero_up_no_bias_rng_isolated',
            'input': 'unit_text_views_minus_neighbor_mean' if tc.TEXT_RESIDUAL_MODE == 'contrast' else 'unit_text_views',
            'pooling': 'masked_mean_after_relu_down', 'delta_normalization': False,
            'residual': 'clip_scale*original_image@text_delta.T', 'inference': 'S+R',
            'coefficient': 1.0, 'objective': 'CE(S,y)+CE(stopgrad(S)+R,y)',
        }

    def forward(self, image):
        with torch.no_grad():
            feature = F.normalize(self.image_encoder(image.to(self.dtype)), dim=-1)
        text = F.normalize(self.text_encoder(self.prompt_learner(), self.tokenized_prompts), dim=-1)
        scale = self.logit_scale.exp()
        student = scale * feature @ text.T
        descriptions, _ = self.view_aggregator(feature, self.multiview_encoder.get_view_features(),
                                              self.multiview_encoder.view_mask)
        mean = scale * torch.einsum('bd,bcd->bc', feature, descriptions)
        delta = self.text_residual(self.residual_views, self.residual_mask)
        residual = scale * feature @ delta.T
        return {'S': student, 'M': mean, 'R': residual, 'Z': student + residual,
                'F': 0.5 * student + 0.5 * mean}


@TRAINER_REGISTRY.register()
class CoOp_TextResidual(CoOp_TextConsistency_Correction):
    model_builder = TextResidualCLIP

    def check_cfg(self, cfg):
        ValidatedTrainer.check_cfg(self, cfg)
        tc = cfg.TRAINER.TEXTCONSISTENCY
        if (tc.TEXT_RESIDUAL_MODE not in ('classname', 'description', 'contrast')
                or type(tc.TEXT_RESIDUAL_DIM) is not int or tc.TEXT_RESIDUAL_DIM < 1
                or type(tc.TEXT_RESIDUAL_K) is not int or tc.TEXT_RESIDUAL_K < 1
                or not math.isfinite(float(tc.TEXT_RESIDUAL_SCALE)) or not 0 < tc.TEXT_RESIDUAL_SCALE <= 1):
            raise ValueError('Invalid text residual mode/dimension/scale/neighbors')
        if (tc.AGGREGATION != 'mean' or tc.USE_FEATURE_ADAPTER or tc.USE_VISUAL_ADAPTER
                or tc.ABLATION_GROUP != 'full' or tc.KD_WEIGHT != 0 or tc.ADAPTIVE_CE_WEIGHT != 0
                or tc.DIVERSITY_WEIGHT != 0 or tc.CONTRASTIVE_WEIGHT != 0
                or tc.KD_LOSS_MODE != 'full_kl' or tc.VIEW_ORDER != 'contiguous'
                or tc.INFERENCE_BRANCH not in self.BRANCHES or tc.ENSEMBLE_ALPHA != .5):
            raise ValueError('Text residual requires protected CoOp, fixed mean reference, and correction CE only')

    def loss_terms(self, outputs, labels):
        return (F.cross_entropy(outputs['S'], labels),
                F.cross_entropy(outputs['S'].detach() + outputs['R'], labels))
