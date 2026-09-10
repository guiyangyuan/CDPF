"""Legacy component ablations. For isolated aggregation controls use run_pilot.py."""
import torch
from torch import nn
from torch.nn import functional as F
from dassl.engine import TRAINER_REGISTRY
from .coop_textconsistency_v2_fg import PromptLearner, TextEncoder
from .textconsistency_common import (
    CachedMultiViewTextEncoder, ViewAggregator, prediction_logits, load_complete_checkpoint,
)
from .textconsistency_runtime import ValidatedTrainer

ABLATION_GROUPS = {
    "baseline": {
        "n_views": 0,
        "use_teacher": False,
        "use_adaptive": False,
        "use_fg": False,
        "use_kd": False,
    },
    "single_view_kd": {
        "n_views": 1,
        "use_teacher": True,
        "use_adaptive": False,
        "use_fg": False,
        "use_kd": True,
    },
    "no_kd": {
        "n_views": None,
        "use_teacher": True,
        "use_adaptive": True,
        "use_fg": False,
        "use_kd": False,
    },
    "v2_no_fg": {
        "n_views": None,
        "use_teacher": True,
        "use_adaptive": True,
        "use_fg": False,
        "use_kd": True,
    },
    "full": {
        "n_views": None,
        "use_teacher": True,
        "use_adaptive": True,
        "use_fg": True,
        "use_kd": True,
    },
}

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        name = cfg.TRAINER.TEXTCONSISTENCY.ABLATION_GROUP
        self.group = ABLATION_GROUPS[name]
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale, self.dtype = clip_model.logit_scale, clip_model.dtype
        self.multiview_encoder, self.view_aggregator = None, nn.Identity()
        if self.group['use_teacher']:
            effective = cfg.clone(); effective.defrost()
            tc = effective.TRAINER.TEXTCONSISTENCY
            if self.group['n_views'] is not None:
                tc.N_VIEWS = self.group['n_views']
            self.multiview_encoder = CachedMultiViewTextEncoder(effective, classnames, clip_model)
            mode = 'hierarchical' if self.group['use_fg'] else ('flat' if self.group['use_adaptive'] else 'mean')
            self.view_aggregator = ViewAggregator(clip_model.text_projection.shape[-1], tc.N_VIEWS,
                                                  mode, tc.N_GROUPS, self.group['use_fg'])

    def forward(self, image, return_attn=False):
        with torch.no_grad():
            features = F.normalize(self.image_encoder(image.to(self.dtype)), dim=-1)
        text = F.normalize(self.text_encoder(self.prompt_learner(), self.tokenized_prompts), dim=-1)
        student = self.logit_scale.exp() * features @ text.t()
        teacher, attention = None, None
        if self.multiview_encoder is not None:
            text, attention = self.view_aggregator(features, self.multiview_encoder.get_view_features(),
                                                   self.multiview_encoder.view_mask)
            teacher = self.logit_scale.exp() * torch.einsum('bd,bcd->bc', features, text)
        return student, teacher, features, attention if return_attn else None


@TRAINER_REGISTRY.register()
class CoOp_TextConsistency_Ablation(ValidatedTrainer):
    model_builder = CustomCLIP

    def check_cfg(self, cfg):
        super().check_cfg(cfg)
        if cfg.TRAINER.TEXTCONSISTENCY.ABLATION_GROUP not in ABLATION_GROUPS:
            raise ValueError('Unknown ablation group')

    def model_inference(self, image):
        student, teacher, _, _ = self.model(image, return_attn=False)
        tc = self.cfg.TRAINER.TEXTCONSISTENCY
        return prediction_logits(student, teacher, tc.INFERENCE_BRANCH, tc.ENSEMBLE_ALPHA)

    def load_model(self, directory, epoch=None):
        return load_complete_checkpoint(self, directory, epoch)
