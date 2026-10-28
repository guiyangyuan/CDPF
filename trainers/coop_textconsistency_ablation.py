"""Validated CoOp student used by the fixed probability-fusion method."""
import torch
from torch import nn
from torch.nn import functional as F
from dassl.engine import TRAINER_REGISTRY
from clip import clip
from .textconsistency_common import prediction_logits, load_complete_checkpoint
from .textconsistency_runtime import ValidatedTrainer


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        value = prompts + self.positional_embedding.type(self.dtype)
        value = value.permute(1, 0, 2)
        value = self.transformer(value)
        value = value.permute(1, 0, 2)
        value = self.ln_final(value).type(self.dtype)
        return value[torch.arange(value.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_ctx = cfg.TRAINER.TEXTCONSISTENCY.N_CTX
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        context = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(context, std=0.02)
        self.ctx = nn.Parameter(context)

        prefix = " ".join(["X"] * n_ctx)
        prompts = [prefix + " " + name.replace("_", " ") + "." for name in classnames]
        tokenized = torch.cat([clip.tokenize(prompt) for prompt in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.n_cls = len(classnames)
        self.tokenized_prompts = tokenized

    def forward(self):
        context = self.ctx
        if context.dim() == 2:
            context = context.unsqueeze(0).expand(self.n_cls, -1, -1)
        return torch.cat([self.token_prefix, context, self.token_suffix], dim=1)

ABLATION_GROUPS = {
    "baseline": {
        "n_views": 0,
        "use_teacher": False,
        "use_adaptive": False,
        "use_fg": False,
        "use_kd": False,
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

    def forward(self, image, return_attn=False):
        with torch.no_grad():
            features = F.normalize(self.image_encoder(image.to(self.dtype)), dim=-1)
        text = F.normalize(self.text_encoder(self.prompt_learner(), self.tokenized_prompts), dim=-1)
        student = self.logit_scale.exp() * features @ text.t()
        return student, None, features, None


@TRAINER_REGISTRY.register()
class CoOp_TextConsistency_Ablation(ValidatedTrainer):
    model_builder = CustomCLIP

    def check_cfg(self, cfg):
        super().check_cfg(cfg)
        if cfg.TRAINER.TEXTCONSISTENCY.ABLATION_GROUP != 'baseline':
            raise ValueError('The current model trains only the baseline CoOp student')

    def model_inference(self, image):
        student, teacher, _, _ = self.model(image, return_attn=False)
        tc = self.cfg.TRAINER.TEXTCONSISTENCY
        return prediction_logits(student, teacher, tc.INFERENCE_BRANCH, tc.ENSEMBLE_ALPHA)

    def load_model(self, directory, epoch=None):
        return load_complete_checkpoint(self, directory, epoch)
