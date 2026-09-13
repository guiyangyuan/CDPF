"""Validated TextConsistency with configurable aggregation and fixed inference rule."""
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from dassl.engine import TRAINER_REGISTRY
from clip import clip
from .textconsistency_common import (
    ViewAggregator, CachedMultiViewTextEncoder, TEMPLATE_FILES, build_view_aggregator,
    prediction_logits, load_complete_checkpoint, load_clip_checkpoint,
)
from .textconsistency_runtime import ValidatedTrainer
from .visual_adapter import ResidualVisualAdapter


def load_clip_to_cpu(cfg):
    return load_clip_checkpoint(cfg)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.TEXTCONSISTENCY.N_CTX
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        prompt_prefix = " ".join(["X"] * n_ctx)
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        return prompts

class HierarchicalViewAggregator(ViewAggregator):
    def __init__(self, dim, n_views, n_groups=4, hidden_dim=None, use_feature_adapter=True):
        super().__init__(dim, n_views, 'hierarchical', n_groups, use_feature_adapter, hidden_dim)


class MultiViewTextEncoder(CachedMultiViewTextEncoder):
    pass


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.multiview_encoder = MultiViewTextEncoder(cfg, classnames, clip_model)
        self.view_aggregator = build_view_aggregator(cfg, clip_model.text_projection.shape[-1])
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        tc = cfg.TRAINER.TEXTCONSISTENCY
        self.visual_adapter = (ResidualVisualAdapter(clip_model.text_projection.shape[-1],
                                                     tc.VISUAL_ADAPTER_DIM, tc.VISUAL_ADAPTER_SCALE)
                               if tc.USE_VISUAL_ADAPTER else None)

    def forward(self, image, return_attn=False):
        with torch.no_grad():
            image_features = F.normalize(self.image_encoder(image.to(self.dtype)), dim=-1)
        if self.visual_adapter is not None:
            image_features = self.visual_adapter(image_features)
        prompts = self.prompt_learner()
        text = F.normalize(self.text_encoder(prompts, self.tokenized_prompts), dim=-1)
        logits = self.logit_scale.exp() * image_features @ text.t()
        adaptive, attention = self.view_aggregator(image_features,
            self.multiview_encoder.get_view_features(), self.multiview_encoder.view_mask)
        teacher = self.logit_scale.exp() * torch.einsum('bd,bcd->bc', image_features, adaptive)
        return logits, teacher, image_features, attention if return_attn else None


@TRAINER_REGISTRY.register()
class CoOp_TextConsistency_V2_FG(ValidatedTrainer):
    model_builder = CustomCLIP

    def model_inference(self, image):
        student, teacher, _, _ = self.model(image, return_attn=False)
        tc = self.cfg.TRAINER.TEXTCONSISTENCY
        return prediction_logits(student, teacher, tc.INFERENCE_BRANCH, tc.ENSEMBLE_ALPHA)

    def load_model(self, directory, epoch=None):
        return load_complete_checkpoint(self, directory, epoch)
