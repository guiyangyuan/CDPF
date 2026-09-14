"""Image-conditioned description detail around a classname text anchor.

The centered variant removes the uniform descriptor component. It uses the same
two adapter matrices as the classname control, not an additional learned head.
"""
import torch
from torch.nn import functional as F
from dassl.engine import TRAINER_REGISTRY

from .coop_text_residual import TextResidualCLIP
from .coop_text_residual_adamw import CoOp_TextResidual_AdamW


def description_attention(features, views, mask, scale):
    logits = scale * torch.einsum('bd,cvd->bcv', features, views)
    return logits.masked_fill(~mask[None], -torch.inf).softmax(-1)


def description_weights(features, views, mask, scale, pooling):
    if pooling == 'uniform':
        uniform = mask.to(features.dtype) / mask.sum(-1, keepdim=True)
        return uniform[None].expand(len(features), -1, -1)
    return description_attention(features, views, mask, scale)


def anchored_delta(adapter, names, views, mask, attention, centered):
    hidden = F.relu(adapter.down(views))
    name_hidden = F.relu(adapter.down(names))
    weights = attention
    if centered:
        uniform = mask.to(weights.dtype) / mask.sum(-1, keepdim=True)
        weights = weights - uniform[None]
    detail = torch.einsum('bcv,cvh->bch', weights, hidden)
    return adapter.scale * adapter.up(name_hidden[None] + detail)


class TextDetailCLIP(TextResidualCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.detail_pooling = cfg.TRAINER.TEXTCONSISTENCY.TEXT_DETAIL_POOLING
        # The parent is fixed to classname mode; this buffer is rebuilt for new classes.
        self.register_buffer('detail_names', self.residual_views[:, 0].clone(), persistent=False)
        attention_identity = ('uniform_valid_description_weights' if self.detail_pooling == 'uniform'
                              else 'masked_softmax(frozen_clip_scale*image@unit_views)')
        input_identity = ('classname_hidden_plus_uniform_description_hidden'
                          if self.detail_pooling == 'uniform'
                          else 'classname_hidden_plus_image_weighted_description_hidden')
        self.correction_identity.update(version=3, kind='classname_anchored_visual_text_detail',
            detail_pooling=self.detail_pooling, attention=attention_identity,
            input=input_identity,
            centering='subtract_uniform_weights' if self.detail_pooling == 'centered' else 'none',
            pooling=('uniform_after_relu_down' if self.detail_pooling == 'uniform'
                     else 'image_conditioned_after_relu_down'),
            residual='clip_scale*einsum(image,per_image_text_delta)',
            classname_training_preserved=False)

    def forward(self, image):
        with torch.no_grad():
            feature = F.normalize(self.image_encoder(image.to(self.dtype)), dim=-1)
        text = F.normalize(self.text_encoder(self.prompt_learner(), self.tokenized_prompts), dim=-1)
        scale = self.logit_scale.exp()
        student = scale * feature @ text.T
        views, mask = self.multiview_encoder.get_view_features(), self.multiview_encoder.view_mask
        descriptions, _ = self.view_aggregator(feature, views, mask)
        mean = scale * torch.einsum('bd,bcd->bc', feature, descriptions)
        attention = description_weights(feature, views, mask, scale, self.detail_pooling)
        delta = anchored_delta(self.text_residual, self.detail_names, views, mask, attention,
                               self.detail_pooling == 'centered')
        residual = scale * torch.einsum('bd,bcd->bc', feature, delta)
        return {'S': student, 'M': mean, 'R': residual, 'Z': student + residual,
                'F': .5 * student + .5 * mean}


class FusionAlignedTextDetailCLIP(TextDetailCLIP):
    """Train and deploy the residual around the same multimodal anchor."""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.correction_identity.update(
            version=4,
            kind='fusion_aligned_visual_text_detail',
            inference='F+R',
            objective='CE(S,y)+CE(stopgrad(F)+R,y)',
            fusion_anchor='F=0.5*(S+M)',
            coefficient=1.0,
        )

    def forward(self, image):
        outputs = super().forward(image)
        outputs['Z'] = outputs['F'] + outputs['R']
        return outputs


@TRAINER_REGISTRY.register()
class CoOp_TextDetail(CoOp_TextResidual_AdamW):
    model_builder = TextDetailCLIP

    def check_cfg(self, cfg):
        super().check_cfg(cfg)
        tc = cfg.TRAINER.TEXTCONSISTENCY
        if tc.TEXT_RESIDUAL_MODE != 'classname' or tc.TEXT_DETAIL_POOLING not in ('uniform', 'attentive', 'centered'):
            raise ValueError('Text detail requires classname anchor and uniform/attentive/centered pooling')


@TRAINER_REGISTRY.register()
class CoOp_TextDetail_FusionAligned(CoOp_TextDetail):
    """Optimize the text residual on the exact fusion branch used at inference."""
    model_builder = FusionAlignedTextDetailCLIP

    def loss_terms(self, outputs, labels):
        return (F.cross_entropy(outputs['S'], labels),
                F.cross_entropy(outputs['F'].detach() + outputs['R'], labels))
