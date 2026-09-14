"""Image-conditioned description detail around a classname text anchor.

The centered variant removes the uniform descriptor component. It uses the same
two adapter matrices as the classname control, not an additional learned head.
"""
import torch
from torch.nn import functional as F
from dassl.engine import TRAINER_REGISTRY

from .coop_text_residual import TextResidualCLIP
from .coop_text_residual_adamw import CoOp_TextResidual_AdamW


def description_radii(views, mask):
    """Masked RMS distance from each class's mean description embedding."""
    weights = mask.to(views.dtype)
    counts = weights.sum(-1, keepdim=True)
    means = (views * weights[..., None]).sum(1) / counts
    squared_distances = (views - means[:, None]).square().sum(-1)
    return ((squared_distances * weights).sum(-1) / counts.squeeze(-1)).sqrt()


def description_tangent_basis(views, mask):
    """Return an orthonormal basis for each centered description cloud."""
    weights = mask.to(views.dtype)
    counts = weights.sum(-1, keepdim=True)
    means = (views * weights[..., None]).sum(1) / counts
    centered = (views - means[:, None]) * weights[..., None]
    work = centered.float()
    _, singular_values, basis = torch.linalg.svd(work, full_matrices=False)
    relative_tolerance = max(work.shape[-2:]) * torch.finfo(work.dtype).eps
    tolerance = singular_values[:, :1] * relative_tolerance
    basis_mask = singular_values > tolerance
    return basis.to(views.dtype), basis_mask


def project_onto_description_tangent(delta, basis, basis_mask):
    """Keep only residual components supported by class descriptions."""
    basis = basis.to(device=delta.device, dtype=delta.dtype)
    coefficients = torch.einsum('bcd,cvd->bcv', delta, basis)
    coefficients = coefficients * basis_mask.to(device=delta.device, dtype=delta.dtype)[None]
    return torch.einsum('bcv,cvd->bcd', coefficients, basis)


def bound_by_description_radius(delta, radii, eps=1e-6):
    """Smoothly bound each class delta while keeping unit derivative at zero."""
    radius = radii.to(device=delta.device, dtype=delta.dtype)[None, :, None]
    ratio = torch.linalg.vector_norm(delta, dim=-1, keepdim=True) / radius.clamp_min(eps)
    small_ratio = 1. - ratio.square() / 3. + 2. * ratio.pow(4) / 15.
    regular_ratio = torch.tanh(ratio) / ratio.clamp_min(eps)
    factor = torch.where(ratio < 1e-3, small_ratio, regular_ratio)
    bounded = delta * factor
    return torch.where(radius > 0., bounded, torch.zeros_like(bounded))


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


class RadiusBoundedTextDetailCLIP(FusionAlignedTextDetailCLIP):
    """Bound each visual text delta by its class's description spread."""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        radii = description_radii(
            self.multiview_encoder.get_view_features(),
            self.multiview_encoder.view_mask,
        )
        self.register_buffer('description_radii', radii, persistent=False)
        self.correction_identity.update(
            version=5,
            kind='description_radius_bounded_visual_text_detail',
            inference='F+R_hat',
            objective='CE(S,y)+CE(stopgrad(F)+R_hat,y)',
            fusion_anchor='F=0.5*(S+M)',
            description_radius='sqrt(mean_valid(||description-mean_valid(description)||^2))',
            bounding='radius*tanh(||delta||/radius)*delta/(||delta||+eps)',
            residual='clip_scale*einsum(image,radius_bounded_per_image_text_delta)',
            coefficient=1.0,
            fitted_hyperparameters=[],
        )

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
        bounded_delta = bound_by_description_radius(delta, self.description_radii)
        residual = scale * torch.einsum('bd,bcd->bc', feature, bounded_delta)
        fusion = .5 * student + .5 * mean
        return {'S': student, 'M': mean, 'R': residual, 'Z': fusion + residual, 'F': fusion}


class SubspaceRadiusTextDetailCLIP(RadiusBoundedTextDetailCLIP):
    """Restrict each residual to its class-description tangent directions."""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        basis, basis_mask = description_tangent_basis(
            self.multiview_encoder.get_view_features(),
            self.multiview_encoder.view_mask,
        )
        self.register_buffer('description_tangent_basis', basis, persistent=False)
        self.register_buffer('description_basis_mask', basis_mask, persistent=False)
        self.correction_identity.update(
            version=6,
            kind='description_subspace_radius_visual_text_detail',
            inference='F+R_subspace_radius',
            objective='CE(S,y)+CE(stopgrad(F)+R_subspace_radius,y)',
            description_subspace='rowspan(description-mean_valid(description))',
            rank_rule='max(rows,dimensions)*float32_epsilon*largest_singular_value',
            operation_order='project_then_radius_bound',
            residual='clip_scale*einsum(image,subspace_projected_radius_bounded_delta)',
            fitted_hyperparameters=[],
        )

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
        projected = project_onto_description_tangent(
            delta, self.description_tangent_basis, self.description_basis_mask)
        bounded_delta = bound_by_description_radius(projected, self.description_radii)
        residual = scale * torch.einsum('bd,bcd->bc', feature, bounded_delta)
        fusion = .5 * student + .5 * mean
        return {'S': student, 'M': mean, 'R': residual, 'Z': fusion + residual, 'F': fusion}


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


@TRAINER_REGISTRY.register()
class CoOp_TextDetail_RadiusBounded(CoOp_TextDetail_FusionAligned):
    """Use the frozen description cloud as a per-class residual trust region."""
    model_builder = RadiusBoundedTextDetailCLIP


@TRAINER_REGISTRY.register()
class CoOp_TextDetail_SubspaceRadius(CoOp_TextDetail_RadiusBounded):
    """Project residuals into frozen class-description tangents before bounding."""
    model_builder = SubspaceRadiusTextDetailCLIP
