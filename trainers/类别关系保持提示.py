"""Description-geometry regularization for shared CoOp prompts."""
import math

import torch
from torch.nn import functional as F
from dassl.engine import TRAINER_REGISTRY

from .coop_textconsistency_v2_fg import (
    CoOp_TextConsistency_V2_FG,
    CustomCLIP,
)
from .textconsistency_common import training_losses


def category_relation_loss(prompt_prototypes, description_prototypes):
    """Match off-diagonal class-similarity geometry to frozen descriptions."""
    if (prompt_prototypes.ndim != 2 or description_prototypes.ndim != 2
            or prompt_prototypes.shape != description_prototypes.shape
            or prompt_prototypes.shape[0] < 2
            or not torch.isfinite(prompt_prototypes).all()
            or not torch.isfinite(description_prototypes).all()):
        raise ValueError('Relation loss requires matching finite [classes, dimensions] tensors')
    prompt = F.normalize(prompt_prototypes.float(), dim=-1)
    description = F.normalize(description_prototypes.detach().float(), dim=-1)
    prompt_relations = prompt @ prompt.T
    description_relations = description @ description.T
    off_diagonal = ~torch.eye(prompt.shape[0], dtype=torch.bool, device=prompt.device)
    return F.mse_loss(prompt_relations[off_diagonal], description_relations[off_diagonal])


class RelationPreservingCLIP(CustomCLIP):
    """Expose learned and frozen class prototypes only during training."""
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        weight = float(cfg.TRAINER.TEXTCONSISTENCY.RELATION_WEIGHT)
        self.relation_identity = {
            'format': 1,
            'kind': 'description_class_relation_mse',
            'formula': 'mean_offdiag((T@T.T-stopgrad(D@D.T))^2)',
            'weight': weight,
            'trainable_parameters_added': 0,
            'inference': 'unchanged_0.5*(S+M)',
        }

    def relation_prototypes(self):
        prompt = F.normalize(
            self.text_encoder(self.prompt_learner(), self.tokenized_prompts), dim=-1)
        views = self.multiview_encoder.get_view_features()
        mask = self.multiview_encoder.view_mask.to(dtype=views.dtype)
        description = (views * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return prompt, F.normalize(description, dim=-1).detach()


@TRAINER_REGISTRY.register()
class CoOp_TextConsistency_RelationPreserving(CoOp_TextConsistency_V2_FG):
    """Train shared prompts while preserving frozen description geometry."""
    model_builder = RelationPreservingCLIP

    def check_cfg(self, cfg):
        super().check_cfg(cfg)
        tc = cfg.TRAINER.TEXTCONSISTENCY
        if (not math.isfinite(float(tc.RELATION_WEIGHT)) or tc.RELATION_WEIGHT < 0
                or tc.AGGREGATION != 'mean' or tc.USE_FEATURE_ADAPTER
                or tc.USE_VISUAL_ADAPTER or tc.KD_WEIGHT != 0
                or tc.ADAPTIVE_CE_WEIGHT != 0 or tc.DIVERSITY_WEIGHT != 0
                or tc.CONTRASTIVE_WEIGHT != 0 or tc.INFERENCE_BRANCH != 'ensemble'
                or tc.ENSEMBLE_ALPHA != 0.5):
            raise ValueError('Relation preservation requires simple mean fusion and a finite '
                             'nonnegative relation weight with other auxiliary losses disabled')

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        student, teacher, _, attention = self.model(image, return_attn=True)
        base_loss, pieces = training_losses(
            student, teacher, label, attention, self.kd_weight,
            self.cfg.TRAINER.TEXTCONSISTENCY.KD_TEMP,
            self.adaptive_ce_weight, self.diversity_weight,
            kd_loss_mode=self.cfg.TRAINER.TEXTCONSISTENCY.KD_LOSS_MODE,
        )
        prompt, description = self.model.relation_prototypes()
        relation = category_relation_loss(prompt, description)
        loss = base_loss + self.cfg.TRAINER.TEXTCONSISTENCY.RELATION_WEIGHT * relation
        self.model_backward_and_update(loss)
        if self.batch_idx + 1 == self.num_batches:
            self.update_lr()
        return {
            'loss': loss.item(),
            **{name: value.item() for name, value in pieces.items()},
            'loss_relation': relation.item(),
            'acc': float(100. * (student.argmax(1) == label).float().mean().item()),
        }
