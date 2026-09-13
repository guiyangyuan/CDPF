"""Protected CoOp training with a class-shared, text-directed correction.

S uses the original frozen image feature. Only the correction branch uses the
adapter. Its loss is CE(stopgrad(S) + Da - D0), not an independent descriptor CE.
"""
import torch
from torch import nn
from torch.nn import functional as F
from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from clip import clip

from .coop_textconsistency_v2_fg import PromptLearner, TextEncoder
from .textconsistency_common import (
    CachedMultiViewTextEncoder, build_view_aggregator, load_complete_checkpoint,
    select_device,
)
from .textconsistency_runtime import ValidatedTrainer
from .visual_adapter import ResidualVisualAdapter
from .zsclip import CUSTOM_TEMPLATES


class CorrectionAdapter(ResidualVisualAdapter):
    def identity(self, description_ce_weight):
        identity = super().identity(description_ce_weight)
        identity['placement'] = 'correction_only_student_uses_original_feature'
        return identity


class CorrectionCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        tc = cfg.TRAINER.TEXTCONSISTENCY
        # Keep the original CoOp construction order and adapter RNG isolation.
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.multiview_encoder = CachedMultiViewTextEncoder(cfg, classnames, clip_model)
        self.view_aggregator = build_view_aggregator(cfg, clip_model.text_projection.shape[-1])
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.visual_adapter = (CorrectionAdapter(clip_model.text_projection.shape[-1],
                               tc.VISUAL_ADAPTER_DIM, tc.VISUAL_ADAPTER_SCALE)
                               if tc.USE_VISUAL_ADAPTER else None)
        views = self.multiview_encoder.get_view_features()
        mask = self.multiview_encoder.view_mask
        description = F.normalize((views * mask[..., None]).sum(1) / mask.sum(1)[:, None], dim=-1)
        template = None
        if tc.CORRECTION_TEXT == 'classname':
            template = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
            prompts = [template.format(name.replace('_', ' ')) for name in classnames]
            device, original = select_device(cfg), next(clip_model.parameters()).device
            clip_model.to(device).eval()
            with torch.no_grad():
                text = torch.cat([F.normalize(clip_model.encode_text(chunk.to(device)).float(), dim=-1).cpu()
                                  for chunk in clip.tokenize(prompts).split(tc.TEXT_BATCH_SIZE)])
            clip_model.to(original)
        else:
            text = description
        # Rebuilt from the current class list at load time, never learned per class.
        self.register_buffer('correction_text', text, persistent=False)
        self.correction_identity = {
            'version': 1, 'enabled': self.visual_adapter is not None,
            'text_source': tc.CORRECTION_TEXT, 'classname_template': template,
            'residual': 'Da-D0', 'inference': 'S+R', 'coefficient': 1.0,
            'objective': 'CE(S,y)+CE(stopgrad(S)+R,y)' if self.visual_adapter is not None else 'CE(S,y)',
            'student_feature': 'original_unit_clip_image',
            'reference_feature': 'renormalized_unit_clip_image_for_exact_zero_residual',
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
        # Adapter normalizes its output; apply the same normalization to the
        # reference so its zero-up initialization gives an exactly zero R.
        reference = F.normalize(feature, dim=-1)
        adapted = self.visual_adapter(feature) if self.visual_adapter is not None else reference
        d0 = scale * reference @ self.correction_text.T
        da = scale * adapted @ self.correction_text.T
        residual = da - d0
        return {'S': student, 'M': mean, 'D0': d0, 'Da': da, 'R': residual,
                'Z': student + residual, 'F': 0.5 * student + 0.5 * mean}


@TRAINER_REGISTRY.register()
class CoOp_TextConsistency_Correction(ValidatedTrainer):
    model_builder = CorrectionCLIP
    BRANCHES = {'student': 'S', 'description': 'M', 'corrected': 'Z', 'simple_fusion': 'F'}

    def check_cfg(self, cfg):
        super().check_cfg(cfg)
        tc = cfg.TRAINER.TEXTCONSISTENCY
        if tc.CORRECTION_TEXT not in ('classname', 'description'):
            raise ValueError('CORRECTION_TEXT must be classname or description')
        if (tc.AGGREGATION != 'mean' or tc.USE_FEATURE_ADAPTER or tc.ABLATION_GROUP != 'full'
                or tc.KD_WEIGHT != 0 or tc.ADAPTIVE_CE_WEIGHT != 0
                or tc.DIVERSITY_WEIGHT != 0 or tc.CONTRASTIVE_WEIGHT != 0
                or tc.KD_LOSS_MODE != 'full_kl' or tc.VIEW_ORDER != 'contiguous'):
            raise ValueError('Correction requires fixed mean text, no KD/query adapter/independent description CE')
        if tc.INFERENCE_BRANCH not in self.BRANCHES or tc.ENSEMBLE_ALPHA != 0.5:
            raise ValueError('Select an explicit correction inference branch; simple fusion has fixed alpha=0.5')
        if not tc.USE_VISUAL_ADAPTER and tc.CORRECTION_TEXT != 'description':
            raise ValueError('The no-adapter control uses description text and original CoOp CE')

    def loss_terms(self, outputs, labels):
        student = F.cross_entropy(outputs['S'], labels)
        correction = (F.cross_entropy(outputs['S'].detach() + outputs['R'], labels)
                      if self.model.visual_adapter is not None else student.new_zeros(()))
        return student, correction

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        outputs = self.model(image)
        student, correction = self.loss_terms(outputs, label)
        loss = student + correction
        self.model_backward_and_update(loss)
        if self.batch_idx + 1 == self.num_batches:
            self.update_lr()
        return {'loss': loss.item(), 'loss_student': student.item(),
                'loss_correction': correction.item(),
                'acc': compute_accuracy(outputs['Z'], label)[0].item()}

    def model_inference(self, image):
        return self.model(image)[self.BRANCHES[self.cfg.TRAINER.TEXTCONSISTENCY.INFERENCE_BRANCH]]

    def load_model(self, directory, epoch=None):
        return load_complete_checkpoint(self, directory, epoch)
