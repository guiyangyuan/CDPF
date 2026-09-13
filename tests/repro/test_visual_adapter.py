"""Behavior tests for a shared adapter in the classification feature path.

Tiny real CLIP, synthetic images; these check software, not method accuracy.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from dassl.config import get_cfg_default
from dassl.engine.trainer import TrainerBase
import train
from trainers.coop_textconsistency_v2_fg import CoOp_TextConsistency_V2_FG
from trainers.textconsistency_common import training_losses
from .test_trainers import tiny_clip

ROOT = Path(__file__).resolve().parents[2]


def make_visual_trainer(output, enabled=True, desc_weight=1., class_slice=slice(0, 3),
                        subsample='base', **overrides):
    cfg = get_cfg_default(); train.extend_cfg(cfg)
    cfg.SEED = 1; cfg.USE_CUDA = False
    cfg.DATASET.ROOT = str(output); cfg.OUTPUT_DIR = str(output)
    cfg.DATASET.NAME = 'Caltech101'; cfg.DATASET.NUM_SHOTS = 16
    cfg.DATASET.SUBSAMPLE_CLASSES = subsample
    cfg.TRAINER.NAME = 'CoOp_TextConsistency_V2_FG'
    cfg.MODEL.BACKBONE.NAME = 'ViT-B/16'
    cfg.OPTIM.MAX_EPOCH = 20; cfg.OPTIM.WARMUP_EPOCH = 0
    tc = cfg.TRAINER.TEXTCONSISTENCY
    tc.DEVICE = 'cpu'; tc.AGGREGATION = 'mean'; tc.USE_FEATURE_ADAPTER = False
    tc.KD_WEIGHT = 0.; tc.ADAPTIVE_CE_WEIGHT = desc_weight
    tc.N_CTX = 4; tc.N_VIEWS = 5
    tc.USE_VISUAL_ADAPTER = enabled; tc.VISUAL_ADAPTER_DIM = 8; tc.VISUAL_ADAPTER_SCALE = .2
    for name, value in overrides.items(): setattr(tc, name, value)
    trainer = CoOp_TextConsistency_V2_FG.__new__(CoOp_TextConsistency_V2_FG)
    TrainerBase.__init__(trainer); trainer.cfg = cfg
    names = list(json.loads((ROOT / 'templates/cal_prompts_full.json').read_text()))[class_slice]
    trainer.dm = SimpleNamespace(dataset=SimpleNamespace(classnames=names, train_x=[], val=[], test=[]))
    trainer.check_cfg(cfg)
    with patch('trainers.textconsistency_runtime.load_clip_checkpoint', side_effect=tiny_clip):
        trainer.build_model()
    trainer.batch_idx = 0; trainer.num_batches = 2
    return trainer


class VisualAdapterBehavior(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.capture = contextlib.redirect_stdout(io.StringIO()); self.capture.__enter__()
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup(); self.capture.__exit__(None, None, None)

    def adapter(self, trainer):
        module = getattr(trainer.model, 'visual_adapter', None)
        self.assertIsNotNone(module, 'Enabled visual adaptation must be present in the model')
        return module

    def test_identity_start_preserves_logits_prompt_initialization_and_sampling_rng(self):
        a = make_visual_trainer(self.root/'a', enabled=False, desc_weight=0.)
        state_a = torch.random.get_rng_state().clone()
        b = make_visual_trainer(self.root/'b', desc_weight=0.)
        state_b = torch.random.get_rng_state().clone()
        c = make_visual_trainer(self.root/'c', desc_weight=1.)
        self.adapter(b); self.adapter(c)
        self.assertTrue(torch.equal(state_a, state_b))
        self.assertTrue(torch.equal(state_b, torch.random.get_rng_state()))
        images = torch.randn(3, 3, 32, 32)
        for model in (b.model, c.model):
            torch.testing.assert_close(a.model.prompt_learner.ctx, model.prompt_learner.ctx, rtol=0, atol=0)
            for original, adapted in zip(a.model(images)[:3], model(images)[:3]):
                torch.testing.assert_close(original, adapted, rtol=1e-6, atol=2e-6)

    def test_actual_student_and_description_logits_use_adapted_features(self):
        trainer = make_visual_trainer(self.root/'model')
        model = trainer.model; adapter = self.adapter(trainer)
        images = torch.randn(3, 3, 32, 32)
        old_s, old_t, raw, _ = model(images)
        with torch.no_grad():
            adapter.down.weight.fill_(.2)
            adapter.up.weight.zero_(); adapter.up.weight[0].fill_(1.)
        # Guarantee an observable nonzero residual independently of CLIP's random output sign.
        with torch.no_grad(): adapter.down.weight.copy_(raw[0].expand_as(adapter.down.weight))
        student, description, used, _ = model(images)
        self.assertFalse(torch.allclose(raw, used))
        self.assertFalse(torch.allclose(old_s, student))
        self.assertFalse(torch.allclose(old_t, description))
        torch.testing.assert_close(used.norm(dim=-1), torch.ones(3))
        text = F.normalize(model.text_encoder(model.prompt_learner(), model.tokenized_prompts), dim=-1)
        views = model.multiview_encoder.get_view_features()
        mask = model.multiview_encoder.view_mask
        mean = F.normalize((views * mask[..., None]).sum(1) / mask.sum(1)[:, None], dim=-1)
        torch.testing.assert_close(student, model.logit_scale.exp() * used @ text.T)
        torch.testing.assert_close(description, model.logit_scale.exp() * used @ mean.T)
        torch.testing.assert_close(raw, F.normalize(model.image_encoder(images), dim=-1))

    def test_description_loss_has_adapter_gradient_and_no_prompt_or_clip_gradient(self):
        trainer = make_visual_trainer(self.root/'model')
        adapter = self.adapter(trainer)
        images = torch.randn(3, 3, 32, 32); labels = torch.tensor([0, 1, 2])
        student, description, _, _ = trainer.model(images)
        desc_loss = F.cross_entropy(description, labels)
        desc_loss.backward()
        self.assertGreater(adapter.up.weight.grad.abs().sum().item(), 0.)
        self.assertIsNone(trainer.model.prompt_learner.ctx.grad)
        for module in (trainer.model.image_encoder, trainer.model.text_encoder):
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in module.parameters()))
        trainer.optim.zero_grad(set_to_none=True)
        student, description, _, _ = trainer.model(images)
        total, _ = training_losses(student, description, labels, {}, 0., 2., 0., 0.)
        expected = torch.autograd.grad(F.cross_entropy(student, labels), adapter.up.weight, retain_graph=True)[0]
        actual = torch.autograd.grad(total, adapter.up.weight)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_optimizer_updates_both_adapter_layers_and_keeps_clip_frozen(self):
        trainer = make_visual_trainer(self.root/'model'); adapter = self.adapter(trainer)
        before = {k: p.detach().clone() for k, p in trainer.model.named_parameters()}
        ids = {id(p) for group in trainer.optim.param_groups for p in group['params']}
        self.assertTrue(all(id(p) in ids for p in adapter.parameters()))
        batch = {'img': torch.randn(3, 3, 32, 32), 'label': torch.tensor([0, 1, 2])}
        trainer.forward_backward(batch)
        trainer.optim.zero_grad(set_to_none=True)
        student, description, _, _ = trainer.model(batch['img'])
        (F.cross_entropy(student, batch['label']) + F.cross_entropy(description, batch['label'])).backward()
        self.assertGreater(adapter.down.weight.grad.abs().sum().item(), 0.)
        trainer.optim.zero_grad(set_to_none=True)
        trainer.forward_backward(batch)
        for name in ('visual_adapter.down.weight', 'visual_adapter.up.weight', 'prompt_learner.ctx'):
            self.assertFalse(torch.equal(before[name], dict(trainer.model.named_parameters())[name]), name)
        for name, param in trainer.model.named_parameters():
            if not name.startswith(('visual_adapter.', 'prompt_learner.', 'view_aggregator.')):
                torch.testing.assert_close(before[name], param, rtol=0, atol=0)
                self.assertIsNone(param.grad)

    def test_description_supervision_changes_visual_update_at_matched_initialization(self):
        b = make_visual_trainer(self.root/'b', desc_weight=0.)
        c = make_visual_trainer(self.root/'c', desc_weight=1.)
        b_adapter = self.adapter(b); c_adapter = self.adapter(c)
        self.assertEqual(b.run_provenance['trainable_parameters'], c.run_provenance['trainable_parameters'])
        batch = {'img': torch.randn(3, 3, 32, 32), 'label': torch.tensor([0, 1, 2])}
        b.forward_backward(batch); c.forward_backward(batch)
        self.assertFalse(torch.equal(b_adapter.up.weight, c_adapter.up.weight))
        # The extra description loss has no prompt path on the shared initial step.
        torch.testing.assert_close(b.model.prompt_learner.ctx, c.model.prompt_learner.ctx, rtol=0, atol=0)

    def test_complete_checkpoint_reloads_base_and_different_novel_class_count(self):
        first = make_visual_trainer(self.root/'base')
        self.adapter(first)
        images = torch.randn(3, 3, 32, 32)
        first.forward_backward({'img': images, 'label': torch.tensor([0, 1, 2])})
        first.save_model(0, self.root/'base')
        same = make_visual_trainer(self.root/'same')
        novel = make_visual_trainer(self.root/'novel', class_slice=slice(5, 9), subsample='new')
        rebuilt = novel.model.prompt_learner.token_suffix.clone()
        for target in (same, novel):
            target.load_model(self.root/'base', 1)
            for name, param in self.adapter(first).named_parameters():
                torch.testing.assert_close(param, dict(self.adapter(target).named_parameters())[name], rtol=0, atol=0)
        torch.testing.assert_close(novel.model.prompt_learner.token_suffix, rebuilt, rtol=0, atol=0)
        for branch in ('student', 'teacher', 'ensemble'):
            first.cfg.TRAINER.TEXTCONSISTENCY.INFERENCE_BRANCH = branch
            same.cfg.TRAINER.TEXTCONSISTENCY.INFERENCE_BRANCH = branch
            novel.cfg.TRAINER.TEXTCONSISTENCY.INFERENCE_BRANCH = branch
            torch.testing.assert_close(first.model_inference(images), same.model_inference(images), rtol=0, atol=0)
            self.assertEqual(tuple(novel.model_inference(images).shape), (3, 4))
        self.assertEqual(first.repro_signature['format'], 3)
        identity = first.repro_signature['visual_adapter']
        self.assertEqual(identity['hidden_dim'], 8)
        self.assertEqual(identity['scale'], .2)
        self.assertEqual(identity['description_ce_weight'], 1.)
        manifest = json.loads((self.root/'base/run_manifest.json').read_text())
        self.assertEqual(manifest['model_signature']['visual_adapter'], identity)

    def test_mismatched_or_incomplete_adapter_checkpoint_rejects_before_mutating(self):
        first = make_visual_trainer(self.root/'base'); self.adapter(first)
        first.forward_backward({'img': torch.randn(3, 3, 32, 32), 'label': torch.tensor([0, 1, 2])})
        first.save_model(0, self.root/'base')
        path = self.root/'base/textconsistency/model.pth.tar-1'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        for change in ('missing_tensor', 'remove_identity', 'different_scale', 'different_loss'):
            with self.subTest(change=change):
                checkpoint = copy.deepcopy(saved)
                options = {}
                if change == 'missing_tensor': checkpoint['state_dict'].pop('visual_adapter.up.weight')
                if change == 'remove_identity': checkpoint['repro_signature'].pop('visual_adapter')
                if change == 'different_scale': options['VISUAL_ADAPTER_SCALE'] = .3
                if change == 'different_loss': options['ADAPTIVE_CE_WEIGHT'] = 0.
                torch.save(checkpoint, path)
                target = make_visual_trainer(self.root/change, **options)
                before = {k: v.clone() for k, v in target.trainable_modules.state_dict().items()}
                with self.assertRaises(ValueError): target.load_model(self.root/'base', 1)
                for name, tensor in target.trainable_modules.state_dict().items():
                    torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)

    def test_disabled_adapter_keeps_legacy_signature_and_optimizer_structure(self):
        trainer = make_visual_trainer(self.root/'legacy', enabled=False, desc_weight=0.)
        self.assertIsNone(getattr(trainer.model, 'visual_adapter', None))
        self.assertEqual(trainer.repro_signature['format'], 2)
        self.assertNotIn('visual_adapter', trainer.repro_signature)
        self.assertEqual(set(trainer.trainable_modules), {'prompt_learner', 'view_aggregator'})

    def test_invalid_or_incompatible_active_adapter_config_is_rejected(self):
        for field, values in {
            'VISUAL_ADAPTER_DIM': (0, -1, 2.5, True),
            'VISUAL_ADAPTER_SCALE': (0., -1., float('nan'), float('inf'), 1.1),
            'AGGREGATION': ('hierarchical', 'flat'),
            'USE_FEATURE_ADAPTER': (True,),
            'KD_WEIGHT': (1.,),
            'KD_LOSS_MODE': ('weighted_nckd',),
            'DIVERSITY_WEIGHT': (1.,),
            'ABLATION_GROUP': ('baseline',),
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    make_visual_trainer(self.root/'invalid', **{field: value})

    def test_historical_export_helpers_reject_even_identity_initialized_adapter(self):
        from scripts.repro import validation_probe as probe
        trainer = make_visual_trainer(self.root/'model')
        self.adapter(trainer)
        # A historical loader omits the new module. Reject even a crafted old-shaped state.
        historical = {'state_dict': {k: v for k, v in trainer.trainable_modules.state_dict().items()
                                    if not k.startswith('visual_adapter.')}}
        with self.assertRaisesRegex(ValueError, 'visual|Visual'):
            probe.strict_load_trainable_modules(trainer.model, historical)
        models = {role: trainer.model for role in probe.ROLES}
        with self.assertRaisesRegex(ValueError, 'visual|Visual'):
            probe.export_batches(models, [], torch.randn(3, 5, 32), torch.ones(3, 5).bool(),
                                 trainer.dm.dataset.classnames, device='cpu')


    def test_historical_h1_export_rejects_before_mutating_model_mode_or_gradients(self):
        from scripts.repro import h1_validation as h1
        trainer = make_visual_trainer(self.root/'model')
        model = trainer.model
        prior_mode = model.training
        prior_flags = {name: p.requires_grad for name, p in model.named_parameters()}
        with self.assertRaisesRegex(ValueError, 'visual|Visual'):
            h1.export_batches(model, [], model.multiview_encoder.get_view_features(),
                              model.multiview_encoder.view_mask, trainer.dm.dataset.classnames,
                              device='cpu')
        self.assertEqual(prior_mode, model.training)
        self.assertEqual(prior_flags, {name: p.requires_grad for name, p in model.named_parameters()})


    def test_main_rejects_visual_flag_for_trainers_that_cannot_use_it(self):
        for name in ('CoOp', 'CoCoOp_TextConsistency_V2_FG', 'CoOp_TextConsistency_Ablation'):
            cfg = get_cfg_default(); train.extend_cfg(cfg)
            cfg.TRAINER.NAME = name
            cfg.OUTPUT_DIR = str(self.root/name)
            cfg.USE_CUDA = False
            cfg.TRAINER.TEXTCONSISTENCY.USE_VISUAL_ADAPTER = True
            args = SimpleNamespace(eval_only=False, no_train=True, model_dir='', load_epoch=None)
            with self.subTest(name=name), patch('train.setup_cfg', return_value=cfg), \
                    patch('train.setup_logger'), patch('train.build_trainer', side_effect=AssertionError('Invalid visual flag reached dataset/model construction')):
                with self.assertRaisesRegex(ValueError, 'visual|Visual'):
                    train.main(args)


if __name__ == '__main__': unittest.main()
