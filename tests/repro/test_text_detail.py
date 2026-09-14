"""Two small checks for the changed pooling path and checkpoint/class handling."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F
from trainers.coop_text_detail import description_attention, description_weights, anchored_delta
from trainers.coop_text_residual import SharedTextResidual
from scripts.repro.text_detail_pilot import confirmation_decision
from .test_text_residual import make_text


class TextDetailSmoke(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.capture = contextlib.redirect_stdout(io.StringIO()); self.capture.__enter__()
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup(); self.capture.__exit__(None, None, None)

    def test_centered_limit_mask_and_image_dependence(self):
        torch.manual_seed(12)
        head = SharedTextResidual(8, 4, .2)
        with torch.no_grad(): head.up.weight.normal_(std=.1)
        names = F.normalize(torch.randn(3, 8), dim=-1)
        views = F.normalize(torch.randn(3, 5, 8), dim=-1)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
        uniform = description_attention(torch.zeros(2, 8), views, mask, 20.)
        delta = anchored_delta(head, names, views, mask, uniform, True)
        expected = head(names[:, None], torch.ones(3, 1, dtype=torch.bool))
        torch.testing.assert_close(delta, expected[None].expand_as(delta), rtol=0, atol=0)
        images = F.normalize(torch.randn(2, 8), dim=-1)
        attention = description_attention(images, views, mask, 20.)
        self.assertTrue((attention[:, ~mask] == 0).all())
        actual = anchored_delta(head, names, views, mask, attention, True)
        self.assertGreater((actual[0] - actual[1]).abs().max().item(), 1e-6)
        changed = views.clone(); changed[~mask] = 100
        torch.testing.assert_close(actual, anchored_delta(head, names, changed, mask,
            description_attention(images, changed, mask, 20.), True), rtol=0, atol=0)
        perm = torch.tensor([2, 0, 1])
        torch.testing.assert_close(actual[:, perm], anchored_delta(head, names[perm], views[perm], mask[perm],
            description_attention(images, views[perm], mask[perm], 20.), True))
        single = anchored_delta(head, names, views, mask, description_attention(images[:1], views, mask, 20.), True)
        torch.testing.assert_close(single, actual[:1])

    def test_uniform_description_control_is_image_independent(self):
        torch.manual_seed(21)
        images = F.normalize(torch.randn(3, 8), dim=-1)
        views = F.normalize(torch.randn(2, 4, 8), dim=-1)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
        weights = description_weights(images, views, mask, 20., 'uniform')
        expected = mask.float() / mask.sum(-1, keepdim=True)
        torch.testing.assert_close(weights, expected[None].expand_as(weights), rtol=0, atol=0)
        changed = F.normalize(torch.randn(3, 8), dim=-1)
        torch.testing.assert_close(weights, description_weights(changed, views, mask, 20., 'uniform'), rtol=0, atol=0)
        self.assertTrue((weights[:, ~mask] == 0).all())

    def test_confirmation_rule_is_fixed_before_scores(self):
        passed = confirmation_decision([2.5, 1.5, -0.5], 0.25)
        self.assertTrue(passed['passes'])
        self.assertEqual(passed['positive_seeds'], 2)
        self.assertFalse(confirmation_decision([2.0, -0.5, -0.5], 0.5)['passes'])
        self.assertFalse(confirmation_decision([2.5, 1.5, -0.5], -0.01)['passes'])

    def test_two_updates_reload_and_new_classes(self):
        baseline = make_text(self.root / 'baseline', 'classname', adamw=True)
        uniform = make_text(self.root / 'uniform', detail='uniform')
        raw = make_text(self.root / 'raw', detail='attentive')
        first = make_text(self.root / 'first', detail='centered')
        batch = {'img': torch.randn(3, 3, 32, 32), 'label': torch.tensor([0, 1, 2])}
        for t in (uniform, raw, first):
            self.assertEqual(t.run_provenance['trainable_parameters'], baseline.run_provenance['trainable_parameters'])
            torch.testing.assert_close(t.model(batch['img'])['Z'], baseline.model(batch['img'])['S'], rtol=0, atol=0)
            correction = t.loss_terms(t.model(batch['img']), batch['label'])[1]
            self.assertIsNone(torch.autograd.grad(correction, t.model.prompt_learner.ctx, allow_unused=True)[0])
        for t in (baseline, uniform, raw, first): t.forward_backward(batch)
        first.save_model(0, self.root / 'first')
        saved = torch.load(self.root / 'first/textconsistency/model.pth.tar-1', weights_only=False)
        same = make_text(self.root / 'same', detail='centered')
        novel = make_text(self.root / 'novel', novel=True, detail='centered')
        novel_names = novel.model.detail_names.clone()
        for t in (same, novel): t.load_model(self.root / 'first', 1)
        same.optim.load_state_dict(saved['optimizer']); same.sched.load_state_dict(saved['scheduler'])
        for key, value in first.model(batch['img']).items():
            torch.testing.assert_close(value, same.model(batch['img'])[key], rtol=0, atol=0)
        torch.testing.assert_close(novel_names, novel.model.detail_names, rtol=0, atol=0)
        self.assertEqual(novel.model(batch['img'])['Z'].shape, (3, 4))
        with self.assertRaisesRegex(ValueError, 'signature mismatch'): raw.load_model(self.root / 'first', 1)
        for t in (baseline, uniform, raw, first, same): t.forward_backward(batch)
        for t in (uniform, raw, first, same):
            torch.testing.assert_close(t.model.prompt_learner.ctx, baseline.model.prompt_learner.ctx, rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in t.model.image_encoder.parameters()))
            self.assertGreater(t.model.text_residual.up.weight.abs().max().item(), 0)
            self.assertEqual(t.optim.residual.param_groups[0]['lr'], .001)
        for key, value in first.trainable_modules.state_dict().items():
            torch.testing.assert_close(value, same.trainable_modules.state_dict()[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
