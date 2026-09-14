"""Five focused checks for the description-radius bounded residual."""
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import tempfile
import unittest

import torch

from trainers import coop_text_detail
from .test_text_residual import make_text

ROOT = Path(__file__).resolve().parents[2]


class DescriptionRadiusTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.capture = contextlib.redirect_stdout(io.StringIO())
        self.capture.__enter__()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()
        self.capture.__exit__(None, None, None)

    def production(self, name):
        value = getattr(coop_text_detail, name, None)
        self.assertIsNotNone(value, f'missing production API: {name}')
        return value

    def study(self):
        path = ROOT / 'scripts/repro/description_radius_study.py'
        self.assertTrue(path.is_file(), 'missing description-radius study runner')
        spec = importlib.util.spec_from_file_location('description_radius_study', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_masked_rms_radius_ignores_padding(self):
        radius = self.production('description_radii')
        views = torch.tensor([
            [[1., 0.], [-1., 0.], [99., 99.]],
            [[0., 0.], [0., 2.], [0., 4.]],
        ])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        actual = radius(views, mask)
        expected = torch.tensor([1., math.sqrt(8. / 3.)])
        torch.testing.assert_close(actual, expected)

        changed = views.clone(); changed[0, 2] = -1000.
        torch.testing.assert_close(radius(changed, mask), expected)

    def test_smooth_bound_has_identity_gradient_at_zero_and_respects_radius(self):
        bound = self.production('bound_by_description_radius')
        delta = torch.tensor([[[3., 4.], [0., 0.], [5., 0.]]])
        radii = torch.tensor([2., 1., 0.])
        actual = bound(delta, radii)
        self.assertAlmostEqual(actual[0, 0].norm().item(), 2. * math.tanh(2.5), places=6)
        self.assertLessEqual(actual[0, 0].norm().item(), 2.)
        torch.testing.assert_close(actual[0, 2], torch.zeros(2))
        self.assertTrue(torch.isfinite(actual).all())

        zero = torch.zeros(1, 2, 3, requires_grad=True)
        bound(zero, torch.tensor([1., 2.])).sum().backward()
        torch.testing.assert_close(zero.grad, torch.ones_like(zero))

    def test_model_starts_at_fusion_and_bounds_large_residual_without_new_parameters(self):
        bounded_cls = self.production('CoOp_TextDetail_RadiusBounded')
        bounded = make_text(self.root / 'bounded', detail='radius_bounded_attentive')
        aligned = make_text(self.root / 'aligned', detail='fusion_aligned_attentive')
        self.assertIsInstance(bounded, bounded_cls)
        self.assertEqual(bounded.run_provenance['trainable_parameters'],
                         aligned.run_provenance['trainable_parameters'])
        images = torch.randn(3, 3, 32, 32)
        initial = bounded.model(images)
        torch.testing.assert_close(initial['R'], torch.zeros_like(initial['R']), rtol=0, atol=0)
        torch.testing.assert_close(initial['Z'], initial['F'], rtol=0, atol=0)

        with torch.no_grad():
            bounded.model.text_residual.up.weight.fill_(50.)
        outputs = bounded.model(images)
        limits = bounded.model.logit_scale.exp() * bounded.model.description_radii
        self.assertTrue((outputs['R'].abs() <= limits[None] + 2e-5).all())
        self.assertEqual(bounded.model.correction_identity['kind'],
                         'description_radius_bounded_visual_text_detail')
        self.assertEqual(bounded.model.correction_identity['fitted_hyperparameters'], [])

    def test_zero_start_update_reload_and_novel_radius_rebuild(self):
        bounded = make_text(self.root / 'bounded', detail='radius_bounded_attentive')
        aligned = make_text(self.root / 'aligned', detail='fusion_aligned_attentive')
        batch = {'img': torch.randn(3, 3, 32, 32), 'label': torch.tensor([0, 1, 2])}
        bounded.forward_backward(batch); aligned.forward_backward(batch)
        torch.testing.assert_close(bounded.model.prompt_learner.ctx,
                                   aligned.model.prompt_learner.ctx, rtol=0, atol=0)
        torch.testing.assert_close(bounded.model.text_residual.up.weight,
                                   aligned.model.text_residual.up.weight, rtol=0, atol=0)
        bounded.save_model(0, self.root / 'bounded')

        same = make_text(self.root / 'same', detail='radius_bounded_attentive')
        novel = make_text(self.root / 'novel', novel=True, detail='radius_bounded_attentive')
        novel_radius = novel.model.description_radii.clone()
        same.load_model(self.root / 'bounded', 1)
        novel.load_model(self.root / 'bounded', 1)
        for key, value in bounded.model(batch['img']).items():
            torch.testing.assert_close(value, same.model(batch['img'])[key], rtol=0, atol=0)
        torch.testing.assert_close(novel.model.description_radii, novel_radius, rtol=0, atol=0)
        self.assertEqual(tuple(novel_radius.shape), (4,))
        self.assertEqual(tuple(novel.model(batch['img'])['Z'].shape), (3, 4))
        with self.assertRaisesRegex(ValueError, 'signature mismatch'):
            aligned.load_model(self.root / 'bounded', 1)

    def test_legacy_reload_identity_and_zero_training_gate_are_fixed(self):
        study = self.study()
        signature = {
            'trainer': 'CoOp_TextDetail_FusionAligned',
            'correction': {
                'version': 4,
                'kind': 'fusion_aligned_visual_text_detail',
                'inference': 'F+R',
                'objective': 'CE(S,y)+CE(stopgrad(F)+R,y)',
                'fusion_anchor': 'F=0.5*(S+M)',
                'coefficient': 1.0,
            },
        }
        checkpoint = {
            'epoch': 20,
            'repro_signature': signature,
            'run_provenance': {
                'seed': 1, 'dataset': 'DescribableTextures', 'shots': 16,
                'subsample': 'base', 'trainable_parameters': 67584,
                'model_signature': signature,
            },
        }
        record = study.validate_legacy_checkpoint(
            checkpoint, study.LEGACY_CHECKPOINT_SHA256)
        self.assertEqual(record['checkpoint_sha256'], study.LEGACY_CHECKPOINT_SHA256)
        self.assertTrue(study.zero_training_decision(0.01, -1.0)['passes'])
        self.assertFalse(study.zero_training_decision(0.0, -1.0)['passes'])
        self.assertFalse(study.zero_training_decision(1.0, -1.01)['passes'])

        checkpoint['repro_signature']['correction']['version'] = 3
        with self.assertRaisesRegex(ValueError, 'legacy checkpoint identity'):
            study.validate_legacy_checkpoint(checkpoint, study.LEGACY_CHECKPOINT_SHA256)
        with self.assertRaisesRegex(ValueError, 'checkpoint SHA256'):
            study.validate_legacy_checkpoint(checkpoint, '0' * 64)


if __name__ == '__main__':
    unittest.main()
