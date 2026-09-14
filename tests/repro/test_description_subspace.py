"""Focused checks for class-description subspace constrained residuals."""
import importlib.util
import unittest
from pathlib import Path
import tempfile

import torch

from trainers import coop_text_detail
from .test_text_residual import make_text

ROOT = Path(__file__).resolve().parents[2]


class DescriptionSubspaceTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def production(self, name):
        value = getattr(coop_text_detail, name, None)
        self.assertIsNotNone(value, f'missing production API: {name}')
        return value

    def study(self):
        path = ROOT / 'scripts/repro/description_subspace_study.py'
        self.assertTrue(path.is_file(), 'missing description-subspace study runner')
        spec = importlib.util.spec_from_file_location('description_subspace_study', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_masked_description_span_removes_unsupported_residual_directions(self):
        build = self.production('description_tangent_basis')
        project = self.production('project_onto_description_tangent')
        views = torch.tensor([
            [[1., 0., 0.], [-1., 0., 0.], [91., 92., 93.]],
            [[0., 2., 0.], [0., -2., 0.], [81., 82., 83.]],
        ])
        mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)

        basis, basis_mask = build(views, mask)
        delta = torch.tensor([[[3., 4., 5.], [3., 4., 5.]]], requires_grad=True)
        actual = project(delta, basis, basis_mask)

        expected = torch.tensor([[[3., 0., 0.], [0., 4., 0.]]])
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        self.assertEqual(basis_mask.sum(-1).tolist(), [1, 1])
        actual.sum().backward()
        torch.testing.assert_close(delta.grad, torch.tensor([[[1., 0., 0.], [0., 1., 0.]]]),
                                   rtol=0, atol=1e-6)

    def test_model_projects_then_bounds_without_adding_trainable_parameters(self):
        trainer_cls = self.production('CoOp_TextDetail_SubspaceRadius')
        constrained = make_text(self.root / 'constrained', detail='subspace_radius_attentive')
        radius_only = make_text(self.root / 'radius', detail='radius_bounded_attentive')
        self.assertIsInstance(constrained, trainer_cls)
        self.assertEqual(constrained.run_provenance['trainable_parameters'],
                         radius_only.run_provenance['trainable_parameters'])
        images = torch.randn(3, 3, 32, 32)
        initial = constrained.model(images)
        torch.testing.assert_close(initial['R'], torch.zeros_like(initial['R']), rtol=0, atol=0)
        torch.testing.assert_close(initial['Z'], initial['F'], rtol=0, atol=0)

        with torch.no_grad():
            constrained.model.text_residual.up.weight.fill_(50.)
        outputs = constrained.model(images)
        limits = constrained.model.logit_scale.exp() * constrained.model.description_radii
        self.assertTrue((outputs['R'].abs() <= limits[None] + 2e-5).all())
        ranks = constrained.model.description_basis_mask.sum(-1)
        valid_descriptions = constrained.model.multiview_encoder.view_mask.sum(-1)
        self.assertTrue((ranks <= valid_descriptions - 1).all())
        identity = constrained.model.correction_identity
        self.assertEqual(identity['kind'], 'description_subspace_radius_visual_text_detail')
        self.assertEqual(identity['operation_order'], 'project_then_radius_bound')
        self.assertEqual(identity['fitted_hyperparameters'], [])

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
        self.assertEqual(record['target_operation'], 'description_subspace_then_radius')
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
