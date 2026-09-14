"""Focused checks for the parameter-free semantic residual gate."""
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from trainers import coop_text_detail
from .test_text_residual import make_text

ROOT = Path(__file__).resolve().parents[2]


class SemanticAlignmentGateTest(unittest.TestCase):
    def test_gate_uses_centered_semantic_alignment_and_handles_degenerate_rows(self):
        gate = getattr(coop_text_detail, 'semantic_alignment_gate', None)
        self.assertIsNotNone(gate, 'missing production API: semantic_alignment_gate')

        student = torch.tensor([
            [3., 1., 0.],
            [3., 1., 0.],
            [3., 1., 0.],
            [3., 1., 0.],
        ])
        description = torch.tensor([
            [5., 1., -2.],
            [5., 1., -2.],
            [5., 1., -2.],
            [3., 1., 0.],
        ])
        semantic_direction = description - student
        residual = torch.stack((
            semantic_direction[0],
            -semantic_direction[1],
            torch.tensor([1., 1., 1.]),
            torch.tensor([2., -1., -1.]),
        ))

        actual = gate(student, description, residual)
        torch.testing.assert_close(actual, torch.tensor([1., 0., 0., 0.]), rtol=0, atol=1e-6)
        shifted = gate(student + 91., description - 37., residual + 12.)
        torch.testing.assert_close(shifted, actual, rtol=0, atol=1e-6)
        self.assertTrue(torch.isfinite(actual).all())

    def test_model_gates_radius_bounded_residual_without_new_parameters(self):
        trainer_cls = getattr(coop_text_detail, 'CoOp_TextDetail_SemanticGatedRadius', None)
        self.assertIsNotNone(
            trainer_cls, 'missing production API: CoOp_TextDetail_SemanticGatedRadius')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            gated = make_text(root / 'gated', detail='semantic_gated_radius_attentive')
            radius = make_text(root / 'radius', detail='radius_bounded_attentive')
            self.assertIsInstance(gated, trainer_cls)
            self.assertEqual(gated.run_provenance['trainable_parameters'],
                             radius.run_provenance['trainable_parameters'])

            images = torch.randn(3, 3, 32, 32)
            initial = gated.model(images)
            torch.testing.assert_close(initial['R'], torch.zeros_like(initial['R']), rtol=0, atol=0)
            torch.testing.assert_close(initial['Q'], torch.zeros_like(initial['Q']), rtol=0, atol=0)
            torch.testing.assert_close(initial['Z'], initial['F'], rtol=0, atol=0)

            with torch.no_grad():
                gated.model.text_residual.up.weight.fill_(50.)
            outputs = gated.model(images)
            expected_gate = coop_text_detail.semantic_alignment_gate(
                outputs['S'], outputs['M'], outputs['R']).detach()
            torch.testing.assert_close(outputs['Q'], expected_gate)
            torch.testing.assert_close(outputs['Z'], outputs['F'] + outputs['Q'][:, None] * outputs['R'])
            self.assertFalse(outputs['Q'].requires_grad)
            self.assertTrue(((0. <= outputs['Q']) & (outputs['Q'] <= 1.)).all())

            labels = torch.tensor([0, 1, 2])
            student_loss, correction_loss = gated.loss_terms(outputs, labels)
            torch.testing.assert_close(student_loss, F.cross_entropy(outputs['S'], labels))
            torch.testing.assert_close(
                correction_loss,
                F.cross_entropy(outputs['F'].detach() + outputs['R'], labels))
            self.assertIsNone(torch.autograd.grad(
                correction_loss, gated.model.prompt_learner.ctx,
                allow_unused=True, retain_graph=True)[0])
            identity = gated.model.correction_identity
            self.assertEqual(identity['kind'], 'semantic_alignment_gated_radius_text_detail')
            self.assertEqual(identity['fitted_hyperparameters'], [])

            fresh = make_text(root / 'fresh', detail='semantic_gated_radius_attentive')
            self.assertEqual(torch.count_nonzero(fresh.model.text_residual.up.weight).item(), 0)
            fresh.forward_backward({'img': images, 'label': labels})
            self.assertGreater(torch.count_nonzero(fresh.model.text_residual.up.weight).item(), 0)
            self.assertEqual(
                fresh.model.correction_identity['training_objective'],
                'CE(S,y)+CE(stopgrad(F)+R_hat,y)')

    def test_offline_study_freezes_formula_and_stopping_rule(self):
        path = ROOT / 'scripts/repro/semantic_alignment_gate_study.py'
        self.assertTrue(path.is_file(), 'missing semantic-alignment study runner')
        spec = importlib.util.spec_from_file_location('semantic_alignment_gate_study', path)
        study = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(study)

        self.assertEqual(study.FORMULA, {
            'simple_fusion': 'F=0.5*(S+M)',
            'semantic_direction': 'D=center(M-S)',
            'residual_direction': 'R0=center(R_hat)',
            'gate': 'q=clamp(cos(R0,D),0,1)',
            'corrected': 'G=F+q*R_hat',
            'fitted_hyperparameters': [],
        })
        self.assertTrue(study.development_decision(0.01, -1.0)['passes'])
        self.assertFalse(study.development_decision(0.0, -1.0)['passes'])
        self.assertFalse(study.development_decision(1.0, -1.01)['passes'])

        student = np.array([[3., 1., 0.], [3., 1., 0.]], dtype=np.float32)
        description = np.array([[5., 1., -2.], [5., 1., -2.]], dtype=np.float32)
        residual = np.stack((description[0] - student[0], student[1] - description[1]))
        computed = study.compute_gated_logits(student, description, residual)
        np.testing.assert_allclose(computed['Q'], np.array([1., 0.], dtype=np.float32),
                                   rtol=0, atol=1e-6)
        np.testing.assert_array_equal(computed['F'], .5 * (student + description))
        np.testing.assert_allclose(computed['G'][0], computed['F'][0] + residual[0],
                                   rtol=0, atol=1e-6)
        np.testing.assert_allclose(computed['G'][1], computed['F'][1],
                                   rtol=0, atol=1e-6)

    def test_offline_analysis_authenticates_exports_and_reports_transitions(self):
        path = ROOT / 'scripts/repro/semantic_alignment_gate_study.py'
        spec = importlib.util.spec_from_file_location('semantic_alignment_gate_analysis', path)
        study = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(study)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, labels in (('基础测试集', np.array([0, 1], dtype=np.int64)),
                                 ('新类别测试集', np.array([0, 1], dtype=np.int64))):
                target = root / name
                target.mkdir()
                student = np.array([[2., 0.], [1., 2.]], dtype=np.float32)
                description = np.array([[3., -1.], [0., 3.]], dtype=np.float32)
                residual = np.array([[1., -1.], [-1., 1.]], dtype=np.float32)
                fusion = .5 * (student + description)
                archive = target / '预测.npz'
                np.savez_compressed(archive, labels=labels, S=student, M=description,
                                    R=residual, F=fusion, Z=fusion + residual)
                digest = hashlib.sha256(archive.read_bytes()).hexdigest()
                (target / '导出.json').write_text(json.dumps({
                    'predictions_sha256': digest,
                    'rows': [{'sample_id': str(i), 'label': int(label)}
                             for i, label in enumerate(labels)],
                    'class_order': ['a', 'b'],
                }))

            analysis = study.analyze_exports(root / '基础测试集', root / '新类别测试集')
            self.assertEqual(analysis['new_trainings'], 0)
            self.assertEqual(analysis['formula'], study.FORMULA)
            self.assertEqual(analysis['splits']['base']['total'], 2)
            self.assertIn('corrections_vs_F', analysis['splits']['novel'])
            self.assertIn('decision', analysis)

            protocol = study.build_protocol(
                root / '基础测试集', root / '新类别测试集', root / '输出', 'test-commit')
            protocol_path = root / '实验协议.json'
            protocol_path.write_text(json.dumps(protocol, ensure_ascii=False))
            completed = subprocess.run(
                [sys.executable, str(path), 'run', '--protocol', str(protocol_path)],
                cwd=ROOT, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            status = json.loads((root / '输出/执行状态.json').read_text())
            self.assertEqual(status['new_trainings_completed'], 0)

            archive = root / '基础测试集/预测.npz'
            archive.write_bytes(archive.read_bytes() + b'tampered')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                study.load_export(root / '基础测试集')

    def test_ucf_confirmation_protocol_is_staged_and_fixed(self):
        path = ROOT / 'scripts/repro/semantic_alignment_ucf_confirmation.py'
        self.assertTrue(path.is_file(), 'missing UCF101 semantic-gate confirmation runner')
        spec = importlib.util.spec_from_file_location('semantic_alignment_ucf_confirmation', path)
        study = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(study)

        self.assertEqual(study.planned_cases(), [
            {'phase': 'UCF101单种子独立筛选', 'seed': 1},
            {'phase': 'UCF101三种子独立确认', 'seed': 2},
            {'phase': 'UCF101三种子独立确认', 'seed': 3},
        ])
        self.assertTrue(study.screening_decision(.01, 0.)['passes'])
        self.assertFalse(study.screening_decision(0., 1.)['passes'])
        self.assertFalse(study.screening_decision(1., -.01)['passes'])
        self.assertTrue(study.confirmation_decision([.1, -.1, .2], 0.)['passes'])
        self.assertFalse(study.confirmation_decision([.1, -.1, -.2], 0.)['passes'])
        protocol = study.build_protocol(
            '/server/data', '/server/clip.pt', '/server/output', 'fixed-commit',
            ROOT / 'reports/experiments/语义一致性门控筛选-20260914-v2/结果分析.json')
        prerequisite_path = Path(protocol['DTD_development_prerequisite']['path'])
        self.assertFalse(prerequisite_path.is_absolute())
        self.assertEqual(protocol['maximum_new_trainings'], 3)


if __name__ == '__main__':
    unittest.main()
