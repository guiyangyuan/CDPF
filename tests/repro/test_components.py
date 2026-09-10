import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from trainers.textconsistency_common import (
    ViewAggregator, description_rows, prediction_logits, training_losses,
    load_complete_checkpoint,
)


class ComponentBehavior(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_masked_padding_cannot_change_predictions(self):
        image = torch.randn(2, 8)
        views = torch.randn(3, 5, 8)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.bool)
        changed = views.clone(); changed[~mask] = 10000
        for mode in ('mean', 'flat', 'hierarchical'):
            with self.subTest(mode=mode):
                model = ViewAggregator(8, 5, mode, n_groups=4, use_feature_adapter=mode != 'mean')
                a, _ = model(image, views, mask)
                b, _ = model(image, changed, mask)
                torch.testing.assert_close(a, b)
                self.assertTrue(torch.isfinite(a).all())

    def test_each_trainable_aggregation_module_receives_gradient(self):
        image = torch.randn(4, 8)
        views = torch.randn(3, 7, 8)
        labels = torch.tensor([0, 1, 2, 1])
        for mode, adapter in [('flat', False), ('flat', True), ('hierarchical', False), ('hierarchical', True)]:
            model = ViewAggregator(8, 7, mode, n_groups=3, use_feature_adapter=adapter)
            features, attention = model(image, views)
            teacher = torch.einsum('bd,bcd->bc', image, features)
            student = torch.randn(4, 3, requires_grad=True)
            loss, _ = training_losses(student, teacher, labels, attention, 1., 2., 1., 0.01)
            loss.backward()
            for name, parameter in model.named_parameters():
                with self.subTest(mode=mode, adapter=adapter, parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(sum(p.grad.abs().sum().item() for p in model.parameters()), 0)
            self.assertGreater(student.grad.abs().sum().item(), 0)

    def test_kd_updates_student_but_does_not_backpropagate_into_teacher(self):
        student = torch.randn(3, 4, requires_grad=True)
        teacher = torch.randn(3, 4, requires_grad=True)
        _, pieces = training_losses(student, teacher, torch.tensor([0, 1, 2]), {}, 1., 2., 1., 0.)
        student_grad, teacher_grad = torch.autograd.grad(pieces['loss_kd'], (student, teacher), allow_unused=True)
        self.assertIsNone(teacher_grad)
        self.assertGreater(student_grad.abs().sum().item(), 0)

    def test_old_contrastive_term_cannot_silently_be_enabled(self):
        with self.assertRaises(ValueError):
            training_losses(torch.randn(2, 3), torch.randn(2, 3), torch.tensor([0, 1]), {}, 0., 1., 1., 0., 0.1)

    def test_duplicate_descriptions_are_not_counted_as_additional_views(self):
        rows = description_rows({'Cat': ['first', 'first', ' second ', '', 'second']}, ['cat'], 7)
        self.assertEqual(rows, [['first', 'second']])
        with self.assertRaises(ValueError):
            description_rows({'Cat': ['first']}, ['dog'], 4)

    def test_explicit_epoch_never_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp, 'textconsistency'); folder.mkdir()
            torch.save({'state_dict': {}, 'epoch': 3}, folder / 'model.pth.tar-3')
            trainer = SimpleNamespace(get_model_names=lambda: ['textconsistency'])
            with self.assertRaises(FileNotFoundError):
                load_complete_checkpoint(trainer, tmp, 2)

    def test_incomplete_state_is_rejected_before_mutating_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = nn.ModuleDict({'prompt_learner': nn.Linear(2, 2), 'view_aggregator': nn.Linear(2, 2)})
            before = module.prompt_learner.weight.detach().clone()
            folder = Path(tmp, 'textconsistency'); folder.mkdir()
            torch.save({'state_dict': {'prompt_learner.weight': torch.ones(2, 2)}, 'epoch': 1}, folder / 'model.pth.tar-1')
            trainer = SimpleNamespace(get_model_names=lambda: ['textconsistency'], _models={'textconsistency': module})
            with self.assertRaises(ValueError):
                load_complete_checkpoint(trainer, tmp, 1)
            torch.testing.assert_close(before, module.prompt_learner.weight)

    def test_baseline_rejects_teacher_or_ensemble_prediction(self):
        with self.assertRaises(ValueError):
            prediction_logits(torch.ones(2), None, 'teacher')

if __name__ == '__main__':
    unittest.main()
