import unittest

import torch
from torch.nn import functional as F

from trainers.textconsistency_common import training_losses


def frozen_full_kl(student, teacher, temperature):
    return (F.kl_div(F.log_softmax(student / temperature, dim=-1),
                     F.softmax(teacher.detach() / temperature, dim=-1),
                     reduction='batchmean') * temperature**2)


def independent_weighted_nckd(student, teacher, labels, temperature):
    """Full categorical KL minus the target/non-target binary KL."""
    q = F.softmax(teacher.detach() / temperature, dim=-1)
    p = F.softmax(student / temperature, dim=-1)
    rows = torch.arange(labels.numel())
    qy, py = q[rows, labels], p[rows, labels]
    full = (q * (q.log() - p.log())).sum(-1)
    binary = (qy * (qy.log() - py.log())
              + (1 - qy) * ((1 - qy).log() - (1 - py).log()))
    return ((full - binary) * temperature**2).mean()


class KDLossBehavior(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        torch.set_num_threads(2)

    def call(self, student, teacher, labels, **kwargs):
        return training_losses(student, teacher, labels, {}, 1., 2., 0., 0.,
                               kd_loss_mode='weighted_nckd', **kwargs)

    def test_default_and_explicit_full_kl_are_bit_exact_to_frozen_formula(self):
        labels = torch.tensor([0, 2, 1])
        teacher = torch.randn(3, 4, dtype=torch.float64)
        base = torch.randn(3, 4, dtype=torch.float64)
        students = [base.clone().requires_grad_() for _ in range(3)]
        default = training_losses(students[0], teacher, labels, {}, 1., 2., 0., 0.)[1]['loss_kd']
        explicit = training_losses(students[1], teacher, labels, {}, 1., 2., 0., 0.,
                                   kd_loss_mode='full_kl')[1]['loss_kd']
        frozen = frozen_full_kl(students[2], teacher, 2.)
        self.assertTrue(torch.equal(default, frozen))
        self.assertTrue(torch.equal(explicit, frozen))
        grads = [torch.autograd.grad(loss, student)[0]
                 for loss, student in zip((default, explicit, frozen), students)]
        self.assertTrue(torch.equal(grads[0], grads[2]))
        self.assertTrue(torch.equal(grads[1], grads[2]))

    def test_weighted_nckd_matches_independent_value_and_student_gradient(self):
        labels = torch.tensor([0, 3, 1])
        teacher = torch.randn(3, 5, dtype=torch.float64)
        a = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        actual = self.call(a, teacher, labels)[1]['loss_kd']
        expected = independent_weighted_nckd(b, teacher, labels, 2.)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
        ga = torch.autograd.grad(actual, a)[0]
        gb = torch.autograd.grad(expected, b)[0]
        torch.testing.assert_close(ga, gb, rtol=1e-10, atol=1e-12)
        self.assertTrue(torch.equal(ga.gather(1, labels[:, None]), torch.zeros(3, 1, dtype=ga.dtype)))

    def test_teacher_is_detached_only_from_kd(self):
        labels = torch.tensor([0, 1, 2])
        student = torch.randn(3, 4, requires_grad=True)
        teacher = torch.randn(3, 4, requires_grad=True)
        _, pieces = training_losses(student, teacher, labels, {}, 1., 2., 1., 0.,
                                    kd_loss_mode='weighted_nckd')
        kd_teacher = torch.autograd.grad(pieces['loss_kd'], teacher, allow_unused=True)[0]
        ce_teacher = torch.autograd.grad(pieces['loss_adaptive_ce'], teacher)[0]
        self.assertIsNone(kd_teacher)
        self.assertGreater(ce_teacher.abs().sum().item(), 0)

    def test_non_targets_are_conditionally_normalized(self):
        labels = torch.tensor([0])
        student = torch.tensor([[50., 0., 0.]], requires_grad=True)
        teacher = torch.tensor([[-50., 3., -3.]])
        kd = self.call(student, teacher, labels)[1]['loss_kd']
        self.assertGreater(kd.item(), 0)
        grad = torch.autograd.grad(kd, student)[0]
        self.assertEqual(grad[0, 0].item(), 0.)
        self.assertNotEqual(grad[0, 1].item(), 0.)

    def test_class_permutation_and_batch_mean_weighting(self):
        labels = torch.tensor([0, 2, 1])
        student = torch.randn(3, 4, dtype=torch.float64)
        teacher = torch.randn(3, 4, dtype=torch.float64)
        value = self.call(student, teacher, labels)[1]['loss_kd']
        order = torch.tensor([2, 0, 3, 1])
        inverse = torch.empty_like(order); inverse[order] = torch.arange(4)
        permuted = self.call(student[:, order], teacher[:, order], inverse[labels])[1]['loss_kd']
        torch.testing.assert_close(value, permuted)
        singles = [self.call(student[i:i+1], teacher[i:i+1], labels[i:i+1])[1]['loss_kd']
                   for i in range(3)]
        torch.testing.assert_close(value, torch.stack(singles).mean())

    def test_extreme_logits_are_finite_and_teacher_mass_is_stable(self):
        student = torch.tensor([[10000., -10000., 0.], [-10000., 10000., 0.]], requires_grad=True)
        teacher = torch.tensor([[10000., 9999., -10000.], [10000., -10000., 9999.]])
        kd = self.call(student, teacher, torch.tensor([0, 0]))[1]['loss_kd']
        self.assertTrue(torch.isfinite(kd))
        self.assertTrue(torch.isfinite(torch.autograd.grad(kd, student)[0]).all())

    def test_two_classes_have_zero_weighted_nckd(self):
        student = torch.randn(4, 2, requires_grad=True)
        teacher = torch.randn(4, 2)
        kd = self.call(student, teacher, torch.tensor([0, 1, 0, 1]))[1]['loss_kd']
        self.assertEqual(kd.item(), 0.)
        self.assertTrue(torch.equal(torch.autograd.grad(kd, student)[0], torch.zeros_like(student)))

    def test_kd_zero_accepts_teacherless_weighted_mode(self):
        student = torch.randn(2, 3)
        total, pieces = training_losses(student, None, torch.tensor([0, 2]), {}, 0., 2., 0., 0.,
                                       kd_loss_mode='weighted_nckd')
        self.assertEqual(pieces['loss_kd'].item(), 0.)
        torch.testing.assert_close(total, F.cross_entropy(student, torch.tensor([0, 2])))

    def test_invalid_options_are_rejected_even_when_kd_is_disabled(self):
        args = (torch.randn(2, 3), None, torch.tensor([0, 1]), {}, 0.)
        for temperature in (0., -1., float('nan'), float('inf')):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                training_losses(*args, temperature, 0., 0.)
        for position in range(3):
            weights = [0., 0., 0.]
            for bad in (-1., float('nan'), float('inf')):
                weights[position] = bad
                with self.subTest(position=position, bad=bad), self.assertRaises(ValueError):
                    training_losses(*args, 2., *weights[:2], contrastive_weight=weights[2])
            weights[position] = 0.
        with self.assertRaises(ValueError):
            training_losses(*args, 2., 0., 0., kd_loss_mode='other')

    def test_active_weighted_mode_rejects_malformed_inputs(self):
        good_student = torch.randn(2, 3)
        cases = [
            (good_student, None, torch.tensor([0, 1])),
            (good_student, torch.randn(2, 4), torch.tensor([0, 1])),
            (torch.randn(2, 1), torch.randn(2, 1), torch.tensor([0, 0])),
            (good_student, torch.randn(2, 3), torch.tensor([0, 3])),
            (good_student, torch.randn(2, 3), torch.tensor([0, -1])),
            (good_student, torch.randn(2, 3), torch.tensor([[0], [1]])),
            (good_student, torch.randn(2, 3), torch.tensor([0, 1], dtype=torch.int32)),
            (torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, dtype=torch.long)),
        ]
        for student, teacher, labels in cases:
            with self.subTest(shapes=(student.shape, None if teacher is None else teacher.shape, labels.shape)):
                with self.assertRaises(ValueError):
                    self.call(student, teacher, labels)


if __name__ == '__main__':
    unittest.main()
