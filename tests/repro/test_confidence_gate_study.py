"""Protocol checks for the staged cross-dataset gate study."""
import unittest

from scripts.repro.confidence_gate_study import (
    confirmation_decision,
    planned_cases,
    screening_decision,
)


class ConfidenceGateStudyTest(unittest.TestCase):
    def test_screening_requires_both_comparisons_and_novel_preservation(self):
        self.assertTrue(screening_decision(.1, .2, 0.)['passes'])
        self.assertFalse(screening_decision(0., .2, 0.)['passes'])
        self.assertFalse(screening_decision(.1, -.1, 0.)['passes'])
        self.assertFalse(screening_decision(.1, .2, -.01)['passes'])

    def test_confirmation_uses_stability_instead_of_one_point_cutoff(self):
        accepted = confirmation_decision([.4, .2, -.1], [.3, .1, -.05], .05)
        self.assertTrue(accepted['passes'])
        self.assertEqual(accepted['residual_positive_seeds'], 2)
        self.assertEqual(accepted['attention_positive_seeds'], 2)
        self.assertFalse(confirmation_decision([.4, -.2, -.1], [.3, .1, -.05], .05)['passes'])

    def test_cases_are_staged_and_unique(self):
        cases = planned_cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual(cases[0], {'phase': 'DTD单种子筛选', 'dataset': 'dtd', 'seed': 1, 'variant': 'uniform'})
        self.assertEqual(cases[-1], {'phase': 'UCF101三种子确认', 'dataset': 'ucf101', 'seed': 3, 'variant': 'attentive'})
        self.assertEqual(len({(c['dataset'], c['seed'], c['variant']) for c in cases}), 12)


if __name__ == '__main__':
    unittest.main()
