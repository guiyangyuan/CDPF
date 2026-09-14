"""Protocol checks for fusion-aligned staged validation."""
import unittest

from scripts.repro.fusion_aligned_study import (
    confirmation_decision,
    planned_cases,
    screening_decision,
)


class FusionAlignedStudyTest(unittest.TestCase):
    def test_single_seed_screen_requires_hm_gain_and_novel_preservation(self):
        self.assertTrue(screening_decision(.01, 0.)['passes'])
        self.assertFalse(screening_decision(0., 1.)['passes'])
        self.assertFalse(screening_decision(1., -.01)['passes'])

    def test_three_seed_confirmation_requires_mean_stability(self):
        accepted = confirmation_decision([.4, .2, -.1], .05)
        self.assertTrue(accepted['passes'])
        self.assertEqual(accepted['positive_seeds'], 2)
        self.assertFalse(confirmation_decision([.4, -.2, -.1], .05)['passes'])
        self.assertFalse(confirmation_decision([.4, .2, -.1], -.01)['passes'])

    def test_plan_has_one_training_per_seed_and_keeps_ucf_last(self):
        cases = planned_cases()
        self.assertEqual(len(cases), 6)
        self.assertEqual(cases[0], {'phase': 'DTD单种子筛选', 'dataset': 'dtd', 'seed': 1})
        self.assertEqual(cases[-1], {'phase': 'UCF101三种子确认', 'dataset': 'ucf101', 'seed': 3})
        self.assertEqual(len({(case['dataset'], case['seed']) for case in cases}), 6)


if __name__ == '__main__':
    unittest.main()
