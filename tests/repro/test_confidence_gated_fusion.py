"""Targeted checks for the frozen confidence-gated fusion formula."""
import unittest

import numpy as np

from scripts.repro.confidence_gated_fusion import (
    confidence_gated_fusion,
    normalized_entropy_confidence,
)


class ConfidenceGatedFusionTest(unittest.TestCase):
    def test_entropy_gate_has_fixed_endpoints(self):
        student = np.array([[0., 0., 0.], [40., -40., -40.]], dtype=np.float32)
        confidence = normalized_entropy_confidence(student)
        self.assertEqual(confidence.shape, (2,))
        self.assertAlmostEqual(float(confidence[0]), 0., places=7)
        self.assertAlmostEqual(float(confidence[1]), 1., places=7)

    def test_fusion_uses_no_tunable_coefficient(self):
        student = np.array([[0., 0.], [30., -30.]], dtype=np.float32)
        description = np.array([[2., 4.], [6., 8.]], dtype=np.float32)
        residual = np.array([[5., -5.], [1., 3.]], dtype=np.float32)
        result = confidence_gated_fusion(student, description, residual)
        np.testing.assert_allclose(result['simple_fusion'][0], [1., 2.], rtol=0, atol=0)
        np.testing.assert_allclose(result['gated_fusion'][0], [1., 2.], rtol=0, atol=1e-7)
        np.testing.assert_allclose(result['gated_fusion'][1], [19., -8.], rtol=0, atol=1e-6)

    def test_rejects_misaligned_or_nonfinite_logits(self):
        good = np.ones((2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'same shape'):
            confidence_gated_fusion(good, good[:1], good)
        bad = good.copy(); bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            confidence_gated_fusion(bad, good, good)


if __name__ == '__main__':
    unittest.main()
