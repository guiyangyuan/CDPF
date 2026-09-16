import importlib.util
from pathlib import Path
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/repro/测量统一推理效率.py"
SPEC = importlib.util.spec_from_file_location("测量统一推理效率", SCRIPT)
efficiency = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(efficiency)


class UnifiedInferenceEfficiencyMath(unittest.TestCase):
    def test_description_mean_ignores_padding_and_normalizes(self):
        features = torch.tensor([
            [[2.0, 0.0], [0.0, 4.0], [999.0, 999.0]],
            [[-3.0, 0.0], [999.0, 999.0], [999.0, 999.0]],
        ])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        actual = efficiency.description_mean_features(features, mask)
        expected = torch.tensor([[2 ** -0.5, 2 ** -0.5], [-1.0, 0.0]])
        torch.testing.assert_close(actual, expected)

    def test_two_fusion_formulas_match_the_registered_definition(self):
        student = torch.tensor([[2.0, -1.0]])
        images = torch.tensor([[2.0, 0.0]])
        classes = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        scale = torch.tensor(2.0)
        description = torch.tensor([[2.0, -2.0]])
        logits = efficiency.fuse_from_features(student, images, classes, scale, "logit_fusion")
        probabilities = efficiency.fuse_from_features(
            student, images, classes, scale, "probability_fusion")
        torch.testing.assert_close(logits, 0.5 * (student + description))
        torch.testing.assert_close(
            probabilities, 0.5 * torch.softmax(student, -1) + 0.5 * torch.softmax(description, -1))
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(1))

    def test_timing_summary_uses_batch_size_for_latency_and_throughput(self):
        result = efficiency.summarize_milliseconds([10.0, 20.0, 30.0], batch_size=5)
        self.assertAlmostEqual(result["批次延迟均值_ms"], 20.0)
        self.assertAlmostEqual(result["单图折算延迟_ms"], 4.0)
        self.assertAlmostEqual(result["吞吐_图每秒"], 250.0)


if __name__ == "__main__":
    unittest.main()
