import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = REPO_ROOT / "论文投稿版本-20260917" / "生成论文材料.py"


def load_generator():
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.use = lambda *_args, **_kwargs: None
    pyplot = types.ModuleType("matplotlib.pyplot")
    patches = types.ModuleType("matplotlib.patches")
    patches.FancyBboxPatch = object
    sys.modules.setdefault("matplotlib", matplotlib)
    sys.modules.setdefault("matplotlib.pyplot", pyplot)
    sys.modules.setdefault("matplotlib.patches", patches)

    spec = importlib.util.spec_from_file_location("paper_material_generator", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PaperElevenDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.generator = load_generator()

    def test_main_results_include_verified_imagenet_values(self):
        results = self.generator.主要结果()

        self.assertEqual(11, len(self.generator.数据集顺序))
        self.assertAlmostEqual(72.38859998959222, results[("ImageNet", "CoOp")]["HM"])
        self.assertAlmostEqual(
            74.05381430160028,
            results[("ImageNet", "固定均值概率融合")]["HM"],
        )

    def test_component_results_cover_all_eleven_datasets(self):
        components = self.generator.组件结果()

        for method in (
            "CoOp",
            "Description only",
            "Logit fusion",
            "Probability fusion",
        ):
            self.assertEqual(
                11,
                sum((dataset, method) in components for dataset in self.generator.数据集顺序),
            )

    def test_seed_level_comparison_contains_33_positive_pairs(self):
        gains = self.generator.读取逐种子增益()

        self.assertEqual(11, len(gains))
        self.assertEqual(33, sum(len(values) for values in gains.values()))
        self.assertTrue(all(value > 0 for values in gains.values() for value in values))


if __name__ == "__main__":
    unittest.main()
