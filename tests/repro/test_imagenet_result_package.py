import json
import re
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "reports/experiments/ImageNet扩展验证-20260918"


def load_json(relative: str):
    return json.loads((RESULT / relative).read_text(encoding="utf-8"))


class ImageNetResultPackageTest(unittest.TestCase):
    def test_completed_current_model_protocol_and_summary_are_preserved(self):
        protocol = load_json("实验协议.json")
        status = load_json("执行状态.json")
        summary = load_json("结果汇总.json")

        self.assertEqual(protocol["数据集"], "ImageNet")
        self.assertEqual(protocol["种子"], [1, 2, 3])
        self.assertEqual(protocol["训练轮数"], 20)
        self.assertEqual(protocol["样本数"], 16)
        self.assertIn("0.5*softmax", protocol["主比较"])
        self.assertEqual(status["状态"], "完成")
        self.assertEqual(status["已完成命令"], status["总命令"])
        self.assertEqual(status["总命令"], 16)
        self.assertEqual(status["GPU"], "NVIDIA GeForce RTX 4090")

        fused = summary["均值"]["mean_probability_fusion"]
        self.assertAlmostEqual(fused["Base均值"], 76.65733333333333)
        self.assertAlmostEqual(fused["Novel均值"], 71.62133333333333)
        self.assertAlmostEqual(fused["均值的HM"], 74.05381430160028)
        self.assertTrue(summary["通过"])

    def test_all_three_seed_manifests_belong_to_the_retained_trainer(self):
        for seed in (1, 2, 3):
            manifest = load_json(f"基线训练/seed{seed}/run_manifest.json")
            signature = manifest["model_signature"]
            self.assertEqual(manifest["seed"], seed)
            self.assertEqual(manifest["dataset"], "ImageNet")
            self.assertEqual(signature["trainer"], "CoOp_TextConsistency_Ablation")
            self.assertEqual(signature["ablation_group"], "baseline")
            self.assertFalse(signature["loss_identity"]["active"])

        package_text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in RESULT.rglob("*")
            if path.is_file()
        )
        self.assertNotIn("CoOp_TextConsistency_V2_FG", package_text)
        self.assertNotIn("description_instance", package_text)
        self.assertNotIn("adk_probability_fusion", package_text)

    def test_checkpoint_and_server_backup_hashes_are_recorded(self):
        expected_checkpoints = {
            1: "e99959bf7587599284b2c52046d3b15d0f5430ef95b4c17cfd803dc2d153cd87",
            2: "6e7548a552474998745b8ed81f7884cf178a52f5ea53b65f1062aff9b0d46da9",
            3: "50b44b7ab304c1388ff87eb2d0b1b08bb45e6c482d408e8a9cbc5f48413ed11d",
        }
        for seed, expected in expected_checkpoints.items():
            actual = (RESULT / f"基线训练/seed{seed}/checkpoint.sha256").read_text().strip()
            self.assertEqual(actual, expected)

        backup_line = (RESULT / "服务器备份.sha256").read_text().strip()
        self.assertRegex(backup_line, r"^[0-9a-f]{64}  ImageNet完整结果备份-20260922\.tar\.gz$")
        self.assertTrue(backup_line.startswith("4847f5104248ca360252909c6cb800c5ab5e1dfccc18d3865446de48a73a9a79"))

    def test_only_final_fusion_evaluations_are_imported(self):
        for seed in (1, 2, 3):
            for split in ("base", "new"):
                directory = RESULT / f"融合评估/seed{seed}/{split}"
                self.assertTrue((directory / "run_manifest.json").is_file())
                self.assertTrue((directory / "effective_config.yaml").is_file())

        relative_paths = [str(path.relative_to(RESULT)) for path in RESULT.rglob("*")]
        self.assertFalse(any("失败" in path or "补丁前" in path for path in relative_paths))
        self.assertFalse(any(path.endswith("predictions.jsonl") for path in relative_paths))
        self.assertFalse(any(re.search(r"model\.pth", path) for path in relative_paths))


if __name__ == "__main__":
    unittest.main()
