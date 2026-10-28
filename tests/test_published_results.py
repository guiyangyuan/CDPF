"""Validate the public CDPF numbers using only the Python standard library."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/summarize_results.py"
RESULTS = ROOT / "results/cdpf_results.json"


class PublishedResultsTests(unittest.TestCase):
    def run_summary(self, path=None):
        command = [sys.executable, "-I", str(SCRIPT)]
        if path is not None:
            command.append(str(path))
        return subprocess.run(command, cwd=tempfile.gettempdir(), capture_output=True, text=True)

    def test_recomputes_paper_macro_and_distinct_seed_pair_gain_without_torch(self):
        result = self.run_summary()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["dataset_count"], 11)
        self.assertEqual(report["seeds"], [1, 2, 3])
        expected = {
            "student": (81.39056206367, 70.33834170233536, 75.06350128002724),
            "mean_probability_fusion": (81.13378960651586, 75.99927856201488, 78.32688685701936),
            "matched_template_probability_fusion": (80.53478973763595, 74.33840980642087, 77.10939766660103),
        }
        for branch, (base, novel, hm) in expected.items():
            with self.subTest(branch=branch):
                self.assertAlmostEqual(report["macro"][branch]["base"], base, places=9)
                self.assertAlmostEqual(report["macro"][branch]["new"], novel, places=9)
                self.assertAlmostEqual(report["macro"][branch]["hm"], hm, places=9)
        comparison = report["cdpf_vs_matched_template"]
        self.assertEqual(comparison["pair_count"], 33)
        self.assertEqual(comparison["directions"], {"positive": 28, "negative": 4, "zero": 1})
        self.assertAlmostEqual(comparison["macro_hm_gain"], 1.21748919041833, places=9)
        self.assertAlmostEqual(comparison["mean_paired_seed_hm_gain"], 1.2512991589415192, places=9)
        self.assertNotAlmostEqual(comparison["macro_hm_gain"], comparison["mean_paired_seed_hm_gain"], places=3)
        self.assertTrue(all(value < 0 for value in comparison["per_seed_hm_gains"]["stanford_cars"].values()))

    def test_rejects_missing_seed_and_corrupt_metrics(self):
        original = json.loads(RESULTS.read_text(encoding="utf-8"))
        mutations = {
            "missing seed": lambda data: data["summary"]["per_seed"]["dtd"].pop("2"),
            "missing dataset": lambda data: data["summary"]["per_seed"].pop("imagenet"),
            "seed HM": lambda data: data["summary"]["per_seed"]["dtd"]["1"]["student"].update(hm=99.0),
            "dataset HM": lambda data: data["summary"]["per_dataset"]["dtd"]["student"].update(hm=99.0),
            "macro HM": lambda data: data["summary"]["macro_hm"].update(student=99.0),
            "invalid accuracy": lambda data: data["summary"]["per_seed"]["dtd"]["1"]["student"].update(base=-1.0),
            "nonfinite accuracy": lambda data: data["summary"]["per_seed"]["dtd"]["1"]["student"].update(base=float("nan")),
            "missing branch": lambda data: data["summary"]["per_seed"]["dtd"]["1"].pop("matched_template_probability_fusion"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corrupt.json"
            for label, mutate in mutations.items():
                with self.subTest(corruption=label):
                    data = copy.deepcopy(original)
                    mutate(data)
                    path.write_text(json.dumps(data), encoding="utf-8")
                    result = self.run_summary(path)
                    self.assertEqual(result.returncode, 1, result.stdout)
                    self.assertIn("Invalid results:", result.stderr)
                    self.assertEqual(result.stdout, "")

    def test_export_retains_accepted_identity_and_original_branch_names(self):
        data = json.loads(RESULTS.read_text(encoding="utf-8"))
        self.assertEqual(data["provenance"]["analysis_sha256"], "c33d1f22113ac7a38e73fb4ec499a70065f02f43a30ae24734301f96e1fb9d10")
        self.assertEqual(len(data["provenance"]["dataset_split"]), 22)
        self.assertEqual(data["protocol"]["alpha"], 0.5)
        self.assertEqual(data["protocol"]["additional_training"], 0)
        self.assertEqual(data["protocol"]["matched_template_pool_size"], 60)
        self.assertEqual(data["main_hm_definition"], "H(mean_seed_base, mean_seed_new)")
        self.assertEqual(set(data["branch_definitions"]), {"student", "mean_probability_fusion", "matched_template_probability_fusion"})
        self.assertIn("adk_probability_fusion", data["summary"]["macro_hm"])
        self.assertIn("classname_probability_fusion", data["summary"]["macro_hm"])
        self.assertAlmostEqual(
            data["summary"]["per_dataset"]["imagenet"]["mean_probability_fusion"]["hm"],
            74.05381430160028,
            places=6,
        )
        wins = [
            case["mean_probability_fusion"]["hm"] > case["student"]["hm"]
            for seeds in data["summary"]["per_seed"].values()
            for case in seeds.values()
        ]
        self.assertEqual(len(wins), 33)
        self.assertTrue(all(wins))
        self.assertNotIn("prediction_cases", data)
        self.assertNotIn("paired_deltas", data["summary"])


if __name__ == "__main__":
    unittest.main()
