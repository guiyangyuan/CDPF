import subprocess
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.repro import validation_analysis as analysis
from scripts.repro import validation_probe as probe


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT.parents[1] / "tc-venv/bin/python"


class ValidationAnalysisCli(unittest.TestCase):
    def test_help_exposes_analyze_and_aggregate(self):
        result = subprocess.run(
            [str(PYTHON), "scripts/repro/validation_analysis.py", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("analyze", result.stdout)
        self.assertIn("aggregate", result.stdout)


def _synthetic_payload(dataset, seed):
    labels = np.array([0, 1, 1], dtype=np.int64)
    logits = {
        "C": np.array([[3, 0], [5, 0], [0, 3]], dtype=np.float32),
        "S0": np.array([[2, 0], [5, 0], [0, 2]], dtype=np.float32),
        "S1": np.array([[4, 0], [0, 4], [0, 4]], dtype=np.float32),
        "M": np.array([[2, 0], [0, 2], [0, 2]], dtype=np.float32),
        "T0": np.array([[1, 0], [2, 0], [0, 1]], dtype=np.float32),
        "T1": np.array([[1, 0], [2, 0], [0, 1]], dtype=np.float32),
    }
    return {
        "logits": logits, "labels": labels,
        "sample_ids": [f"{dataset}-{seed}-{i}" for i in range(3)],
        "classnames": ["zero", "one"], "teacher_exact_equal": True,
        "forward_precision": "float32",
    }


def _provenance(dataset, seed):
    return {
        "source_sha256": analysis.EXPECTED_SOURCES[dataset],
        "clip_sha256": probe.CLIP_SHA256,
        "template_sha256": analysis.EXPECTED_TEMPLATES[dataset],
        "config_sha256": "f" * 64,
        "config_compatibility_sha256": dataset * 4,
        "checkpoint_sha256": {role: f"{dataset}-{seed}-{role}" for role in probe.ROLES},
    }


class ValidationAnalysisContracts(unittest.TestCase):
    def test_metrics_use_stable_softmax_sum_brier_and_probability_margin(self):
        logits = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
        result = analysis.branch_metrics(logits, np.array([0, 1]))
        probability = np.exp(2.0) / (np.exp(2.0) + 1.0)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["accuracy_pct"], 100.0)
        self.assertAlmostEqual(result["nll"], -np.log(probability))
        self.assertAlmostEqual(result["multiclass_brier"], 2 * (1 - probability) ** 2)
        self.assertAlmostEqual(result["probability_top1_top2_margin"], 2 * probability - 1)
        extreme = analysis.branch_metrics(np.array([[-10000.0, 10000.0]]), np.array([0]))
        self.assertTrue(np.isfinite(extreme["nll"]))
        self.assertAlmostEqual(extreme["nll"], 20000.0)
        fused = analysis.fixed_probability_fusion_metrics(
            np.array([[-10000.0, 10000.0]]), np.array([[-10000.0, 10000.0]]), np.array([0]))
        self.assertTrue(np.isfinite(fused["metrics"]["nll"]))
        self.assertAlmostEqual(fused["metrics"]["nll"], 20000.0)

    def test_analysis_rejects_hash_nonfinite_and_row_permutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            export = root / "export"
            probe.write_export_artifact(export, _synthetic_payload("dtd", 1), {
                "dataset_key": "dtd", "seed": 1, "protocol": probe.PROTOCOL,
                "provenance": _provenance("dtd", 1),
            })
            receipt = json.loads((export / "export.json").read_text())
            receipt["rows"][0], receipt["rows"][1] = receipt["rows"][1], receipt["rows"][0]
            (export / "export.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "sample alignment"):
                analysis.load_export(export / "export.json")
            receipt["rows"][0], receipt["rows"][1] = receipt["rows"][1], receipt["rows"][0]
            receipt["npz_sha256"] = "0" * 64
            (export / "export.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "hash"):
                analysis.load_export(export / "export.json")

    def test_end_to_end_nine_case_analysis_reports_all_combinations_and_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reports = []
            for dataset in analysis.DATASETS:
                for seed in (1, 2, 3):
                    export = root / f"export-{dataset}-{seed}"
                    probe.write_export_artifact(export, _synthetic_payload(dataset, seed), {
                        "dataset_key": dataset, "seed": seed, "protocol": probe.PROTOCOL,
                        "provenance": _provenance(dataset, seed),
                    })
                    report = analysis.analyze_export(export / "export.json", root / f"analysis-{dataset}-{seed}")
                    self.assertEqual(len(report["combinations"]), 18)
                    self.assertEqual(len(report["predictions"]), 24)
                    reports.append(report)
            aggregate = analysis.aggregate_reports(reports)
            self.assertEqual(aggregate["case_count"], 9)
            self.assertTrue(aggregate["guards"]["all_passed"])
            self.assertEqual(set(aggregate["dataset_means"]), set(analysis.DATASETS))
            self.assertIn("sample_sd", aggregate["macro"]["S1+M_logits"])
            self.assertEqual(len(aggregate["complete_macro_metrics"]), 24)
            self.assertIn("multiclass_brier", aggregate["complete_macro_metrics"]["T1"])

    def test_aggregate_rejects_missing_duplicate_and_mixed_provenance(self):
        template = {
            "status": "passed", "dataset_key": "ucf101", "seed": 1,
            "protocol": probe.PROTOCOL,
            "provenance": _provenance("ucf101", 1),
            "branches": {}, "combinations": {
                "S1+M_logits": {"accuracy_pct": 80.0}, "C+M_logits": {"accuracy_pct": 70.0},
                "C+M_probability": {"accuracy_pct": 70.0}, "S1+M_probability": {"accuracy_pct": 80.0},
                "S0+M_logits": {"accuracy_pct": 60.0},
            },
        }
        with self.assertRaisesRegex(ValueError, "nine"):
            analysis.aggregate_reports([template])
        cases = []
        for dataset in analysis.DATASETS:
            for seed in (1, 2, 3):
                case = copy.deepcopy(template); case["dataset_key"] = dataset; case["seed"] = seed
                case["provenance"] = _provenance(dataset, seed)
                cases.append(case)
        cases[-1]["seed"] = 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analysis.aggregate_reports(cases)
        cases[-1]["seed"] = 3; cases[-1]["provenance"]["clip_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "compatible"):
            analysis.aggregate_reports(cases)


if __name__ == "__main__":
    unittest.main()
