import subprocess
import sys
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.repro import validation_analysis as analysis
from scripts.repro import validation_probe as probe


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
FIXED_INVENTORY = json.loads((
    ROOT / "reports/diagnostics/error-analysis-20260911/evidence/proposed-probe-inputs.json"
).read_text())
FIXED_CASES = {
    (case["dataset"], case["seed"]): case for case in FIXED_INVENTORY["cases"]
}


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


def _guard_payload(dataset, seed):
    """Three classes make equal-logit and equal-probability fusion disagree."""
    return {
        "logits": {
            "C": np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]], dtype=np.float32),
            "S0": np.array([[-6, 0, -6], [-6, -6, 0], [0, -6, -6]], dtype=np.float32),
            "S1": np.array([[-6, -4, -6], [0, 4, 0], [0, 0, 4]], dtype=np.float32),
            "M": np.array([[1, -2, 0], [0, 1, -2], [-2, 0, 1]], dtype=np.float32),
            "T0": np.array([[1, -2, 0], [0, 1, -2], [-2, 0, 1]], dtype=np.float32),
            "T1": np.array([[1, -2, 0], [0, 1, -2], [-2, 0, 1]], dtype=np.float32),
        },
        "labels": np.array([0, 1, 2], dtype=np.int64),
        "sample_ids": [f"{dataset}-{seed}-guard-{index}" for index in range(3)],
        "classnames": ["zero", "one", "two"],
        "teacher_exact_equal": True,
        "forward_precision": "float32",
    }


def _row_digest(payload):
    rows = [(sample_id, int(label), payload["classnames"][int(label)])
            for sample_id, label in zip(payload["sample_ids"], payload["labels"])]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()


def _fixture_inventory(root, payload_factory):
    cases = []
    for dataset in analysis.DATASETS:
        for seed in (1, 2, 3):
            payload = payload_factory(dataset, seed)
            checkpoints = {}
            for role, weight in (("coop", 0.0), ("kd0", 0.0), ("kd1", 1.0)):
                checkpoints[role] = {
                    "workspace_relative_checkpoint": f"fixture/{dataset}/{seed}/{role}.pt",
                    "sha256": hashlib.sha256(f"checkpoint-{dataset}-{seed}-{role}".encode()).hexdigest(),
                    "manifest_sha256": hashlib.sha256(f"manifest-{dataset}-{seed}-{role}".encode()).hexdigest(),
                    "model_source_sha256": probe.EXPECTED_SOURCES[dataset],
                    "effective_kd_weight": weight, "epoch": 20,
                }
            cases.append({
                "dataset": dataset, "seed": seed, "split": "val", "candidate_classes": "base",
                "train_samples_sha256": hashlib.sha256(f"train-{dataset}-{seed}".encode()).hexdigest(),
                "val_samples_sha256": _row_digest(payload), "clip_sha256": probe.CLIP_SHA256,
                "template_sha256": analysis.EXPECTED_TEMPLATES[dataset],
                "coop_training_template_sha256": None, "checkpoints": checkpoints,
            })
    inventory = {"cases": cases}
    path = root / "fixture-inventory.json"
    path.write_text(json.dumps(inventory))
    digest = probe.sha256_file(path)
    return path, digest, {(case["dataset"], case["seed"]): case for case in cases}


def _provenance(dataset, seed, cases=FIXED_CASES, inventory_sha256=probe.FIXED_INVENTORY_SHA256):
    fixed_case = cases[(dataset, seed)]
    return {
        "source_sha256": analysis.EXPECTED_SOURCES[dataset],
        "clip_sha256": probe.CLIP_SHA256,
        "template_sha256": analysis.EXPECTED_TEMPLATES[dataset],
        "config_sha256": "f" * 64,
        "config_compatibility_sha256": probe.json_digest({"dataset": dataset}),
        "inventory_sha256": inventory_sha256,
        "inventory_case_sha256": probe.json_digest(fixed_case),
        "train_samples_sha256": fixed_case["train_samples_sha256"],
        "val_samples_sha256": fixed_case["val_samples_sha256"],
        "checkpoint_sha256": {
            role: fixed_case["checkpoints"][role]["sha256"] for role in probe.ROLES
        },
        "manifest_sha256": {
            role: fixed_case["checkpoints"][role]["manifest_sha256"] for role in probe.ROLES
        },
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
            inventory, inventory_sha, cases = _fixture_inventory(root, _synthetic_payload)
            trust = {"trusted_inventory_path": inventory,
                     "trusted_inventory_sha256": inventory_sha}
            export = root / "export"
            probe.write_export_artifact(export, _synthetic_payload("dtd", 1), {
                "dataset_key": "dtd", "seed": 1, "protocol": probe.PROTOCOL,
                "provenance": _provenance("dtd", 1, cases, inventory_sha),
            })
            receipt = json.loads((export / "export.json").read_text())
            analysis.load_export(export / "export.json", **trust)
            receipt["npz"] = "/unavailable/download-origin/logits.npz"
            (export / "export.json").write_text(json.dumps(receipt))
            analysis.load_export(export / "export.json", **trust)
            receipt["npz"] = str(export / "logits.npz")
            receipt["rows"][0]["sample_id"] = "different-sample-id"
            (export / "export.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "trusted validation rows"):
                analysis.load_export(export / "export.json", **trust)
            receipt["rows"][0]["sample_id"] = "dtd-1-0"
            receipt["class_order"] = ["one", "zero"]
            for row in receipt["rows"]:
                row["classname"] = receipt["class_order"][row["label"]]
            (export / "export.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "trusted validation rows"):
                analysis.load_export(export / "export.json", **trust)
            receipt["class_order"] = ["zero", "one"]
            for row in receipt["rows"]:
                row["classname"] = receipt["class_order"][row["label"]]
            receipt["rows"][0], receipt["rows"][1] = receipt["rows"][1], receipt["rows"][0]
            (export / "export.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "sample alignment"):
                analysis.load_export(export / "export.json", **trust)
            receipt["rows"][0], receipt["rows"][1] = receipt["rows"][1], receipt["rows"][0]
            prepared_path = root / "prepared.json"
            prepared = {
                "status": "prepared", "protocol": probe.PROTOCOL,
                "dataset_key": "dtd", "seed": 1,
                "inventory": {"case": cases[("dtd", 1)]},
                "samples": {"classes": ["zero", "one"], "val": [
                    {key: row[key] for key in ("sample_id", "label", "classname")}
                    for row in receipt["rows"]]},
                "sample_hashes": {"val": cases[("dtd", 1)]["val_samples_sha256"]},
            }
            prepared_path.write_text(json.dumps(prepared))
            receipt["prepared"] = str(prepared_path)
            receipt["prepared_sha256"] = probe.sha256_file(prepared_path)
            (export / "export.json").write_text(json.dumps(receipt))
            analysis.load_export(export / "export.json", **trust)
            prepared["samples"]["classes"] = ["one", "zero"]
            prepared_path.write_text(json.dumps(prepared))
            receipt["prepared_sha256"] = probe.sha256_file(prepared_path)
            (export / "export.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "prepared identity"):
                analysis.load_export(export / "export.json", **trust)
            receipt["npz_sha256"] = "0" * 64
            (export / "export.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "hash"):
                analysis.load_export(export / "export.json", **trust)

    def test_end_to_end_nine_case_analysis_reports_all_combinations_and_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory, inventory_sha, cases = _fixture_inventory(root, _guard_payload)
            trust = {"trusted_inventory_path": inventory,
                     "trusted_inventory_sha256": inventory_sha}
            reports = []
            for dataset in analysis.DATASETS:
                for seed in (1, 2, 3):
                    export = root / f"export-{dataset}-{seed}"
                    probe.write_export_artifact(export, _guard_payload(dataset, seed), {
                        "dataset_key": dataset, "seed": seed, "protocol": probe.PROTOCOL,
                        "provenance": _provenance(dataset, seed, cases, inventory_sha),
                    })
                    report = analysis.analyze_export(
                        export / "export.json", root / f"analysis-{dataset}-{seed}", **trust)
                    self.assertEqual(len(report["combinations"]), 18)
                    self.assertEqual(len(report["predictions"]), 24)
                    self.assertEqual(report["predictions"]["S1+M_logits"], [0, 1, 2])
                    self.assertEqual(report["predictions"]["S1+M_probability"], [1, 1, 2])
                    reports.append(report)
            aggregate = analysis.aggregate_reports(reports, **trust)
            self.assertEqual(aggregate["case_count"], 9)
            self.assertTrue(aggregate["guards"]["all_passed"])
            self.assertEqual(set(aggregate["dataset_means"]), set(analysis.DATASETS))
            self.assertIn("sample_sd", aggregate["macro"]["S1+M_logits"])
            self.assertEqual(len(aggregate["complete_macro_metrics"]), 24)
            self.assertIn("multiclass_brier", aggregate["complete_macro_metrics"]["T1"])
            self.assertEqual(aggregate["deltas_pp"]["candidate_vs_C_probability_macro_pp"], 0.0)

    def test_aggregate_recomputes_complete_analysis_from_referenced_exports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inventory, inventory_sha, cases = _fixture_inventory(root, _guard_payload)
            trust = {"trusted_inventory_path": inventory,
                     "trusted_inventory_sha256": inventory_sha}
            reports = []
            for dataset in analysis.DATASETS:
                for seed in (1, 2, 3):
                    export = root / f"export-{dataset}-{seed}"
                    probe.write_export_artifact(export, _guard_payload(dataset, seed), {
                        "dataset_key": dataset, "seed": seed, "protocol": probe.PROTOCOL,
                        "provenance": _provenance(dataset, seed, cases, inventory_sha),
                    })
                    reports.append(analysis.analyze_export(
                        export / "export.json", root / f"analysis-{dataset}-{seed}", **trust))
            analysis.aggregate_reports(reports, **trust)

            changed = copy.deepcopy(reports)
            changed[0]["combinations"]["S1+M_logits"]["accuracy_pct"] = 99.5
            with self.assertRaisesRegex(ValueError, "analysis integrity"):
                analysis.aggregate_reports(changed, **trust)

            nonfinite = copy.deepcopy(reports)
            nonfinite[0]["branches"]["C"]["nll"] = float("nan")
            with self.assertRaisesRegex(ValueError, "analysis integrity"):
                analysis.aggregate_reports(nonfinite, **trust)

            changed_count = copy.deepcopy(reports)
            changed_count[0]["branches"]["C"]["correct"] -= 1
            with self.assertRaisesRegex(ValueError, "analysis integrity"):
                analysis.aggregate_reports(changed_count, **trust)

            changed_prediction = copy.deepcopy(reports)
            changed_prediction[0]["predictions"]["C"][0] = 2
            with self.assertRaisesRegex(ValueError, "analysis integrity"):
                analysis.aggregate_reports(changed_prediction, **trust)

            missing_export = copy.deepcopy(reports)
            missing_export[0]["export"] = str(root / "missing-export.json")
            with self.assertRaisesRegex(ValueError, "referenced export"):
                analysis.aggregate_reports(missing_export, **trust)

            bad_hash = copy.deepcopy(reports)
            bad_hash[0]["export_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "referenced export"):
                analysis.aggregate_reports(bad_hash, **trust)

            missing_schema = copy.deepcopy(reports)
            del missing_schema[0]["schema_version"]
            with self.assertRaisesRegex(ValueError, "analysis schema"):
                analysis.aggregate_reports(missing_schema, **trust)

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
        uniformly_wrong = []
        for dataset in analysis.DATASETS:
            for seed in (1, 2, 3):
                case = copy.deepcopy(template); case["dataset_key"] = dataset; case["seed"] = seed
                case["provenance"] = _provenance(dataset, seed)
                case["provenance"]["clip_sha256"] = "a" * 64
                case["provenance"]["template_sha256"] = "b" * 64
                case["provenance"]["inventory_sha256"] = "c" * 64
                case["provenance"]["checkpoint_sha256"] = {
                    role: f"uniformly-wrong-{dataset}-{seed}-{role}" for role in probe.ROLES
                }
                case["provenance"]["manifest_sha256"] = {
                    role: f"uniformly-wrong-manifest-{dataset}-{seed}-{role}" for role in probe.ROLES
                }
                uniformly_wrong.append(case)
        with self.assertRaisesRegex(ValueError, "trusted inventory"):
            analysis.aggregate_reports(uniformly_wrong)


if __name__ == "__main__":
    unittest.main()
