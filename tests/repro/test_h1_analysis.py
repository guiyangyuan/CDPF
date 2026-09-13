import copy
import unittest

import numpy as np

from scripts.repro import h1_analysis as analysis


def current_receipt(dataset="dtd", mode="weighted_nckd"):
    return {
        "dataset_key": dataset,
        "seed": 1,
        "mode": mode,
        "protocol": {"split": "val", "subsample_classes": "base", "seed": 1},
        "rows": [
            {"row": 0, "sample_id": "a", "label": 0, "classname": "zero"},
            {"row": 1, "sample_id": "b", "label": 1, "classname": "one"},
            {"row": 2, "sample_id": "c", "label": 0, "classname": "zero"},
        ],
        "class_order": ["zero", "one"],
        "checkpoint": {"checkpoint_sha256": "1" * 64},
        "source_sha256": "2" * 64,
    }


def current_arrays():
    s = np.array([[4, 0], [0, 4], [4, 0]], dtype=np.float32)
    t = np.array([[3, 0], [0, 3], [3, 0]], dtype=np.float32)
    m = np.array([[2, 0], [0, 2], [0, 2]], dtype=np.float32)
    return {
        "labels": np.array([0, 1, 0], dtype=np.int64),
        "S": s,
        "T": t,
        "M": m,
        "ensemble": np.float32(0.5) * (s + t),
    }


def historical_arrays():
    values = current_arrays()
    return {
        "labels": values["labels"].copy(),
        "C": np.array([[0, 2], [2, 0], [0, 2]], dtype=np.float32),
        "S0": np.array([[0, 2], [2, 0], [0, 2]], dtype=np.float32),
        "S1": values["S"].copy(),
        "T0": values["T"].copy(),
        "T1": values["T"].copy(),
        "M": values["M"].copy(),
    }


class AnchorAuditTests(unittest.TestCase):
    def test_anchor_independently_matches_all_frozen_branches_and_diagnostics(self):
        receipt = current_receipt(mode="full_kl")
        result = analysis.compute_anchor_audit(
            receipt, current_arrays(), receipt, historical_arrays())

        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["guards"]["all_passed"])
        self.assertEqual(set(result["metrics"]), {"S", "T", "M", "ensemble", "S+M_logits"})
        self.assertEqual(result["comparisons"]["S"]["reference_branch"], "S1")
        self.assertEqual(result["comparisons"]["T"]["reference_branch"], "T1")
        self.assertEqual(result["comparisons"]["M"]["max_abs_error"], 0.0)

    def test_anchor_stops_when_logits_exceed_frozen_tolerance(self):
        changed = current_arrays()
        changed["S"] = changed["S"].copy()
        changed["S"][0, 0] += np.float32(0.001)
        changed["ensemble"] = np.float32(0.5) * (changed["S"] + changed["T"])

        result = analysis.compute_anchor_audit(
            current_receipt(mode="full_kl"), changed,
            current_receipt(mode="full_kl"), historical_arrays())

        self.assertEqual(result["status"], "stopped")
        self.assertFalse(result["guards"]["S_logits_allclose"])

    def test_anchor_rejects_nonfinite_or_misaligned_inputs(self):
        bad = current_arrays()
        bad["M"] = bad["M"].copy(); bad["M"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            analysis.compute_anchor_audit(current_receipt(mode="full_kl"), bad,
                                          current_receipt(mode="full_kl"), historical_arrays())
        wrong = historical_arrays(); wrong["labels"] = np.array([1, 1, 0], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "labels|rows"):
            analysis.compute_anchor_audit(current_receipt(mode="full_kl"), current_arrays(),
                                          current_receipt(mode="full_kl"), wrong)


class CandidateAnalysisTests(unittest.TestCase):
    def test_candidate_preserves_raw_metrics_and_fixed_historical_comparators(self):
        receipt = current_receipt("ucf101")
        result = analysis.compute_candidate_analysis(
            receipt, current_arrays(), receipt, historical_arrays())

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["candidate"], "S+M_logits")
        self.assertEqual(set(result["current_metrics"]),
                         {"S", "T", "M", "ensemble", "S+M_logits"})
        self.assertEqual(set(result["comparators"]),
                         {"C+M_logits", "C+M_probability", "S0+M_logits"})
        self.assertEqual(result["current_metrics"]["S+M_logits"]["accuracy_pct"], 100.0)
        self.assertAlmostEqual(result["comparators"]["C+M_logits"]["accuracy_pct"],
                               100.0 / 3.0)
        self.assertTrue(result["description_baseline_guard"]["all_passed"])

    def test_candidate_rejects_changed_historical_description_baseline(self):
        historical = historical_arrays(); historical["M"] = historical["M"].copy()
        historical["M"][0, 0] += np.float32(0.01)
        with self.assertRaisesRegex(ValueError, "description baseline"):
            analysis.compute_candidate_analysis(
                current_receipt(), current_arrays(), current_receipt(), historical)

    def test_candidate_rejects_wrong_mode_rows_classes_and_partial_arrays(self):
        with self.assertRaisesRegex(ValueError, "weighted_nckd"):
            analysis.compute_candidate_analysis(
                current_receipt(mode="full_kl"), current_arrays(),
                current_receipt(mode="full_kl"), historical_arrays())
        wrong = current_receipt(); wrong["class_order"] = ["one", "zero"]
        with self.assertRaisesRegex(ValueError, "rows/classes"):
            analysis.compute_candidate_analysis(
                current_receipt(), current_arrays(), wrong, historical_arrays())
        partial = current_arrays(); partial.pop("T")
        with self.assertRaisesRegex(ValueError, "exactly"):
            analysis.compute_candidate_analysis(
                current_receipt(), partial, current_receipt(), historical_arrays())


class AggregateTests(unittest.TestCase):
    def case(self, dataset, candidate, c_logits, c_probability, s0):
        result = analysis.compute_candidate_analysis(
            current_receipt(dataset), current_arrays(), current_receipt(dataset), historical_arrays())
        for metrics, name, value in (
                (result["current_metrics"], "S+M_logits", candidate),
                (result["comparators"], "C+M_logits", c_logits),
                (result["comparators"], "C+M_probability", c_probability),
                (result["comparators"], "S0+M_logits", s0)):
            metrics[name].update({"correct": int(round(value * 10)), "total": 1000,
                                  "accuracy_pct": float(value)})
        return result

    def set_counts(self, report, method, correct, total):
        metrics = (report["current_metrics"] if method == "S+M_logits"
                   else report["comparators"])
        metrics[method].update({"correct": correct, "total": total,
                                "accuracy_pct": 100.0 * correct / total})

    def test_fixed_three_dataset_equal_weight_aggregation_passes(self):
        reports = [
            self.case("ucf101", 60.0, 60.5, 60.2, 58.0),
            self.case("dtd", 70.0, 69.0, 69.5, 68.0),
            self.case("fgvc_aircraft", 80.0, 79.0, 79.5, 78.0),
        ]
        result = analysis.aggregate_candidates(reports)
        self.assertEqual(result["status"], "passed")
        self.assertAlmostEqual(result["macro_accuracy_pct"]["S+M_logits"], 70.0)
        self.assertAlmostEqual(result["deltas_pp"]["vs_C+M_logits"]["ucf101"], -0.5)
        self.assertTrue(result["guards"]["all_passed"])
        self.assertEqual(result["weighting"], "equal weight across exactly three datasets")

    def test_aggregate_stops_on_any_fixed_guard_failure(self):
        reports = [
            self.case("ucf101", 58.0, 60.0, 60.0, 57.0),
            self.case("dtd", 70.0, 69.0, 69.0, 68.0),
            self.case("fgvc_aircraft", 80.0, 79.0, 79.0, 78.0),
        ]
        result = analysis.aggregate_candidates(reports)
        self.assertEqual(result["status"], "stopped")
        self.assertFalse(result["guards"]["vs_C+M_logits"])

    def test_strict_macro_guard_treats_exact_fractional_tie_as_not_positive(self):
        reports = [self.case(name, 50, 0, 0, 0)
                   for name in ("ucf101", "dtd", "fgvc_aircraft")]
        totals = (204, 96, 200)
        for report, total, candidate, baseline in zip(
                reports, totals, (18, 6, 55), (1, 2, 80)):
            self.set_counts(report, "S+M_logits", candidate, total)
            self.set_counts(report, "C+M_logits", 0, total)
            self.set_counts(report, "C+M_probability", 0, total)
            self.set_counts(report, "S0+M_logits", baseline, total)

        result = analysis.aggregate_candidates(reports)

        self.assertEqual(result["status"], "stopped")
        self.assertFalse(result["guards"]["vs_S0+M_logits"])

    def test_aggregate_rejects_accuracy_inconsistent_with_integer_counts(self):
        reports = [self.case(name, 70, 69, 69, 68)
                   for name in ("ucf101", "dtd", "fgvc_aircraft")]
        reports[0]["current_metrics"]["S+M_logits"]["accuracy_pct"] = 70.0001
        with self.assertRaisesRegex(ValueError, "correct/total"):
            analysis.aggregate_candidates(reports)

    def test_aggregate_refuses_partial_duplicate_or_mixed_reports(self):
        base = [self.case(name, 70, 69, 69, 68)
                for name in ("ucf101", "dtd", "fgvc_aircraft")]
        for reports in (base[:2], [base[0], base[0], base[2]]):
            with self.subTest(cases=[r["dataset_key"] for r in reports]):
                with self.assertRaisesRegex(ValueError, "exactly three|once"):
                    analysis.aggregate_candidates(reports)
        mixed = copy.deepcopy(base); mixed[2]["seed"] = 2
        with self.assertRaisesRegex(ValueError, "seed1"):
            analysis.aggregate_candidates(mixed)


if __name__ == "__main__":
    unittest.main()
