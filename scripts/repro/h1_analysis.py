#!/usr/bin/env python3
"""Audit and aggregate the fixed H1 seed-1 Base-validation study."""

from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.repro import h1_validation
from scripts.repro import validation_analysis as historical_analysis
from scripts.repro import validation_probe as probe


DATASETS = ("ucf101", "dtd", "fgvc_aircraft")
CURRENT_BRANCHES = ("S", "T", "M", "ensemble")
HISTORICAL_BRANCHES = ("C", "S0", "S1", "M", "T0", "T1")
RTOL = 1e-5
ATOL = 1e-4


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _fresh_output(path: Path) -> Path:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"Output must be a fresh path: {target}")
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_case_inputs(
        current_receipt: Mapping[str, Any], current: Mapping[str, np.ndarray],
        historical_receipt: Mapping[str, Any], historical: Mapping[str, np.ndarray],
        *, expected_mode: str,
) -> tuple[str, np.ndarray]:
    dataset = current_receipt.get("dataset_key")
    if dataset not in DATASETS or current_receipt.get("seed") != 1:
        raise ValueError("H1 analysis requires one fixed seed1 dataset case")
    if current_receipt.get("mode") != expected_mode:
        raise ValueError(f"H1 case requires {expected_mode} mode")
    if historical_receipt.get("dataset_key") != dataset or historical_receipt.get("seed") != 1:
        raise ValueError("Historical export dataset/seed differs from the current case")
    if current_receipt.get("rows") != historical_receipt.get("rows") or \
            current_receipt.get("class_order") != historical_receipt.get("class_order"):
        raise ValueError("Current and historical rows/classes are not exactly aligned")
    if set(current) != {"labels", *CURRENT_BRANCHES}:
        raise ValueError("Current export must contain exactly labels/S/T/M/ensemble")
    if set(historical) != {"labels", *HISTORICAL_BRANCHES}:
        raise ValueError("Historical export must contain exactly its six branches and labels")
    labels = np.asarray(current["labels"])
    if labels.dtype != np.int64 or not np.array_equal(labels, historical["labels"]):
        raise ValueError("Current and historical labels are not exactly aligned")
    n, c = len(labels), len(current_receipt.get("class_order", []))
    if n == 0 or c < 2:
        raise ValueError("H1 analysis requires nonempty multiclass Base validation arrays")
    for name in CURRENT_BRANCHES:
        value = np.asarray(current[name])
        if value.dtype != np.float32 or value.shape != (n, c) or not np.isfinite(value).all():
            raise ValueError(f"Current {name} must be a finite float32 [N,C] array")
    for name in HISTORICAL_BRANCHES:
        value = np.asarray(historical[name])
        if value.dtype != np.float32 or value.shape != (n, c) or not np.isfinite(value).all():
            raise ValueError(f"Historical {name} must be a finite float32 [N,C] array")
    expected_ensemble = np.float32(0.5) * (current["S"] + current["T"])
    if not np.array_equal(current["ensemble"], expected_ensemble):
        raise ValueError("Current ensemble is not the fixed .5*(S+T) diagnostic")
    return dataset, labels


def _metrics(current: Mapping[str, np.ndarray], labels: np.ndarray) -> dict[str, Any]:
    result = {name: historical_analysis.branch_metrics(current[name], labels)
              for name in CURRENT_BRANCHES}
    result["S+M_logits"] = historical_analysis.branch_metrics(
        np.float64(0.5) * (current["S"].astype(np.float64) + current["M"].astype(np.float64)),
        labels)
    return result


def _comparison(current: np.ndarray, reference: np.ndarray, reference_branch: str) -> dict[str, Any]:
    difference = np.abs(current.astype(np.float64) - reference.astype(np.float64))
    prediction_equal = bool(np.array_equal(np.argmax(current, axis=1),
                                           np.argmax(reference, axis=1)))
    logits_allclose = bool(np.allclose(current, reference, rtol=RTOL, atol=ATOL))
    return {
        "reference_branch": reference_branch,
        "prediction_exact_equal": prediction_equal,
        "logits_allclose": logits_allclose,
        "rtol": RTOL,
        "atol": ATOL,
        "max_abs_error": float(np.max(difference)),
    }


def compute_anchor_audit(
        current_receipt: Mapping[str, Any], current: Mapping[str, np.ndarray],
        historical_receipt: Mapping[str, Any], historical: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    dataset, labels = _validate_case_inputs(
        current_receipt, current, historical_receipt, historical, expected_mode="full_kl")
    if dataset != "dtd":
        raise ValueError("The fixed H1 anchor is DTD/full_kl only")
    branches = {"S": "S1", "T": "T1", "M": "M"}
    comparisons = {name: _comparison(current[name], historical[reference], reference)
                   for name, reference in branches.items()}
    guards = {}
    for name, value in comparisons.items():
        guards[f"{name}_predictions_exact"] = value["prediction_exact_equal"]
        guards[f"{name}_logits_allclose"] = value["logits_allclose"]
    guards["all_passed"] = all(guards.values())
    return {
        "status": "passed" if guards["all_passed"] else "stopped",
        "schema_version": 1,
        "analysis_kind": "anchor_audit",
        "dataset_key": dataset,
        "seed": 1,
        "mode": "full_kl",
        "fixed_tolerance": {"rtol": RTOL, "atol": ATOL},
        "metrics": _metrics(current, labels),
        "comparisons": comparisons,
        "guards": guards,
        "claims": {"split": "Base validation only", "novel_or_hm": False,
                   "hypothesis_proven": False, "automatic_training_action": False},
    }


def compute_candidate_analysis(
        current_receipt: Mapping[str, Any], current: Mapping[str, np.ndarray],
        historical_receipt: Mapping[str, Any], historical: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    dataset, labels = _validate_case_inputs(
        current_receipt, current, historical_receipt, historical,
        expected_mode="weighted_nckd")
    description = _comparison(current["M"], historical["M"], "M")
    if not description["prediction_exact_equal"] or not description["logits_allclose"]:
        raise ValueError("Current description baseline differs from authenticated historical M")
    current_metrics = _metrics(current, labels)
    c_logits = np.float64(0.5) * (
        historical["C"].astype(np.float64) + historical["M"].astype(np.float64))
    s0_logits = np.float64(0.5) * (
        historical["S0"].astype(np.float64) + historical["M"].astype(np.float64))
    comparators = {
        "C+M_logits": historical_analysis.branch_metrics(c_logits, labels),
        "C+M_probability": historical_analysis.fixed_probability_fusion_metrics(
            historical["C"], historical["M"], labels)["metrics"],
        "S0+M_logits": historical_analysis.branch_metrics(s0_logits, labels),
    }
    return {
        "status": "passed",
        "schema_version": 1,
        "analysis_kind": "candidate",
        "dataset_key": dataset,
        "seed": 1,
        "mode": "weighted_nckd",
        "candidate": "S+M_logits",
        "fixed_alpha": 0.5,
        "current_metrics": current_metrics,
        "comparators": comparators,
        "description_baseline_guard": {**description, "all_passed": True},
        "claims": {"split": "Base validation only", "novel_or_hm": False,
                   "hypothesis_proven": False, "automatic_training_action": False},
    }


def aggregate_candidates(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(reports) != 3:
        raise ValueError("H1 aggregate requires exactly three candidate reports")
    by_dataset = {report.get("dataset_key"): report for report in reports}
    if set(by_dataset) != set(DATASETS) or len(by_dataset) != len(reports):
        raise ValueError("Each fixed candidate dataset must occur once")
    for dataset in DATASETS:
        report = by_dataset[dataset]
        if report.get("status") != "passed" or report.get("analysis_kind") != "candidate" or \
                report.get("seed") != 1:
            raise ValueError("Aggregate accepts only complete seed1 candidate reports")
        if report.get("mode") != "weighted_nckd" or report.get("candidate") != "S+M_logits":
            raise ValueError("Aggregate candidate mode/formula is mixed")
        if set(report.get("current_metrics", {})) != {"S", "T", "M", "ensemble", "S+M_logits"} or \
                set(report.get("comparators", {})) != {
                    "C+M_logits", "C+M_probability", "S0+M_logits"}:
            raise ValueError("Candidate report is partial or has extra methods")
    methods = ("S+M_logits", "C+M_logits", "C+M_probability", "S0+M_logits")
    dataset_accuracy = {}
    exact_accuracy: dict[str, dict[str, Fraction]] = {}
    for dataset in DATASETS:
        report = by_dataset[dataset]
        metric_records = {
            "S+M_logits": report["current_metrics"]["S+M_logits"],
            **{name: report["comparators"][name] for name in methods[1:]},
        }
        exact_accuracy[dataset] = {}
        totals = set()
        for name, metric in metric_records.items():
            correct, total, accuracy = metric.get("correct"), metric.get("total"), \
                metric.get("accuracy_pct")
            if isinstance(correct, bool) or isinstance(total, bool) or \
                    not isinstance(correct, int) or not isinstance(total, int) or \
                    total <= 0 or correct < 0 or correct > total:
                raise ValueError("Aggregate metrics require valid integer correct/total counts")
            expected_accuracy = 100.0 * correct / total
            if not isinstance(accuracy, (int, float)) or not np.isfinite(accuracy) or \
                    float(accuracy) != expected_accuracy:
                raise ValueError("Aggregate accuracy is inconsistent with correct/total counts")
            totals.add(total)
            exact_accuracy[dataset][name] = Fraction(100 * correct, total)
        if len(totals) != 1:
            raise ValueError("Aggregate methods do not use the same dataset total")
        dataset_accuracy[dataset] = {
            name: float(exact_accuracy[dataset][name]) for name in methods}
    exact_macro = {
        method: sum((exact_accuracy[d][method] for d in DATASETS), Fraction(0, 1)) / 3
        for method in methods}
    macro = {method: float(value) for method, value in exact_macro.items()}
    exact_deltas = {
        f"vs_{comparison}": {
            dataset: exact_accuracy[dataset]["S+M_logits"] -
                     exact_accuracy[dataset][comparison] for dataset in DATASETS}
        for comparison in methods[1:]
    }
    for name, values in exact_deltas.items():
        values["macro"] = exact_macro["S+M_logits"] - exact_macro[name.removeprefix("vs_")]
    deltas = {name: {key: float(value) for key, value in values.items()}
              for name, values in exact_deltas.items()}
    guards = {
        "vs_C+M_logits": exact_deltas["vs_C+M_logits"]["macro"] >= 0 and
                         min(exact_deltas["vs_C+M_logits"][d] for d in DATASETS) >= -1,
        "vs_C+M_probability": exact_deltas["vs_C+M_probability"]["macro"] >= 0 and
                              min(exact_deltas["vs_C+M_probability"][d]
                                  for d in DATASETS) >= -1,
        "vs_S0+M_logits": exact_deltas["vs_S0+M_logits"]["macro"] > 0,
    }
    guards["all_passed"] = all(guards.values())
    return {
        "status": "passed" if guards["all_passed"] else "stopped",
        "schema_version": 1,
        "analysis_kind": "aggregate",
        "case_count": 3,
        "seed": 1,
        "candidate": "S+M_logits",
        "weighting": "equal weight across exactly three datasets",
        "dataset_accuracy_pct": dataset_accuracy,
        "macro_accuracy_pct": macro,
        "deltas_pp": deltas,
        "exact_guard_arithmetic_pct": {
            "macro": {name: str(value) for name, value in exact_macro.items()},
            "deltas": {name: {key: str(value) for key, value in values.items()}
                       for name, values in exact_deltas.items()},
        },
        "guards": guards,
        "cases": [copy.deepcopy(by_dataset[dataset]) for dataset in DATASETS],
        "claims": {"split": "Base validation only", "novel_or_hm": False,
                   "hypothesis_proven": False, "exploratory_only": True},
    }


def _write_case(
        kind: str, export_path: Path, historical_path: Path, output_dir: Path,
) -> dict[str, Any]:
    output = _fresh_output(output_dir)
    current_receipt, current = h1_validation.load_export(export_path)
    historical_receipt, historical = historical_analysis.load_export(historical_path)
    result = (compute_anchor_audit if kind == "anchor" else compute_candidate_analysis)(
        current_receipt, current, historical_receipt, historical)
    result.update({
        "current_export": str(Path(export_path).resolve()),
        "current_export_sha256": _sha256(Path(export_path)),
        "historical_export": str(Path(historical_path).resolve()),
        "historical_export_sha256": _sha256(Path(historical_path)),
    })
    output.mkdir(parents=True)
    _atomic_json(output / "analysis.json", result)
    return result


def run_anchor_audit(export_path: Path, historical_path: Path, output_dir: Path) -> dict[str, Any]:
    return _write_case("anchor", export_path, historical_path, output_dir)


def run_candidate_analysis(export_path: Path, historical_path: Path,
                           output_dir: Path) -> dict[str, Any]:
    return _write_case("candidate", export_path, historical_path, output_dir)


def _reload_candidate(path: Path) -> dict[str, Any]:
    report_path = Path(path).resolve()
    saved = json.loads(report_path.read_text())
    if saved.get("analysis_kind") != "candidate":
        raise ValueError("Aggregate input is not a candidate analysis")
    export_path, historical_path = Path(saved["current_export"]), Path(saved["historical_export"])
    if _sha256(export_path) != saved.get("current_export_sha256") or \
            _sha256(historical_path) != saved.get("historical_export_sha256"):
        raise ValueError("Candidate analysis input export changed")
    current_receipt, current = h1_validation.load_export(export_path)
    historical_receipt, historical = historical_analysis.load_export(historical_path)
    expected = compute_candidate_analysis(current_receipt, current, historical_receipt, historical)
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError("Candidate analysis differs from reloaded authentic exports")
    return saved


def run_aggregate(analysis_paths: Sequence[Path], output_dir: Path) -> dict[str, Any]:
    output = _fresh_output(output_dir)
    reports = [_reload_candidate(path) for path in analysis_paths]
    result = aggregate_candidates(reports)
    result["analysis_inputs"] = [
        {"path": str(Path(path).resolve()), "sha256": _sha256(Path(path))}
        for path in analysis_paths]
    output.mkdir(parents=True)
    _atomic_json(output / "aggregate.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("anchor", "candidate"):
        command = commands.add_parser(name)
        command.add_argument("--export", type=Path, required=True)
        command.add_argument("--historical", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--analysis", type=Path, nargs=3, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "anchor":
        result = run_anchor_audit(args.export, args.historical, args.output_dir)
    elif args.command == "candidate":
        result = run_candidate_analysis(args.export, args.historical, args.output_dir)
    else:
        result = run_aggregate(args.analysis, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
