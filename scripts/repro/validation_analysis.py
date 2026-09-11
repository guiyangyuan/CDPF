#!/usr/bin/env python3
"""Analyze and aggregate frozen validation-only crossed-inference exports."""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    from scripts.repro import validation_probe as probe
except ImportError:
    import validation_probe as probe


DATASETS = ("ucf101", "dtd", "fgvc_aircraft")
EXPECTED_SOURCES = probe.EXPECTED_SOURCES
EXPECTED_TEMPLATES = {
    "ucf101": "3acc805cbcd115331cc975f6146b24de2fc1732c45ee54a57c25259facbc6b9a",
    "dtd": "dabdcafded7babd67046d0f90abeb510b648bdaaf397b884f265f3460a7942a9",
    "fgvc_aircraft": "d86833c2483c1d12907abeb6198412f1d0a020bd12999a3f170967a7f549e3b8",
}
STUDENTS = ("C", "S0", "S1")
TEACHERS = ("M", "T0", "T1")
FIXED_INVENTORY_PATH = (Path(__file__).resolve().parents[2] /
    "reports/diagnostics/error-analysis-20260911/evidence/proposed-probe-inputs.json")


def _fresh_output(path: Path) -> Path:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"Output must be a fresh path: {target}")
    return target


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _validate_logits(logits: np.ndarray, labels: np.ndarray) -> tuple[int, int]:
    value = np.asarray(logits)
    targets = np.asarray(labels)
    if value.ndim != 2 or targets.ndim != 1 or value.shape[0] != targets.shape[0]:
        raise ValueError("Logits/labels require aligned [N,C]/[N] shapes")
    if value.shape[0] == 0 or value.shape[1] < 2 or not np.isfinite(value).all():
        raise ValueError("Logits must be a nonempty finite multiclass matrix")
    if not np.issubdtype(targets.dtype, np.integer) or np.any(targets < 0) or np.any(targets >= value.shape[1]):
        raise ValueError("Labels are invalid for the logits class order")
    return value.shape


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    shifted = value - np.max(value, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise ValueError("Softmax produced non-finite probabilities")
    return probabilities


def stable_log_softmax(logits: np.ndarray) -> np.ndarray:
    value = np.asarray(logits, dtype=np.float64)
    maximum = np.max(value, axis=1, keepdims=True)
    return value - maximum - np.log(np.exp(value - maximum).sum(axis=1, keepdims=True))


def probability_metrics(probabilities: np.ndarray, labels: np.ndarray,
                        log_probabilities: np.ndarray | None = None) -> dict[str, Any]:
    probs, targets = np.asarray(probabilities, dtype=np.float64), np.asarray(labels)
    _validate_logits(probs, targets)
    if np.any(probs < 0) or not np.allclose(probs.sum(axis=1), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Probability rows must be nonnegative and sum to one")
    predictions = np.argmax(probs, axis=1); correct = int(np.sum(predictions == targets)); n = len(targets)
    if log_probabilities is None:
        true_probability = probs[np.arange(n), targets]
        nll = float("inf") if np.any(true_probability <= 0) else float(-np.log(true_probability).mean())
    else:
        log_probs = np.asarray(log_probabilities, dtype=np.float64)
        if log_probs.shape != probs.shape or not np.isfinite(log_probs).all():
            raise ValueError("Log probabilities must be finite and aligned")
        nll = float(-log_probs[np.arange(n), targets].mean())
    one_hot = np.zeros_like(probs); one_hot[np.arange(n), targets] = 1.0
    sorted_probs = np.sort(probs, axis=1)
    return {"correct": correct, "total": n, "accuracy_pct": 100.0 * correct / n,
            "nll": nll, "multiclass_brier": float(np.square(probs - one_hot).sum(axis=1).mean()),
            "probability_top1_top2_margin": float((sorted_probs[:, -1] - sorted_probs[:, -2]).mean())}


def branch_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    _validate_logits(logits, labels)
    value = np.asarray(logits, dtype=np.float64)
    return probability_metrics(stable_softmax(value), labels, stable_log_softmax(value))


def fixed_probability_fusion_metrics(student_logits: np.ndarray, teacher_logits: np.ndarray,
                                     labels: np.ndarray) -> dict[str, Any]:
    student_log = stable_log_softmax(student_logits); teacher_log = stable_log_softmax(teacher_logits)
    fused_log = np.logaddexp(student_log, teacher_log) - math.log(2.0)
    fused_probability = 0.5 * np.exp(student_log) + 0.5 * np.exp(teacher_log)
    return {"metrics": probability_metrics(fused_probability, labels, fused_log),
            "probabilities": fused_probability}


def _trusted_inventory_cases() -> dict[tuple[str, int], dict[str, Any]]:
    if probe.sha256_file(FIXED_INVENTORY_PATH) != probe.FIXED_INVENTORY_SHA256:
        raise ValueError("Authenticated fixed inventory hash mismatch")
    inventory = json.loads(FIXED_INVENTORY_PATH.read_text())
    cases = {}
    for dataset in DATASETS:
        for seed in (1, 2, 3):
            cases[(dataset, seed)] = probe.validate_inventory(inventory, dataset, seed)
    return cases


def _validate_trusted_provenance(dataset: str, seed: int,
                                 provenance: Mapping[str, Any]) -> dict[str, Any]:
    case = _trusted_inventory_cases()[(dataset, seed)]
    expected = {
        "source_sha256": EXPECTED_SOURCES[dataset],
        "clip_sha256": probe.CLIP_SHA256,
        "template_sha256": EXPECTED_TEMPLATES[dataset],
        "inventory_sha256": probe.FIXED_INVENTORY_SHA256,
        "inventory_case_sha256": probe.json_digest(case),
        "train_samples_sha256": case["train_samples_sha256"],
        "val_samples_sha256": case["val_samples_sha256"],
        "checkpoint_sha256": {role: case["checkpoints"][role]["sha256"] for role in probe.ROLES},
        "manifest_sha256": {role: case["checkpoints"][role]["manifest_sha256"] for role in probe.ROLES},
    }
    differing = sorted(key for key, value in expected.items() if provenance.get(key) != value)
    if differing:
        raise ValueError(f"Artifact settings are not compatible with trusted inventory: {differing}")
    for field in ("config_sha256", "config_compatibility_sha256"):
        value = provenance.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Artifact provenance differs from trusted inventory chain: {field}")
    return case


def load_export(export_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(export_path).resolve()
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "passed" or receipt.get("schema_version") != 1:
        raise ValueError("Export status/schema is not complete")
    if receipt.get("protocol") != probe.PROTOCOL or set(receipt.get("branches", [])) != set(probe.BRANCHES):
        raise ValueError("Export protocol/branch schema mismatch")
    dataset, seed = receipt.get("dataset_key"), receipt.get("seed")
    if dataset not in DATASETS or seed not in (1, 2, 3):
        raise ValueError("Export dataset/seed is outside the fixed nine cases")
    provenance = receipt.get("provenance", {})
    _validate_trusted_provenance(dataset, seed, provenance)
    if receipt.get("prepared") is not None and \
            probe.sha256_file(Path(receipt["prepared"])) != receipt.get("prepared_sha256"):
        raise ValueError("Export prepared-receipt hash mismatch")
    npz = Path(receipt["npz"])
    if not npz.is_absolute():
        npz = path.parent / npz
    if probe.sha256_file(npz) != receipt.get("npz_sha256"):
        raise ValueError("Export NPZ hash mismatch")
    with np.load(npz, allow_pickle=False) as archive:
        if set(archive.files) != {"labels", *probe.BRANCHES}:
            raise ValueError("Export NPZ has missing or extra arrays")
        arrays = {name: archive[name].copy() for name in archive.files}
    labels = arrays.pop("labels")
    if labels.dtype != np.int64:
        raise ValueError("Export labels must be int64")
    classes, rows = receipt.get("class_order", []), receipt.get("rows", [])
    if not classes or len(set(classes)) != len(classes) or len(rows) != len(labels):
        raise ValueError("Export class/sample alignment metadata is invalid")
    if [row.get("row") for row in rows] != list(range(len(rows))) or \
            [row.get("label") for row in rows] != labels.tolist() or \
            any(row.get("classname") != classes[int(row["label"])] for row in rows):
        raise ValueError("Export sample alignment differs from NPZ labels")
    if len({row.get("sample_id") for row in rows}) != len(rows):
        raise ValueError("Export sample IDs are duplicated")
    for name in probe.BRANCHES:
        if arrays[name].dtype != np.float32 or arrays[name].shape != (len(labels), len(classes)):
            raise ValueError(f"Export {name} dtype/shape mismatch")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Export {name} contains non-finite logits")
    arrays["labels"] = labels
    return receipt, arrays


def _pair_audit(student_prediction: np.ndarray, teacher_prediction: np.ndarray,
                labels: np.ndarray) -> dict[str, int]:
    student_correct, teacher_correct = student_prediction == labels, teacher_prediction == labels
    return {"prediction_disagreement": int(np.sum(student_prediction != teacher_prediction)),
            "both_wrong_error_overlap": int(np.sum(~student_correct & ~teacher_correct)),
            "student_wrong_teacher_correct": int(np.sum(~student_correct & teacher_correct)),
            "student_correct_teacher_wrong": int(np.sum(student_correct & ~teacher_correct))}


def _paired_vs_c(candidate_prediction: np.ndarray, c_prediction: np.ndarray,
                 labels: np.ndarray) -> dict[str, int]:
    candidate_correct, c_correct = candidate_prediction == labels, c_prediction == labels
    return {"rescue_vs_C": int(np.sum(candidate_correct & ~c_correct)),
            "harm_vs_C": int(np.sum(~candidate_correct & c_correct)),
            "both_correct": int(np.sum(candidate_correct & c_correct)),
            "both_wrong": int(np.sum(~candidate_correct & ~c_correct)),
            "prediction_disagreement_vs_C": int(np.sum(candidate_prediction != c_prediction))}


def analyze_export(export_path: Path, output_dir: Path) -> dict[str, Any]:
    output = _fresh_output(output_dir)
    receipt, arrays = load_export(export_path)
    labels = arrays["labels"]
    branches, combinations, predictions, pair_audits, paired_vs_c = {}, {}, {}, {}, {}
    for name in probe.BRANCHES:
        branches[name] = branch_metrics(arrays[name], labels)
        predictions[name] = np.argmax(arrays[name], axis=1).astype(int).tolist()
    c_prediction = np.argmax(arrays["C"], axis=1)
    for student in STUDENTS:
        for teacher in TEACHERS:
            pair_audits[f"{student}_vs_{teacher}"] = _pair_audit(
                np.argmax(arrays[student], axis=1), np.argmax(arrays[teacher], axis=1), labels)
            logits_name = f"{student}+{teacher}_logits"
            fused_logits = 0.5 * arrays[student].astype(np.float64) + 0.5 * arrays[teacher].astype(np.float64)
            combinations[logits_name] = branch_metrics(fused_logits, labels)
            logits_prediction = np.argmax(fused_logits, axis=1)
            predictions[logits_name] = logits_prediction.astype(int).tolist()
            paired_vs_c[logits_name] = _paired_vs_c(logits_prediction, c_prediction, labels)
            probability_name = f"{student}+{teacher}_probability"
            probability_result = fixed_probability_fusion_metrics(arrays[student], arrays[teacher], labels)
            fused_probability = probability_result["probabilities"]
            combinations[probability_name] = probability_result["metrics"]
            probability_prediction = np.argmax(fused_probability, axis=1)
            predictions[probability_name] = probability_prediction.astype(int).tolist()
            paired_vs_c[probability_name] = _paired_vs_c(probability_prediction, c_prediction, labels)
    report = {"status": "passed", "schema_version": 1, "dataset_key": receipt["dataset_key"],
              "seed": receipt["seed"], "protocol": receipt["protocol"],
              "provenance": copy.deepcopy(receipt.get("provenance", {})),
              "export": str(Path(export_path).resolve()), "export_sha256": probe.sha256_file(Path(export_path)),
              "sample_ids": [row["sample_id"] for row in receipt["rows"]],
              "labels": labels.astype(int).tolist(), "class_order": receipt["class_order"],
              "teacher_tensor_exact_equal": receipt["teacher_tensor_exact_equal"],
              "teacher_logits_exact_equal": receipt.get("teacher_logits_exact_equal"),
              "tool": {"validation_analysis_sha256": probe.sha256_file(Path(__file__).resolve())},
              "branches": branches, "combinations": combinations, "student_teacher_audits": pair_audits,
              "paired_comparisons_vs_C": paired_vs_c, "predictions": predictions,
              "claims": {"split": "Base validation only", "novel_or_hm": False,
                         "significance_test": False, "automatic_training_action": False}}
    output.mkdir(parents=True); _atomic_json(output / "analysis.json", report)
    return report


def _mean_sd(values: list[float]) -> dict[str, Any]:
    return {"values": values, "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None}


def aggregate_reports(reports: list[Mapping[str, Any]]) -> dict[str, Any]:
    if len(reports) != 9:
        raise ValueError("Aggregate requires exactly nine complete cases")
    expected = {(dataset, seed) for dataset in DATASETS for seed in (1, 2, 3)}
    keys = [(report.get("dataset_key"), report.get("seed")) for report in reports]
    if len(set(keys)) != len(keys):
        raise ValueError("Aggregate contains a duplicate dataset/seed case")
    if set(keys) != expected:
        raise ValueError("Aggregate is missing one or more fixed nine cases")
    if any(report.get("status") != "passed" or report.get("protocol") != probe.PROTOCOL for report in reports):
        raise ValueError("Aggregate contains incomplete or mixed protocol results")
    for report in reports:
        _validate_trusted_provenance(report["dataset_key"], report["seed"],
                                     report.get("provenance", {}))
    clips = {report.get("provenance", {}).get("clip_sha256") for report in reports}
    if clips != {probe.CLIP_SHA256}:
        raise ValueError("Aggregate settings are not compatible: CLIP differs")
    for dataset in DATASETS:
        group = [report for report in reports if report["dataset_key"] == dataset]
        source = {report["provenance"].get("source_sha256") for report in group}
        template = {report["provenance"].get("template_sha256") for report in group}
        compatibility = {report["provenance"].get("config_compatibility_sha256",
                                                  report["provenance"].get("config_sha256")) for report in group}
        if source != {EXPECTED_SOURCES[dataset]} or template != {EXPECTED_TEMPLATES[dataset]} or \
                len(compatibility) != 1:
            raise ValueError(f"Aggregate settings are not compatible for {dataset}")
    for role in probe.ROLES:
        hashes = [report["provenance"].get("checkpoint_sha256", {}).get(role) for report in reports]
        if any(not value for value in hashes) or len(set(hashes)) != 9:
            raise ValueError(f"Aggregate checkpoint provenance is missing or duplicated for {role}")
    selected = ("S1+M_logits", "C+M_logits", "S1+M_probability", "C+M_probability", "S0+M_logits")
    for report in reports:
        if any(name not in report.get("combinations", {}) for name in selected):
            raise ValueError("Aggregate report lacks a fixed guard combination")
    dataset_means = {}
    for dataset in DATASETS:
        group = sorted((r for r in reports if r["dataset_key"] == dataset), key=lambda r: r["seed"])
        dataset_means[dataset] = {name: _mean_sd([float(r["combinations"][name]["accuracy_pct"])
                                                  for r in group]) for name in selected}
    macro = {name: _mean_sd([dataset_means[dataset][name]["mean"] for dataset in DATASETS]) for name in selected}
    all_methods = list(probe.BRANCHES) + [f"{student}+{teacher}_{mode}"
        for student in STUDENTS for teacher in TEACHERS for mode in ("logits", "probability")]
    metric_fields = ("accuracy_pct", "nll", "multiclass_brier", "probability_top1_top2_margin")
    complete_dataset_metrics = {}
    for dataset in DATASETS:
        group = sorted((r for r in reports if r["dataset_key"] == dataset), key=lambda r: r["seed"])
        complete_dataset_metrics[dataset] = {}
        for method in all_methods:
            complete_dataset_metrics[dataset][method] = {}
            for field in metric_fields:
                values = [float((r["branches"] if method in probe.BRANCHES else r["combinations"])[method][field])
                          for r in group]
                complete_dataset_metrics[dataset][method][field] = _mean_sd(values)
    complete_macro_metrics = {}
    for method in all_methods:
        complete_macro_metrics[method] = {}
        for field in metric_fields:
            values = [complete_dataset_metrics[dataset][method][field]["mean"] for dataset in DATASETS]
            complete_macro_metrics[method][field] = _mean_sd(values)
    deltas = {
        "candidate_vs_C_logits_macro_pp": macro["S1+M_logits"]["mean"] - macro["C+M_logits"]["mean"],
        "candidate_vs_C_probability_macro_pp": macro["S1+M_logits"]["mean"] - macro["C+M_probability"]["mean"],
        "candidate_vs_S0_logits_macro_pp": macro["S1+M_logits"]["mean"] - macro["S0+M_logits"]["mean"],
        "candidate_vs_C_logits_dataset_pp": {dataset: dataset_means[dataset]["S1+M_logits"]["mean"] -
                                               dataset_means[dataset]["C+M_logits"]["mean"] for dataset in DATASETS},
        "candidate_vs_C_probability_dataset_pp": {dataset: dataset_means[dataset]["S1+M_logits"]["mean"] -
                                                    dataset_means[dataset]["C+M_probability"]["mean"] for dataset in DATASETS},
    }
    guards = {
        "S1+M_vs_C+M_logits": deltas["candidate_vs_C_logits_macro_pp"] >= 0 and
            min(deltas["candidate_vs_C_logits_dataset_pp"].values()) >= -1.0,
        "S1+M_vs_C+M_probability": deltas["candidate_vs_C_probability_macro_pp"] >= 0 and
            min(deltas["candidate_vs_C_probability_dataset_pp"].values()) >= -1.0,
        "S1+M_vs_S0+M_logits": deltas["candidate_vs_S0_logits_macro_pp"] > 0,
    }
    guards["all_passed"] = all(guards.values())
    return {"status": "passed", "case_count": 9, "protocol": probe.PROTOCOL,
            "fixed_candidate": "S1+M_logits",
            "trusted_inventory": {"path": str(FIXED_INVENTORY_PATH),
                                  "sha256": probe.FIXED_INVENTORY_SHA256},
            "weighting": "three-seed mean within dataset; equal mean across three datasets",
            "cases": [{"dataset_key": r["dataset_key"], "seed": r["seed"],
                       "branches": r["branches"], "combinations": r["combinations"]}
                      for r in sorted(reports, key=lambda r: (DATASETS.index(r["dataset_key"]), r["seed"]))],
            "dataset_means": dataset_means, "macro": macro,
            "complete_dataset_metrics": complete_dataset_metrics,
            "complete_macro_metrics": complete_macro_metrics,
            "deltas_pp": deltas, "guards": guards,
            "claims": {"split": "Base validation only", "novel_or_hm": False,
                       "significance_test": False, "automatic_training_action": False,
                       "duplicated_teacher_weighting": "T0/T1 retained in reports and excluded from guards"}}


def aggregate_files(paths: list[Path], output_dir: Path) -> dict[str, Any]:
    output = _fresh_output(output_dir)
    reports = [json.loads(Path(path).read_text()) for path in paths]
    result = aggregate_reports(reports)
    result["inputs"] = [{"path": str(Path(path).resolve()), "sha256": probe.sha256_file(Path(path))} for path in paths]
    output.mkdir(parents=True); _atomic_json(output / "aggregate.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="analyze one completed export on CPU")
    analyze.add_argument("--export", type=Path, required=True); analyze.add_argument("--output-dir", type=Path, required=True)
    aggregate = commands.add_parser("aggregate", help="aggregate exactly nine completed analyses")
    aggregate.add_argument("--analyses", type=Path, nargs=9, required=True)
    aggregate.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze_export(args.export, args.output_dir) if args.command == "analyze" else aggregate_files(args.analyses, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
