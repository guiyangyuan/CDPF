#!/usr/bin/env python3
"""Validate and recompute the published CDPF results; Python standard library only.

Run from any directory: python3 tools/summarize_results.py [results.json]
All accuracies and gains use percentage points. The JSON key ``new`` means Novel.
"""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys


DATASETS = (
    "caltech101", "dtd", "eurosat", "fgvc_aircraft", "food101", "imagenet",
    "oxford_flowers", "oxford_pets", "stanford_cars", "sun397", "ucf101",
)
SEEDS = ("1", "2", "3")
CDPF = "mean_probability_fusion"
MATCHED = "matched_template_probability_fusion"
DEFAULT_RESULTS = Path(__file__).resolve().parents[1] / "results/cdpf_results.json"


def require_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} must contain exactly {', '.join(sorted(keys))}")


def require_close(stored, computed, label):
    if (isinstance(stored, bool) or not isinstance(stored, (int, float))
            or not math.isfinite(stored)
            or not math.isclose(stored, computed, rel_tol=0.0, abs_tol=1e-9)):
        raise ValueError(f"{label} does not match the recomputed value {computed}")


def harmonic_mean(base, novel):
    return 2.0 * base * novel / (base + novel) if base + novel else 0.0


def validate_metrics(metrics, label):
    require_keys(metrics, ("base", "new", "hm"), label)
    for name, value in metrics.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0.0 <= value <= 100.0):
            raise ValueError(f"{label}.{name} must be a finite percentage in [0, 100]")
    require_close(metrics["hm"], harmonic_mean(metrics["base"], metrics["new"]), f"{label}.hm")


def summarize(data):
    """Recompute all branch macros and the 33 paired CDPF/control HM directions.

    Per-dataset paper HM is H(mean Base across seeds, mean Novel across seeds).
    Paper macro HM is the unweighted mean of those 11 per-dataset HMs.
    The separate seed statistic averages H(Base, Novel) over the 33 seed pairs.
    """
    if data["status"] != "complete" or data["main_hm_definition"] != "H(mean_seed_base, mean_seed_new)":
        raise ValueError("Only complete results using the paper HM definition are supported")
    protocol = data["protocol"]
    if protocol["seeds"] != [1, 2, 3] or protocol["dataset_count"] != 11:
        raise ValueError("The published protocol requires 11 datasets and seeds 1, 2, 3")
    summary = data["summary"]
    require_keys(summary["per_seed"], DATASETS, "summary.per_seed datasets")
    require_keys(summary["per_dataset"], DATASETS, "summary.per_dataset datasets")
    if (sorted(summary["complete_datasets"]) != list(DATASETS)
            or summary["incomplete_datasets"] or summary["macro_dataset_count"] != 11):
        raise ValueError("The dataset completeness declaration is inconsistent")

    branches = set(summary["macro_hm"])
    if not {"student", CDPF, MATCHED} <= branches:
        raise ValueError("Required student, CDPF, and matched-template branches are missing")
    require_keys(summary["macro_dataset_counts"], branches, "summary.macro_dataset_counts")
    per_dataset = {}
    for dataset in DATASETS:
        seed_rows = summary["per_seed"][dataset]
        require_keys(seed_rows, SEEDS, f"{dataset} seeds")
        require_keys(summary["per_dataset"][dataset], branches, f"{dataset} stored branches")
        for seed in SEEDS:
            require_keys(seed_rows[seed], branches, f"{dataset}/seed{seed} branches")
            for branch in branches:
                validate_metrics(seed_rows[seed][branch], f"{dataset}/seed{seed}/{branch}")
        per_dataset[dataset] = {}
        for branch in sorted(branches):
            base = statistics.fmean(seed_rows[seed][branch]["base"] for seed in SEEDS)
            novel = statistics.fmean(seed_rows[seed][branch]["new"] for seed in SEEDS)
            metrics = {"base": base, "new": novel, "hm": harmonic_mean(base, novel)}
            stored = summary["per_dataset"][dataset][branch]
            validate_metrics(stored, f"{dataset}/{branch}")
            for metric, computed in metrics.items():
                require_close(stored[metric], computed, f"{dataset}/{branch}/{metric}")
            per_dataset[dataset][branch] = metrics

    macro = {}
    for branch in sorted(branches):
        if summary["macro_dataset_counts"][branch] != 11:
            raise ValueError(f"{branch} must include all 11 datasets")
        macro[branch] = {
            metric: statistics.fmean(per_dataset[dataset][branch][metric] for dataset in DATASETS)
            for metric in ("base", "new", "hm")
        }
        require_close(summary["macro_hm"][branch], macro[branch]["hm"], f"macro_hm/{branch}")
        macro[branch]["mean_seed_hm"] = statistics.fmean(
            summary["per_seed"][dataset][seed][branch]["hm"]
            for dataset in DATASETS for seed in SEEDS
        )

    gains = {
        dataset: {
            seed: summary["per_seed"][dataset][seed][CDPF]["hm"]
            - summary["per_seed"][dataset][seed][MATCHED]["hm"]
            for seed in SEEDS
        }
        for dataset in DATASETS
    }
    paired = [value for row in gains.values() for value in row.values()]
    directions = {
        "positive": sum(value > 1e-9 for value in paired),
        "negative": sum(value < -1e-9 for value in paired),
        "zero": sum(abs(value) <= 1e-9 for value in paired),
    }
    return {
        "status": "validated",
        "units": "percentage_points",
        "dataset_count": len(DATASETS),
        "seeds": [1, 2, 3],
        "definitions": {
            "base_and_new": "Equal-weight dataset means of the three-seed mean accuracies; new means Novel",
            "hm": "Equal-weight mean of 11 H(mean_seed_base, mean_seed_new); paper main-table HM",
            "mean_seed_hm": "Mean of 33 individual seed HMs; a separate stability statistic",
        },
        "macro": macro,
        "cdpf_vs_matched_template": {
            "cdpf_branch": CDPF,
            "control_branch": MATCHED,
            "pair_count": len(paired),
            "directions": directions,
            "macro_hm_gain": macro[CDPF]["hm"] - macro[MATCHED]["hm"],
            "mean_paired_seed_hm_gain": statistics.fmean(paired),
            "per_seed_hm_gains": gains,
            "interpretation": "Seeds are nested within datasets; 33 pairs are not 33 independent datasets",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=DEFAULT_RESULTS)
    args = parser.parse_args()
    try:
        data = json.loads(args.results.read_text(encoding="utf-8"))
        report = summarize(data)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Invalid results: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
