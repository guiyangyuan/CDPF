#!/usr/bin/env python3
"""Independent strict audit for the fixed DTD/Aircraft comparison."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev

DATASET_NAMES = {"dtd": "DescribableTextures", "fgvc_aircraft": "FGVCAircraft"}
VARIANTS = ("baseline", "hierarchical_adapter", "flat_dual_adapter")
BRANCHES = {"baseline": ("student",), "hierarchical_adapter": ("student", "teacher", "ensemble"),
            "flat_dual_adapter": ("student", "teacher", "ensemble")}
PARAMETERS = {"baseline": 2048, "hierarchical_adapter": 790018, "flat_dual_adapter": 790018}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module); return module


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_frozen(source_root, prepared):
    source_root = Path(source_root)
    for relative, expected in prepared["files_sha256"].items():
        path = source_root / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Prepared source dependency changed: {relative}")
    here = Path(__file__).resolve()
    operational = {here.name: digest(here), "cross_dataset_study.py": digest(here.with_name("cross_dataset_study.py"))}
    if operational != prepared["operational_scripts_sha256"]:
        raise ValueError("Prepared operational scripts changed before audit")


def comparable(config):
    value = copy.deepcopy(config)
    for field in ("OUTPUT_DIR", "MODEL_DIR", "RESUME", "EVAL_ONLY", "LOAD_EPOCH", "SEED"):
        value.pop(field, None)
    value.get("DATASET", {}).pop("SUBSAMPLE_CLASSES", None)
    value.get("TRAINER", {}).get("TEXTCONSISTENCY", {}).pop("INFERENCE_BRANCH", None)
    return value


def family_config(config):
    """Remove only preregistered structural/KD differences for cross-variant comparison."""
    value = comparable(config)
    value.get("TRAINER", {}).pop("NAME", None)
    tc = value.get("TRAINER", {}).get("TEXTCONSISTENCY", {})
    for field in ("AGGREGATION", "USE_FEATURE_ADAPTER", "ABLATION_GROUP", "KD_WEIGHT"):
        tc.pop(field, None)
    return value


def validate_metric_samples(metric, expected):
    if metric.get("split") != "test" or metric.get("test_samples_sha256") != expected["test_sha256"]:
        raise ValueError("Evaluation split/test hash differs from preparation")
    if (metric.get("train_samples_sha256") != expected["train_sha256"] or
            metric.get("val_samples_sha256") != expected["val_sha256"]):
        raise ValueError("Evaluation train/validation sample hash differs from preparation")


def checkpoint_audit(path, metric, manifest, variant, epoch):
    import torch
    if not path.is_file() or not path.with_name(path.name + ".sha256").is_file():
        raise FileNotFoundError(f"Checkpoint or SHA sidecar missing: {path}")
    actual = digest(path); sidecar = path.with_name(path.name + ".sha256").read_text().strip()
    if actual != sidecar or metric["checkpoints"] != [{"path": str(path.resolve()), "sha256": actual, "epoch": epoch}]:
        raise ValueError("Checkpoint hash/path/epoch evidence differs")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("epoch") != epoch or state.get("repro_signature") != manifest["model_signature"]:
        raise ValueError("Checkpoint epoch or model signature differs")
    provenance = state.get("run_provenance", {})
    for field in ("dataset", "seed", "shots", "code_sha256", "train_samples_sha256", "effective_losses"):
        if provenance.get(field) != manifest.get(field):
            raise ValueError(f"Checkpoint provenance differs: {field}")
    tensors = {k: v for k, v in state.get("state_dict", {}).items()
               if not k.endswith(("token_prefix", "token_suffix"))}
    if not tensors or any(not torch.is_tensor(v) or not torch.isfinite(v).all().item() for v in tensors.values()):
        raise ValueError("Checkpoint contains missing or non-finite parameters")
    if sum(v.numel() for v in tensors.values()) != PARAMETERS[variant]:
        raise ValueError("Checkpoint parameter count differs from fixed model")


def expected_loader(prepared, dataset, seed, split):
    rows = prepared["datasets"][dataset]["loader_evidence"]
    return next(row for row in rows if row["seed"] == seed)["splits"][split]


def parse_log(path):
    text = path.read_text(); totals = re.findall(r"\* total:\s*([0-9,]+)", text)
    correct = re.findall(r"\* correct:\s*([0-9,]+)", text)
    if not totals or not correct:
        raise ValueError(f"Evaluation log lacks correct/total: {path}")
    return int(correct[-1].replace(",", "")), int(totals[-1].replace(",", ""))


def audit_dataset(source_root, root, prepared, kind, dataset):
    source_root, root = Path(source_root), Path(root)
    summarize = load(source_root / "scripts/repro/summarize.py", f"cross_summary_{kind}_{dataset}")
    protocol_path = root / kind / dataset / "protocol.json"
    stage = next(s for s in prepared["protocol"]["stages"] if s["name"] == f"{kind}_{dataset}")
    scoped = {"dataset": DATASET_NAMES[dataset], "seeds": stage["seeds"], "epochs": stage["epochs"],
              "variants": list(VARIANTS), "shots": 16, "kd_weight": 1.0,
              "code_sha256": prepared["model_source_sha256"], "fixed_settings": prepared["protocol"]["fixed_settings"],
              "view_orders": prepared["protocol"]["view_orders"]}
    if not protocol_path.is_file() or json.loads(protocol_path.read_text()) != scoped:
        raise ValueError("Dataset protocol identity changed")
    preflight_path = protocol_path.parent / "data_preflight.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text()) != prepared["datasets"][dataset]:
        raise ValueError("Dataset preflight identity is missing or changed")
    summary_rows = summarize.collect(protocol_path.parent)
    if len(summary_rows) != 7:
        raise ValueError("Expected exactly seven variant/branch summary rows")
    configs = {}; detailed = []
    for variant in VARIANTS:
        for seed in stage["seeds"]:
            directory = protocol_path.parent / variant / f"seed{seed}"
            manifest = json.loads((directory / "run_manifest.json").read_text())
            if manifest.get("trainable_parameters") != PARAMETERS[variant]:
                raise ValueError("Manifest parameter count differs from fixed model")
            evidence = expected_loader(prepared, dataset, seed, "base")
            for field in ("train", "val"):
                if manifest.get(f"{field}_samples_sha256") != evidence[f"{field}_sha256"]:
                    raise ValueError(f"Prepared {field} sample hash differs")
            import yaml
            cfg = yaml.safe_load((directory / "effective_config.yaml").read_text())
            key = (dataset, seed); normalized = family_config(cfg)
            if key in configs and configs[key] != normalized:
                raise ValueError("Training configuration drift outside explicit model differences")
            configs[key] = normalized
            cp = directory / "textconsistency" / f"model.pth.tar-{stage['epochs']}"
            for branch in BRANCHES[variant]:
                scores = {}
                for split in ("base", "new"):
                    metric_path = directory / f"eval_{split}_{branch}/metrics.json"
                    metric = json.loads(metric_path.read_text())
                    expected = expected_loader(prepared, dataset, seed, split)
                    validate_metric_samples(metric, expected)
                    classes = expected["classes"]
                    order = metric.get("view_order_manifest")
                    if variant == "baseline":
                        if order is not None or metric.get("description_counts") != []:
                            raise ValueError("Baseline unexpectedly has description evidence")
                    else:
                        if not isinstance(order, dict) or [r.get("classname") for r in order.get("classes", [])] != classes:
                            raise ValueError("Description class order differs from loader")
                        counts = metric.get("description_counts")
                        if [r.get("count") for r in order["classes"]] != counts:
                            raise ValueError("Description counts differ from order manifest")
                    checkpoint_audit(cp, metric, manifest, variant, stage["epochs"])
                    log_path = root / "logs" / f"{kind}_{dataset}"
                    command_index = next(i for i, cmd in enumerate(stage["commands"])
                                         if cmd[cmd.index("--output-dir") + 1] == str(directory / f"eval_{split}_{branch}"))
                    correct, total = parse_log(log_path / f"{command_index:03d}.stdout.log")
                    accuracy = float(metric["metrics"]["accuracy"])
                    if total != expected["test_count"] or not math.isclose(accuracy, 100 * correct / total, abs_tol=5e-4):
                        raise ValueError("Evaluation count or accuracy differs from log/loader")
                    scores[split] = accuracy
                hm = 2 * scores["base"] * scores["new"] / sum(scores.values()) if sum(scores.values()) else 0.0
                detailed.append({"dataset": dataset, "variant": variant, "seed": seed, "branch": branch,
                                 "base": scores["base"], "novel": scores["new"], "hm": hm})
    for row in summary_rows:
        row["dataset"] = dataset
    return detailed, summary_rows


def final_analysis(rows, summaries):
    if len(rows) != 42 or len(summaries) != 14:
        raise ValueError("All 84 formal evaluations must exist before final analysis")
    primary = [r for r in rows if r["branch"] == ("student" if r["variant"] == "baseline" else "ensemble")]
    comparisons = []
    for dataset in DATASET_NAMES:
        indexed = {(r["variant"], r["seed"]): r for r in primary if r["dataset"] == dataset}
        for left, right, label in (("hierarchical_adapter", "flat_dual_adapter", "hierarchical_minus_flat"),
                                   ("hierarchical_adapter", "baseline", "method_minus_baseline")):
            values = []
            for seed in (1, 2, 3):
                values.append({"seed": seed, **{metric: indexed[left, seed][metric] - indexed[right, seed][metric]
                                                for metric in ("base", "novel", "hm")}})
            comparisons.append({"dataset": dataset, "comparison": label, "per_seed": values,
                                "mean": {m: mean(v[m] for v in values) for m in ("base", "novel", "hm")},
                                "sample_sd": {m: stdev(v[m] for v in values) for m in ("base", "novel", "hm")}})
    return {"passed": True, "seed_branch_rows": rows, "summary_rows": summaries, "comparisons": comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--study-root", required=True, type=Path); parser.add_argument("--preparation-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=[f"{k}_{d}" for k in ("smoke", "formal") for d in DATASET_NAMES])
    parser.add_argument("--final", action="store_true"); args = parser.parse_args(argv)
    if bool(args.stage) == bool(args.final): parser.error("choose exactly one of --stage or --final")
    prepared = json.loads((args.preparation_dir / "preparation.json").read_text())
    verify_frozen(args.source_root, prepared)
    if args.stage:
        kind, dataset = args.stage.split("_", 1); rows, summaries = audit_dataset(args.source_root, args.study_root, prepared, kind, dataset)
        output = args.study_root / "audit" / f"{args.stage}.json"; output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"passed": True, "rows": rows, "summaries": summaries}, indent=2)); return
    rows, summaries = [], []
    for dataset in DATASET_NAMES:
        r, s = audit_dataset(args.source_root, args.study_root, prepared, "formal", dataset); rows += r; summaries += s
    result = final_analysis(rows, summaries); (args.study_root / "analysis.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__": main()
