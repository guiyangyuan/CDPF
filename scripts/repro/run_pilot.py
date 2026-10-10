#!/usr/bin/env python3
"""Build reproducible commands for the CoOp student used by fixed fusion."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "ucf101": "UCF101",
    "dtd": "DescribableTextures",
    "fgvc_aircraft": "FGVCAircraft",
    "caltech101": "Caltech101",
    "oxford_pets": "OxfordPets",
    "food101": "Food101",
    "oxford_flowers": "OxfordFlowers",
    "eurosat": "EuroSAT",
    "stanford_cars": "StanfordCars",
    "sun397": "SUN397",
}


def build_commands(
    data_root,
    output_root,
    seeds,
    epochs,
    variants,
    device="cuda",
    python=sys.executable,
    dataset="ucf101",
    kd_weight=0.0,
):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if variants != ["baseline"]:
        raise ValueError("Only the current CoOp baseline is supported")
    if not math.isfinite(kd_weight) or kd_weight < 0:
        raise ValueError("KD weight must be a finite nonnegative number")
    commands = []
    for seed in seeds:
        train_dir = Path(output_root) / "baseline" / f"seed{seed}"
        common = [
            python,
            str(ROOT / "train.py"),
            "--root",
            str(Path(data_root).resolve()),
            "--seed",
            str(seed),
            "--trainer",
            "CoOp_TextConsistency_Ablation",
            "--dataset-config-file",
            str(ROOT / f"configs/datasets/{dataset}.yaml"),
            "--config-file",
            str(ROOT / "configs/repro/扩展验证.yaml"),
        ]
        options = [
            "DATASET.NUM_SHOTS",
            "16",
            "TRAINER.TEXTCONSISTENCY.DEVICE",
            device,
            "TRAINER.TEXTCONSISTENCY.AGGREGATION",
            "none",
            "TRAINER.TEXTCONSISTENCY.USE_FEATURE_ADAPTER",
            "False",
            "TRAINER.TEXTCONSISTENCY.VIEW_ORDER",
            "contiguous",
            "TRAINER.TEXTCONSISTENCY.ORDER_SEED",
            "0",
            "TRAINER.TEXTCONSISTENCY.ABLATION_GROUP",
            "baseline",
            "TRAINER.TEXTCONSISTENCY.INFERENCE_BRANCH",
            "student",
            "TRAINER.TEXTCONSISTENCY.KD_WEIGHT",
            "0.0",
            "OPTIM.MAX_EPOCH",
            str(epochs),
        ]
        commands.append(common + ["--output-dir", str(train_dir)] + options + ["DATASET.SUBSAMPLE_CLASSES", "base"])
        for split in ("base", "new"):
            commands.append(
                common
                + [
                    "--output-dir",
                    str(train_dir / f"eval_{split}_student"),
                    "--model-dir",
                    str(train_dir),
                    "--load-epoch",
                    str(epochs),
                    "--eval-only",
                ]
                + options
                + ["DATASET.SUBSAMPLE_CLASSES", split]
            )
    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dataset", choices=list(DATASETS), default="ucf101")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("Epochs must be positive and seeds must be distinct nonnegative integers")

    output = args.output_root.resolve()
    commands = build_commands(
        args.data_root,
        output,
        args.seeds,
        args.epochs,
        ["baseline"],
        args.device,
        dataset=args.dataset,
        kd_weight=0.0,
    )
    plan = {
        "purpose": "current_fixed_fusion_coop_baseline",
        "dataset": DATASETS[args.dataset],
        "dataset_key": args.dataset,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "variants": ["baseline"],
        "shots": 16,
        "fixed_settings": {"n_ctx": 4, "n_views": 20, "ensemble_alpha": 0.5},
        "checkpoint_rule": "specified final epoch; no test-based selection",
        "commands": commands,
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    if output.exists():
        raise FileExistsError(f"Choose a fresh output root: {output}")
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use a GPU server for the full run")
    from preflight import check
    from trainers.textconsistency_runtime import source_digest

    plan["data_preflight"] = check(args.data_root, args.dataset)
    plan["code_sha256"] = source_digest()
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(plan, indent=2))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "Dassl.pytorch"), env.get("PYTHONPATH", "")])
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    for index, command in enumerate(commands):
        print(f"Running {index + 1}/{len(commands)}: {command}", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
