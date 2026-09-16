#!/usr/bin/env python3
"""Evaluate a frozen CoOp checkpoint with description-mean logit fusion.

This is intentionally an evaluator, not a trainer or a hyperparameter search.
Every full run uses epoch 20 and alpha 0.5 and writes to a fresh directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import sys
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import torch
from torch.nn import functional as F


EPOCH = 20
ALPHA = 0.5
TRAINABLE_PARAMETERS = 2048
ALLOWED_SEEDS = (1, 2, 3)
DATASETS = {
    "ucf101": ("UCF101", "d74a9770d0d814c56269486e2492327881633d98314281d0f13a584ee279f17f"),
    "dtd": ("DescribableTextures", "9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba"),
    "fgvc_aircraft": ("FGVCAircraft", "9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba"),
    "caltech101": ("Caltech101", "9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba"),
    "oxford_pets": ("OxfordPets", "9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba"),
    "food101": ("Food101", "9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba"),
    "oxford_flowers": ("OxfordFlowers", "9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba"),
    "eurosat": ("EuroSAT", "9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba"),
}
EXPECTED_SIGNATURE = {
    "format": 2,
    "trainer": "CoOp_TextConsistency_Ablation",
    "n_ctx": 4,
    "aggregation": "none",
    "feature_adapter": False,
    "n_groups": None,
    "n_views": 20,
    "ablation_group": "baseline",
    "template_sha256": None,
    "view_order": "contiguous",
    "order_seed": 0,
    "order_version": 1,
}
EXPECTED_LOSSES = {"kd": 0.0, "adaptive_ce": 1.0, "diversity": 0.0, "contrastive": 0.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_source_digest(source_root: Path) -> str:
    """Reproduce trainers.textconsistency_runtime.source_digest without imports."""
    root = Path(source_root).resolve()
    files = [root / "train.py"]
    for folder in ("trainers", "datasets", "clip", "Dassl.pytorch/dassl"):
        files.extend(sorted((root / folder).rglob("*.py")))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Source tree is incomplete: {missing[0]}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def frozen_fusion_logits(
    student_logits: torch.Tensor,
    image_features: torch.Tensor,
    description_features: torch.Tensor,
    view_mask: torch.Tensor,
    logit_scale: torch.Tensor | float,
) -> dict[str, torch.Tensor]:
    """Return all six fixed logit/probability branches.

    Description vectors are normalized individually, averaged only over valid
    positions, and normalized again. Image vectors are normalized independently.
    """
    if student_logits.ndim != 2 or image_features.ndim != 2 or description_features.ndim != 3:
        raise ValueError("Expected student [B,C], image [B,D], descriptions [C,V,D]")
    if view_mask.ndim != 2 or tuple(view_mask.shape) != tuple(description_features.shape[:2]):
        raise ValueError("Description mask shape differs from description tensor")
    batch, classes = student_logits.shape
    if image_features.shape[0] != batch or description_features.shape[0] != classes:
        raise ValueError("Batch/class dimensions are inconsistent")
    if image_features.shape[1] != description_features.shape[2]:
        raise ValueError("Image and description feature dimensions differ")
    tensors = (student_logits, image_features, description_features)
    if any(not torch.is_floating_point(value) or not torch.isfinite(value).all().item() for value in tensors):
        raise ValueError("Fusion inputs must be finite floating-point tensors")
    mask = view_mask.to(device=description_features.device, dtype=torch.bool)
    if not mask.any(dim=1).all().item():
        raise ValueError("Each class requires at least one valid description")
    scale = torch.as_tensor(logit_scale, device=image_features.device, dtype=image_features.dtype)
    if scale.numel() != 1 or not torch.isfinite(scale).all().item() or scale.item() <= 0:
        raise ValueError("logit_scale must be one finite positive scalar")

    descriptions = F.normalize(description_features.float(), dim=-1)
    weights = mask.to(descriptions.dtype) / mask.sum(dim=1, keepdim=True)
    class_features = F.normalize(torch.einsum("cv,cvd->cd", weights, descriptions), dim=-1)
    images = F.normalize(image_features.float(), dim=-1)
    description_logits = scale.float() * images @ class_features.t()
    description_logits = description_logits.to(student_logits.dtype)
    simple_fusion = (student_logits + description_logits) * ALPHA

    instance_scores = scale.float() * torch.einsum("bd,cvd->bcv", images, descriptions)
    expanded_mask = mask.unsqueeze(0)
    attention = torch.softmax(
        instance_scores.masked_fill(~expanded_mask, torch.finfo(instance_scores.dtype).min), dim=-1)
    attention = attention * expanded_mask.to(attention.dtype)
    attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    instance_features = F.normalize(torch.einsum("bcv,cvd->bcd", attention, descriptions), dim=-1)
    instance_logits = (scale.float() * torch.einsum("bd,bcd->bc", images, instance_features)).to(student_logits.dtype)

    student_probability = torch.softmax(student_logits, dim=-1)
    mean_probability = torch.softmax(description_logits, dim=-1)
    instance_probability = torch.softmax(instance_logits, dim=-1)
    mean_probability_fusion = ALPHA * student_probability + ALPHA * mean_probability
    adk_probability_fusion = (0.5 * student_probability +
                              0.25 * mean_probability + 0.25 * instance_probability)
    outputs = (description_logits, simple_fusion, instance_logits,
               mean_probability_fusion, adk_probability_fusion)
    if any(not torch.isfinite(value).all().item() for value in outputs):
        raise ValueError("Fusion produced non-finite logits")
    return {
        "student": student_logits,
        "description_mean": description_logits,
        "simple_fusion": simple_fusion,
        "description_instance": instance_logits,
        "mean_probability_fusion": mean_probability_fusion,
        "adk_probability_fusion": adk_probability_fusion,
        "description_mean_features": class_features,
        "description_instance_features": instance_features,
        "description_attention": attention,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required configuration is missing: {path}")
    import yaml
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _parse_log(path: Path) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Historical evaluation log is missing: {path}")
    text = path.read_text(errors="replace")
    correct = re.findall(r"\* correct:\s*([0-9,]+)", text)
    totals = re.findall(r"\* total:\s*([0-9,]+)", text)
    if not correct or not totals:
        raise ValueError(f"Historical log lacks final correct/total: {path}")
    return int(correct[-1].replace(",", "")), int(totals[-1].replace(",", ""))


def _same_value(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise ValueError(message)


def _validate_seed(seed: int) -> None:
    if seed not in ALLOWED_SEEDS:
        raise ValueError("Seed must be one of 1, 2, or 3")


def _expected_signature(dataset: str) -> dict[str, Any]:
    if dataset == "ucf101":
        return {field: value for field, value in EXPECTED_SIGNATURE.items()
                if field not in ("view_order", "order_seed", "order_version")} | {"format": 1}
    return dict(EXPECTED_SIGNATURE)


def _validate_provenance(payload: Mapping[str, Any], dataset: str, seed: int, split: str) -> None:
    dataset_name = DATASETS[dataset][0]
    expected = {
        "dataset": dataset_name,
        "seed": seed,
        "shots": 16,
        "subsample": split,
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "effective_losses": EXPECTED_LOSSES,
    }
    for field, value in expected.items():
        _same_value(payload.get(field), value, f"Baseline provenance mismatch: {field}")
    signature = payload.get("model_signature")
    if not isinstance(signature, dict):
        raise ValueError("Baseline model signature is missing")
    expected_signature = _expected_signature(dataset)
    if set(signature) != set(expected_signature) | {"clip_sha256"}:
        raise ValueError("Baseline model signature has unexpected or missing fields")
    for field, value in expected_signature.items():
        _same_value(signature.get(field), value, f"Baseline is not the fixed epoch-20 student: {field}")
    if not isinstance(signature.get("clip_sha256"), str) or len(signature["clip_sha256"]) != 64:
        raise ValueError("Baseline CLIP SHA-256 is missing")
    for field in ("code_sha256", "train_samples_sha256", "val_samples_sha256", "test_samples_sha256"):
        if not isinstance(payload.get(field), str) or len(payload[field]) != 64:
            raise ValueError(f"Baseline provenance lacks a valid {field}")


def _normalized_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Ignore only paths that are explicitly relocatable for copied evidence."""
    value = json.loads(json.dumps(config))
    value["OUTPUT_DIR"] = "<relocated-output>"
    value.setdefault("DATASET", {})["ROOT"] = "<relocated-data>"
    return value


def _validate_source_config(actual: Mapping[str, Any], expected: Mapping[str, Any],
                            dataset: str, label: str) -> dict[str, Any] | None:
    actual_value = _normalized_config(actual)
    expected_value = _normalized_config(expected)
    if actual_value == expected_value:
        return None
    actual_tc = actual_value.get("TRAINER", {}).get("TEXTCONSISTENCY", {})
    expected_tc = expected_value.get("TRAINER", {}).get("TEXTCONSISTENCY", {})
    if dataset == "ucf101" and actual_tc.get("KD_WEIGHT") == 1.0 and expected_tc.get("KD_WEIGHT") == 0.0:
        adjusted = json.loads(json.dumps(actual_value))
        adjusted["TRAINER"]["TEXTCONSISTENCY"]["KD_WEIGHT"] = 0.0
        if adjusted == expected_value:
            return {
                "scope": "ucf101_legacy_baseline_only",
                "config": label,
                "historical_declared_kd_weight": 1.0,
                "source_derived_kd_weight": 0.0,
                "effective_kd_weight": 0.0,
                "reason": "baseline ablation disables KD; manifest/checkpoint effective loss remains zero",
            }
    raise ValueError(f"{label.capitalize()} config differs from source-derived baseline config")


def validate_baseline_metadata(
    baseline_dir: Path,
    dataset: str,
    seed: int,
    split: str,
    expected_source_sha256: str,
    *,
    expected_train_config: Mapping[str, Any] | None = None,
    expected_eval_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate hashes and JSON/YAML/log evidence without loading a model."""
    if dataset not in DATASETS or split not in ("base", "new"):
        raise ValueError("Unknown dataset or split")
    _validate_seed(seed)
    dataset_name, pinned_source = DATASETS[dataset]
    if expected_source_sha256 != pinned_source:
        raise ValueError(f"{dataset} requires frozen source digest {pinned_source}")
    baseline = Path(baseline_dir).resolve()
    train_manifest = _read_json(baseline / "run_manifest.json")
    eval_dir = baseline / f"eval_{split}_student"
    eval_manifest = _read_json(eval_dir / "run_manifest.json")
    metric = _read_json(eval_dir / "metrics.json")

    _validate_provenance(train_manifest, dataset, seed, "base")
    _validate_provenance(eval_manifest, dataset, seed, split)
    _validate_provenance(metric, dataset, seed, split)
    for field in ("model_signature", "code_sha256", "train_samples_sha256", "val_samples_sha256",
                  "test_samples_sha256", "effective_losses", "trainable_parameters"):
        if metric.get(field) != eval_manifest.get(field):
            raise ValueError(f"Evaluation manifest/metric mismatch: {field}")
    if split == "base" and train_manifest["train_samples_sha256"] != metric["train_samples_sha256"]:
        raise ValueError("Baseline training sample hash differs from evaluation evidence")
    if train_manifest["code_sha256"] != expected_source_sha256 or metric["code_sha256"] != expected_source_sha256:
        raise ValueError("Baseline was not trained by the requested frozen source")
    if metric.get("state") != "loaded_checkpoint" or metric.get("branch") != "student" or metric.get("split") != "test":
        raise ValueError("Historical baseline evaluation is not loaded-checkpoint student test inference")
    if metric.get("ensemble_alpha") != ALPHA:
        raise ValueError("Historical fixed alpha differs")

    checkpoint = baseline / "textconsistency" / f"model.pth.tar-{EPOCH}"
    sidecar = checkpoint.with_name(checkpoint.name + ".sha256")
    if not checkpoint.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Checkpoint or sidecar is missing: {checkpoint}")
    actual_checkpoint_sha = sha256_file(checkpoint)
    sidecar_sha = sidecar.read_text().strip()
    if actual_checkpoint_sha != sidecar_sha:
        raise ValueError("Checkpoint SHA-256 sidecar does not match checkpoint bytes")
    records = metric.get("checkpoints")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError("Historical evaluation must identify exactly one checkpoint")
    record = records[0]
    if record.get("epoch") != EPOCH or record.get("sha256") != actual_checkpoint_sha:
        raise ValueError("Relocated checkpoint differs from historical epoch/SHA evidence")

    correct, total = _parse_log(eval_dir / "log.txt")
    accuracy = metric.get("metrics", {}).get("accuracy")
    if not isinstance(accuracy, (float, int)) or not math.isfinite(float(accuracy)):
        raise ValueError("Historical student accuracy is missing or non-finite")
    if total <= 0 or not math.isclose(float(accuracy), 100.0 * correct / total, abs_tol=5e-4):
        raise ValueError("Historical student metric differs from log correct/total")

    train_config = _read_yaml(baseline / "effective_config.yaml")
    eval_config = _read_yaml(eval_dir / "effective_config.yaml")
    compatibility = []
    if expected_train_config is not None:
        fact = _validate_source_config(train_config, expected_train_config, dataset, "training")
        if fact is not None:
            compatibility.append(fact)
    if expected_eval_config is not None:
        fact = _validate_source_config(eval_config, expected_eval_config, dataset, "evaluation")
        if fact is not None:
            compatibility.append(fact)

    return {
        "baseline_dir": str(baseline),
        "eval_dir": str(eval_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": actual_checkpoint_sha,
        "metric": metric,
        "train_manifest": train_manifest,
        "eval_manifest": eval_manifest,
        "train_config": train_config,
        "eval_config": eval_config,
        "config_compatibility": compatibility,
        "historical_student": {"correct": correct, "total": total, "accuracy": float(accuracy)},
    }


def _module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load source module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_modules(source_root: Path):
    root = Path(source_root).resolve()
    sys.path[:0] = [str(root), str(root / "Dassl.pytorch")]
    runner = _module_from_path("_frozen_fusion_source_run_pilot", root / "scripts/repro/run_pilot.py")
    training = _module_from_path("_frozen_fusion_source_train", root / "train.py")
    return runner, training


def _namespace_from_command(command: list[str], output_dir: Path) -> SimpleNamespace:
    def value(flag: str, default: Any = "") -> Any:
        return command[command.index(flag) + 1] if flag in command else default
    if "--eval-only" in command:
        opts = command[command.index("--eval-only") + 1:]
    else:
        opts = command[command.index("--output-dir") + 2:]
    return SimpleNamespace(
        root=value("--root"), output_dir=str(output_dir), resume="", seed=int(value("--seed", -1)),
        source_domains=None, target_domains=None, transforms=None, trainer=value("--trainer"),
        backbone="", head="", config_file=value("--config-file"),
        dataset_config_file=value("--dataset-config-file"), eval_only="--eval-only" in command,
        model_dir=value("--model-dir"), load_epoch=int(value("--load-epoch", EPOCH)), no_train=False,
        opts=opts,
    )


def build_source_configs(source_root: Path, data_root: Path, output_dir: Path,
                         dataset: str, seed: int, split: str):
    """Build the exact baseline train/eval cfg through the frozen source APIs."""
    runner, training = _source_modules(source_root)
    output_root = Path(output_dir).resolve().parent / "_source_derived"
    commands = runner.build_commands(
        data_root, output_root, [seed], EPOCH, ["baseline"], device="cuda",
        python=sys.executable, dataset=dataset, kd_weight=1.0,
    )
    train_command = next(command for command in commands if "--eval-only" not in command)
    suffix = f"eval_{split}_student"
    eval_command = next(command for command in commands
                        if "--eval-only" in command and Path(command[command.index("--output-dir") + 1]).name == suffix)
    train_cfg = training.setup_cfg(_namespace_from_command(train_command, output_root / "baseline" / f"seed{seed}"))
    eval_cfg = training.setup_cfg(_namespace_from_command(eval_command, output_dir))
    import yaml
    return train_cfg, eval_cfg, yaml.safe_load(train_cfg.dump()), yaml.safe_load(eval_cfg.dump()), training


def _ensure_disjoint_fresh_output(output_dir: Path, forbidden: Iterable[Path]) -> Path:
    output = Path(output_dir).resolve()
    for raw in forbidden:
        parent = Path(raw).resolve()
        if output == parent or parent in output.parents:
            raise ValueError(f"Output directory is inside immutable input: {parent}")
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    return output


def _sample_inventory(items: Iterable[Any], data_root: Path) -> str:
    root = Path(data_root).resolve()
    records = []
    for item in items or []:
        path = Path(item.impath).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Dataset image is outside data root: {path}") from exc
        stat = path.stat()
        records.append((str(relative), int(item.label), item.classname, stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(records, ensure_ascii=False).encode()).hexdigest()


def description_order_evidence(encoder: Any, classnames: list[str]) -> dict[str, Any]:
    """Return source evidence, filling the old UCF contiguous-only interface."""
    existing = getattr(encoder, "view_order_manifest", None)
    if existing is not None:
        return existing
    counts = list(encoder.description_counts)
    if len(counts) != len(classnames) or any(not isinstance(count, int) or count < 1 for count in counts):
        raise ValueError("Description counts do not match the class order")
    payload = {
        "version": 1,
        "policy": "contiguous",
        "seed": 0,
        "n_groups": 4,
        "classes": [{"classname": name, "count": count, "order": list(range(count))}
                    for name, count in zip(classnames, counts)],
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return payload


def _validate_checkpoint_tensors(checkpoint_path: Path, manifest: Mapping[str, Any]) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != EPOCH or checkpoint.get("repro_signature") != manifest["model_signature"]:
        raise ValueError("Checkpoint epoch/signature differs from baseline manifest")
    provenance = checkpoint.get("run_provenance", {})
    for field in ("dataset", "seed", "shots", "code_sha256", "train_samples_sha256",
                  "effective_losses", "trainable_parameters", "model_signature"):
        if provenance.get(field) != manifest.get(field):
            raise ValueError(f"Checkpoint training provenance mismatch: {field}")
    tensors = {name: value for name, value in checkpoint.get("state_dict", {}).items()
               if not name.endswith(("token_prefix", "token_suffix"))}
    if not tensors or any(not torch.is_tensor(value) or not torch.isfinite(value).all().item()
                          for value in tensors.values()):
        raise ValueError("Checkpoint contains missing or non-finite tensors")
    if sum(value.numel() for value in tensors.values()) != TRAINABLE_PARAMETERS:
        raise ValueError("Checkpoint is not the 2048-parameter CoOp baseline")


def _branch_stats(predictions: list[int], labels: list[int]) -> dict[str, Any]:
    total = len(labels)
    if len(predictions) != total or total == 0:
        raise ValueError("Prediction count differs from nonempty loader count")
    correct = sum(prediction == label for prediction, label in zip(predictions, labels))
    return {"correct": correct, "total": total, "accuracy": 100.0 * correct / total}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    _validate_seed(args.seed)
    source_root = args.source_root.resolve()
    baseline_dir = args.baseline_dir.resolve()
    data_root = args.data_root.resolve()
    output_dir = _ensure_disjoint_fresh_output(args.output_dir, (source_root, baseline_dir, data_root))
    actual_source_sha = model_source_digest(source_root)
    if actual_source_sha != args.expected_source_sha256:
        raise ValueError("Frozen source tree SHA-256 differs from --expected-source-sha256")

    train_cfg, cfg, expected_train, expected_eval, training = build_source_configs(
        source_root, data_root, output_dir, args.dataset, args.seed, args.split)
    metadata = validate_baseline_metadata(
        baseline_dir, args.dataset, args.seed, args.split, args.expected_source_sha256,
        expected_train_config=expected_train, expected_eval_config=expected_eval,
    )
    if args.validate_only:
        return {
            "status": "validated_metadata",
            "dataset": args.dataset,
            "seed": args.seed,
            "subsample": args.split,
            "source_sha256": actual_source_sha,
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "historical_student": metadata["historical_student"],
        }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen fusion evaluation; no output was written")

    output_dir.mkdir(parents=True)
    try:
        from dassl.utils import set_random_seed
        from trainers.textconsistency_common import CachedMultiViewTextEncoder, load_clip_checkpoint
        from trainers.textconsistency_runtime import sample_digest

        set_random_seed(args.seed)
        trainer = training.build_trainer(cfg)
        current = trainer.run_provenance
        metric = metadata["metric"]
        for field in ("dataset", "seed", "shots", "code_sha256", "train_samples_sha256",
                      "val_samples_sha256", "test_samples_sha256", "trainable_parameters", "model_signature"):
            if current.get(field) != metric.get(field):
                raise ValueError(f"Current loader/model provenance differs from historical baseline: {field}")
        trainer.load_model(baseline_dir, epoch=EPOCH)
        loaded = getattr(trainer, "loaded_checkpoints", [])
        if len(loaded) != 1 or loaded[0]["sha256"] != metadata["checkpoint_sha256"] or loaded[0]["epoch"] != EPOCH:
            raise ValueError("Original trainer loader did not reload the fixed checkpoint")
        _validate_checkpoint_tensors(Path(metadata["checkpoint"]), metadata["train_manifest"])

        clip_model = load_clip_checkpoint(cfg)
        encoder = CachedMultiViewTextEncoder(cfg, trainer.dm.dataset.classnames, clip_model)
        if sha256_file(source_root / "clip/ViT-B-16.pt") != trainer.repro_signature["clip_sha256"]:
            raise ValueError("CLIP file SHA differs from the baseline signature")
        descriptions = encoder.get_view_features().to(trainer.device)
        mask = encoder.view_mask.to(trainer.device)
        shared_logit_scale = float(trainer.model.logit_scale.exp().detach().cpu().item())
        del clip_model

        classnames = list(trainer.dm.dataset.classnames)
        order_evidence = description_order_evidence(encoder, classnames)
        samples = trainer.dm.dataset.test
        before_inventory = _sample_inventory(samples, data_root)
        source_before = model_source_digest(source_root)
        input_hashes_before = {
            "checkpoint": sha256_file(Path(metadata["checkpoint"])),
            "sidecar": sha256_file(Path(metadata["checkpoint"] + ".sha256")),
            "historical_metrics": sha256_file(Path(metadata["eval_dir"]) / "metrics.json"),
            "historical_log": sha256_file(Path(metadata["eval_dir"]) / "log.txt"),
        }

        predictions = {name: [] for name in (
            "student", "description_mean", "simple_fusion", "description_instance",
            "mean_probability_fusion", "adk_probability_fusion",
        )}
        labels: list[int] = []
        rows: list[dict[str, Any]] = []
        trainer.set_model_mode("eval")
        with torch.no_grad():
            for batch in trainer.test_loader:
                image, label = trainer.parse_batch_test(batch)
                student, _, image_features, _ = trainer.model(image, return_attn=False)
                logits = frozen_fusion_logits(
                    student, image_features, descriptions, mask, trainer.model.logit_scale.exp())
                batch_predictions = {name: logits[name].argmax(dim=1).cpu().tolist() for name in predictions}
                batch_labels = label.cpu().tolist()
                paths = list(batch["impath"])
                if len(paths) != len(batch_labels):
                    raise ValueError("Loader path count differs from labels")
                for index, (path_text, target) in enumerate(zip(paths, batch_labels)):
                    path = Path(path_text).resolve()
                    try:
                        relative = path.relative_to(data_root)
                    except ValueError as exc:
                        raise ValueError(f"Prediction image lies outside data root: {path}") from exc
                    branch_values = {name: batch_predictions[name][index] for name in predictions}
                    rows.append({
                        "relative_path": str(relative),
                        "label": target,
                        "label_classname": classnames[target],
                        "predictions": {name: {"label": value, "classname": classnames[value]}
                                        for name, value in branch_values.items()},
                    })
                labels.extend(batch_labels)
                for name in predictions:
                    predictions[name].extend(batch_predictions[name])

        stats = {name: _branch_stats(values, labels) for name, values in predictions.items()}
        historical = metadata["historical_student"]
        if (stats["student"]["correct"] != historical["correct"] or
                stats["student"]["total"] != historical["total"] or
                not math.isclose(stats["student"]["accuracy"], historical["accuracy"], abs_tol=5e-4)):
            raise ValueError("Reloaded student result differs from historical metrics/log")
        if len(rows) != len(samples) or any(stats[name]["total"] != len(rows) for name in stats):
            raise ValueError("Prediction JSONL count differs from loader/metrics")

        predictions_path = output_dir / "predictions.jsonl"
        predictions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        predictions_sha = sha256_file(predictions_path)
        after_inventory = _sample_inventory(samples, data_root)
        source_after = model_source_digest(source_root)
        input_hashes_after = {
            "checkpoint": sha256_file(Path(metadata["checkpoint"])),
            "sidecar": sha256_file(Path(metadata["checkpoint"] + ".sha256")),
            "historical_metrics": sha256_file(Path(metadata["eval_dir"]) / "metrics.json"),
            "historical_log": sha256_file(Path(metadata["eval_dir"]) / "log.txt"),
        }
        if source_after != source_before or after_inventory != before_inventory or input_hashes_after != input_hashes_before:
            raise ValueError("Frozen source/data/baseline inputs changed during evaluation")

        import yaml
        effective_cfg = yaml.safe_load(cfg.dump())
        result = {
            "status": "passed",
            "dataset": DATASETS[args.dataset][0],
            "dataset_key": args.dataset,
            "seed": args.seed,
            "subsample": args.split,
            "split": "test",
            "fixed_protocol": {
                "epoch": EPOCH,
                "alpha": ALPHA,
                "formula": "simple_fusion=(student_logits+description_mean_logits)/2",
                "description_mean": "normalize(each valid description), mean(valid), normalize(mean)",
                "logits": "exp(CLIP.logit_scale) * normalized_image @ normalized_text.T",
                "shared_logit_scale": shared_logit_scale,
                "attention_tau": 1.0 / shared_logit_scale,
                "selection": "all branches reported; no search or training",
            },
            "method_identity": {
                "adk_probability_fusion": "controlled CoOp adaptation of the public ADK formulas; not official code or published-score replication",
                "formula_source": "https://arxiv.org/html/2512.17313v1#S3.SS4",
                "frozen_components": ["CLIP", "description text features"],
                "trainable_components": [],
            },
            "additional_formulas": {
                "description_attention": "masked_softmax(exp(CLIP.logit_scale)*cos(image, description), dim=descriptions)",
                "description_instance": "normalize(sum(description_attention*normalized_description))",
                "mean_probability_fusion": "0.5*softmax(student_logits)+0.5*softmax(description_mean_logits)",
                "adk_probability_fusion": "0.5*softmax(student_logits)+0.25*softmax(description_mean_logits)+0.25*softmax(description_instance_logits)",
                "description_loss_context_gradient": 0.0,
            },
            "metrics": stats,
            "historical_student": historical,
            "checkpoint": {"path": metadata["checkpoint"], "sha256": metadata["checkpoint_sha256"],
                           "sidecar": metadata["checkpoint"] + ".sha256", "epoch": EPOCH},
            "source": {"root": str(source_root), "model_sha256": actual_source_sha,
                       "expected_model_sha256": args.expected_source_sha256,
                       "evaluator_sha256": sha256_file(Path(__file__).resolve())},
            "assets": {"clip_sha256": trainer.repro_signature["clip_sha256"],
                       "template_sha256": encoder.template_sha256},
            "effective_config": effective_cfg,
            "baseline_declared_config": {
                "training_kd_weight": metadata["train_config"]["TRAINER"]["TEXTCONSISTENCY"]["KD_WEIGHT"],
                "evaluation_kd_weight": metadata["eval_config"]["TRAINER"]["TEXTCONSISTENCY"]["KD_WEIGHT"],
                "source_derived_current_kd_weight": effective_cfg["TRAINER"]["TEXTCONSISTENCY"]["KD_WEIGHT"],
                "effective_kd_weight": current["effective_losses"]["kd"],
                "compatibility": metadata["config_compatibility"],
            },
            "description": {"counts": encoder.description_counts,
                            "view_order_manifest": order_evidence},
            "class_order": classnames,
            "samples": {"train_sha256": sample_digest(trainer.dm.dataset.train_x, data_root),
                        "val_sha256": sample_digest(trainer.dm.dataset.val, data_root),
                        "test_sha256": sample_digest(samples, data_root),
                        "test_inventory_sha256": before_inventory},
            "predictions": {"path": str(predictions_path), "sha256": predictions_sha, "count": len(rows)},
            "input_hashes_before": input_hashes_before,
            "input_hashes_after": input_hashes_after,
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                            "device": "cuda"},
        }
        _atomic_json(output_dir / "metrics.json", result)
        return result
    except Exception as exc:
        _atomic_json(output_dir / "metrics.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--seed", type=int, choices=ALLOWED_SEEDS, required=True)
    parser.add_argument("--split", choices=("base", "new"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate source/baseline metadata and hashes without loading CLIP or writing output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
