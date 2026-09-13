#!/usr/bin/env python3
"""Authenticate, prepare, and export the bounded H1 seed-1 validation cases."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F

from scripts.repro import validation_analysis as historical_analysis
from scripts.repro import validation_probe as probe


ROOT = Path(__file__).resolve().parents[2]
FIXED_REFERENCE_PATH = ROOT / "reports/preparation/h1-cpu-20260913/reference-snapshot.json"
FIXED_REFERENCE_SHA256 = "dd69f959be308f88c265a29679f0441cf8008a1250ae8ec8ba17785ee4b72c9e"
FIXED_INVENTORY_PATH = ROOT / "reports/diagnostics/error-analysis-20260911/evidence/proposed-probe-inputs.json"
FIXED_INVENTORY_SHA256 = probe.FIXED_INVENTORY_SHA256
DATASETS = tuple(probe.DATASETS)
CURRENT_SOURCE_SHA256 = "387ea6e46aa0fa6cba5109e64baab94a548b35c200a1945267d392bc35a36677"
HISTORICAL_CONFIG_SHA256 = {
    "ucf101": "e15b6f46099d52cd5a24c17abea447707eb96fc9085316f297d47fbee3e0019f",
    "dtd": "f1cf575cdc1055ba83d96a507b9cc5969c2e6e997143df9f9a38f2431e17c154",
    "fgvc_aircraft": "5fb6f16c2972cd242061c63c1c410477403c35d195f17ceabb134e520acf852c",
}
BRANCHES = ("S", "T", "M", "ensemble")
MODES = ("full_kl", "weighted_nckd")
PROTOCOL = {
    "version": 1,
    "seed": 1,
    "shots": 16,
    "epochs": 20,
    "split": "val",
    "subsample_classes": "base",
    "backbone": "ViT-B/16",
    "precision": "fp32",
    "temperature": 2.0,
    "kd_weight": 1.0,
    "teacher_ce_weight": 1.0,
    "ensemble_alpha": 0.5,
    "test_no_test": True,
    "final_model": "last_step",
    "export_sampler": "SequentialSampler",
    "export_workers": 0,
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _fresh_output(path: Path, protected: tuple[Path, ...] = ()) -> Path:
    return probe._fresh_output(Path(path), protected)


def _export_protected_paths(
        prepared_path: Path, model_dir: Path, prepared: Mapping[str, Any],
) -> tuple[Path, ...]:
    paths = [Path(prepared_path).resolve(), Path(prepared_path).resolve().parent,
             Path(model_dir).resolve(), FIXED_REFERENCE_PATH.resolve(), FIXED_INVENTORY_PATH.resolve()]

    def add(value: Any) -> None:
        if isinstance(value, (str, Path)) and str(value):
            paths.append(Path(value).resolve())

    for key in ("source_root", "data_root", "future_config_path"):
        add(prepared.get(key))
    for section, keys in (("inventory", ("path",)), ("reference", ("path",)),
                          ("historical", ("export", "npz")),
                          ("historical_config", ("path",)),
                          ("assets", ("clip", "template")),
                          ("fewshot_cache", ("path",))):
        record = prepared.get(section, {})
        if isinstance(record, Mapping):
            for key in keys:
                add(record.get(key))
    metadata = prepared.get("source_metadata", [])
    if isinstance(metadata, list):
        for record in metadata:
            if isinstance(record, Mapping):
                add(record.get("path"))
    return tuple(dict.fromkeys(paths))


def _validate_output_disjoint(path: Path, protected: tuple[Path, ...]) -> None:
    target = Path(path).resolve()
    for item in protected:
        immutable = Path(item).resolve()
        if target == immutable or target in immutable.parents or immutable in target.parents:
            raise ValueError(f"Output must be disjoint from immutable input: {immutable}")


def loss_identity(mode: str) -> dict[str, Any]:
    weighting = {
        "full_kl": "none_full_distribution_kl",
        "weighted_nckd": "teacher_probability_mass_outside_ground_truth",
    }
    if mode not in weighting:
        raise ValueError(f"Unknown H1 KD loss mode: {mode}")
    return {
        "format": 1,
        "mode": mode,
        "temperature": 2.0,
        "effective_weight": 1.0,
        "active": True,
        "non_target_weighting": weighting[mode],
    }


def validate_prepared_identity(prepared: Mapping[str, Any]) -> None:
    if prepared.get("status") != "prepared" or prepared.get("schema_version") != 1 or \
            prepared.get("protocol") != PROTOCOL:
        raise ValueError("H1 prepared receipt status/schema/protocol mismatch")
    if prepared.get("seed") != 1:
        raise ValueError("H1 prepared receipt must use fixed seed1")
    if prepared.get("mode") not in MODES:
        raise ValueError("H1 prepared receipt has an unknown loss mode")
    if prepared.get("source_sha256") != CURRENT_SOURCE_SHA256:
        raise ValueError("H1 prepared receipt source SHA-256 mismatch")
    if prepared.get("loss_identity") != loss_identity(prepared["mode"]):
        raise ValueError("H1 prepared receipt loss identity mismatch")


def load_reference_snapshot(path: Path = FIXED_REFERENCE_PATH) -> dict[str, Any]:
    target = Path(path).resolve()
    if probe.sha256_file(target) != FIXED_REFERENCE_SHA256:
        raise ValueError("Historical reference snapshot SHA-256 mismatch")
    return _read_json(target)


def validate_historical_reference(
        snapshot: Mapping[str, Any], dataset: str, seed: int = 1) -> dict[str, Any]:
    if dataset not in DATASETS or seed != 1 or snapshot.get("seed") != 1:
        raise ValueError("H1 historical reference requires a fixed dataset seed1 case")
    authoritative = load_reference_snapshot()
    if dict(snapshot) != authoritative:
        raise ValueError("Historical reference differs from the authenticated snapshot")
    expected_top = {
        "state": "reference_snapshot_only_not_executed",
        "candidate": "SN+M_logits",
        "alpha": 0.5,
        "shots": 16,
        "epochs": 20,
        "kd_loss_mode": "weighted_nckd",
        "temperature": 2.0,
        "kd_weight": 1.0,
        "proposed_max_new_training_runs": 4,
        "gpu_queue_created": False,
        "formal_training_started": False,
    }
    if any(snapshot.get(key) != value for key, value in expected_top.items()):
        raise ValueError("Historical reference protocol mismatch")
    cases = [copy.deepcopy(case) for case in snapshot.get("cases", [])
             if case.get("dataset") == dataset and case.get("seed") == seed]
    if len(cases) != 1:
        raise ValueError("Historical reference case is missing or duplicated")
    case = cases[0]
    required = {"dataset", "seed", "controls", "historical_export_sha256",
                "historical_logits_sha256"}
    if set(case) != required or any(
            not isinstance(case[field], str) or len(case[field]) != 64
            for field in ("historical_export_sha256", "historical_logits_sha256")):
        raise ValueError("Historical reference case schema mismatch")
    trusted = [row for row in authoritative["cases"]
               if row.get("dataset") == dataset and row.get("seed") == seed]
    if len(trusted) != 1 or case != trusted[0]:
        raise ValueError("Historical reference case differs from the authenticated snapshot")
    return case


def validate_historical_config(path: Path, dataset: str) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError("Unknown dataset historical config")
    target = Path(path).resolve()
    if not target.is_file() or probe.sha256_file(target) != HISTORICAL_CONFIG_SHA256[dataset]:
        raise ValueError("historical kd1 effective config is not the authenticated config")
    value = yaml.safe_load(target.read_text())
    if not isinstance(value, dict):
        raise ValueError("Historical kd1 effective config is malformed")
    tc = value.get("TRAINER", {}).get("TEXTCONSISTENCY", {})
    expected = {
        "SEED": 1,
        "DATASET.NAME": probe.DATASETS[dataset][0],
        "DATASET.NUM_SHOTS": 16,
        "DATASET.SUBSAMPLE_CLASSES": "base",
        "OPTIM.MAX_EPOCH": 20,
        "TRAINER.NAME": "CoOp_TextConsistency_V2_FG",
        "TRAINER.TEXTCONSISTENCY.KD_WEIGHT": 1.0,
        "TRAINER.TEXTCONSISTENCY.KD_TEMP": 2.0,
        "TRAINER.TEXTCONSISTENCY.ADAPTIVE_CE_WEIGHT": 1.0,
        "TRAINER.TEXTCONSISTENCY.DIVERSITY_WEIGHT": 0.0,
        "TRAINER.TEXTCONSISTENCY.CONTRASTIVE_WEIGHT": 0.0,
        "TRAINER.TEXTCONSISTENCY.AGGREGATION": "hierarchical",
        "TRAINER.TEXTCONSISTENCY.USE_FEATURE_ADAPTER": True,
    }
    actual = {
        "SEED": value.get("SEED"),
        "DATASET.NAME": value.get("DATASET", {}).get("NAME"),
        "DATASET.NUM_SHOTS": value.get("DATASET", {}).get("NUM_SHOTS"),
        "DATASET.SUBSAMPLE_CLASSES": value.get("DATASET", {}).get("SUBSAMPLE_CLASSES"),
        "OPTIM.MAX_EPOCH": value.get("OPTIM", {}).get("MAX_EPOCH"),
        "TRAINER.NAME": value.get("TRAINER", {}).get("NAME"),
        **{f"TRAINER.TEXTCONSISTENCY.{key}": tc.get(key) for key in (
            "KD_WEIGHT", "KD_TEMP", "ADAPTIVE_CE_WEIGHT", "DIVERSITY_WEIGHT",
            "CONTRASTIVE_WEIGHT", "AGGREGATION", "USE_FEATURE_ADAPTER")},
    }
    differing = sorted(key for key, expected_value in expected.items()
                       if actual.get(key) != expected_value)
    if differing:
        raise ValueError(f"Historical kd1 effective config protocol mismatch: {differing}")
    return value


def _resolve_npz(export_path: Path, receipt: Mapping[str, Any]) -> Path:
    npz = Path(receipt["npz"])
    if not npz.is_absolute():
        return export_path.parent / npz
    if npz.is_file():
        return npz
    return export_path.parent / npz.name


def validate_historical_export(
        path: Path, dataset: str, reference_case: Mapping[str, Any],
) -> dict[str, Any]:
    target = Path(path).resolve()
    if probe.sha256_file(target) != reference_case["historical_export_sha256"]:
        raise ValueError("Historical export JSON differs from the fixed reference snapshot")
    receipt, arrays = historical_analysis.load_export(
        target,
        trusted_inventory_path=FIXED_INVENTORY_PATH,
        trusted_inventory_sha256=FIXED_INVENTORY_SHA256,
    )
    if receipt.get("dataset_key") != dataset or receipt.get("seed") != 1:
        raise ValueError("Historical export dataset/seed mismatch")
    npz = _resolve_npz(target, receipt)
    if probe.sha256_file(npz) != reference_case["historical_logits_sha256"]:
        raise ValueError("Historical logits differ from the fixed reference snapshot")
    return {
        "export": str(target),
        "export_sha256": probe.sha256_file(target),
        "npz": str(npz.resolve()),
        "npz_sha256": probe.sha256_file(npz),
        "rows": copy.deepcopy(receipt["rows"]),
        "class_order": list(receipt["class_order"]),
        "branches": sorted(arrays),
    }


def validate_sample_binding(
        expected: list[Mapping[str, Any]], actual: list[Mapping[str, Any]], split: str,
) -> None:
    actual_ids = [row.get("sample_id") for row in actual]
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError(f"{split} sample binding contains duplicate IDs")
    expected_ids = [row.get("sample_id") for row in expected]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError(f"Trusted {split} sample binding contains duplicate IDs")
    fields = ("sample_id", "label", "classname")
    expected_core = [tuple(row.get(field) for field in fields) for row in expected]
    actual_core = [tuple(row.get(field) for field in fields) for row in actual]
    if expected_core != actual_core:
        raise ValueError(f"{split} sample order or identity differs from the prepared binding")


def image_content_receipt(rows: list[Mapping[str, Any]], data_root: Path) -> dict[str, Any]:
    root = Path(data_root).resolve()
    digest = hashlib.sha256()
    files, decoded_modes, decoded_sizes = [], set(), set()
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Image sample ID is missing")
        path = (root / sample_id).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Image escapes data root: {sample_id}") from exc
        file_sha = probe.sha256_file(path)
        with Image.open(path) as image:
            image.load()
            decoded_modes.add(str(image.mode))
            decoded_sizes.add(tuple(int(value) for value in image.size))
        record = {"sample_id": sample_id, "sha256": file_sha, "size": path.stat().st_size}
        files.append(record)
        digest.update(json.dumps({**record, "label": int(row["label"]),
                                  "classname": row["classname"]},
                                 sort_keys=True, ensure_ascii=False).encode())
    if not files:
        raise ValueError("Image content receipt cannot be empty")
    return {
        "count": len(files),
        "ordered_sha256": digest.hexdigest(),
        "files": files,
        "decoded_modes": sorted(decoded_modes),
        "decoded_sizes": [list(size) for size in sorted(decoded_sizes)],
    }


def validate_image_content(
        rows: list[Mapping[str, Any]], data_root: Path, expected: Mapping[str, Any], split: str,
) -> None:
    try:
        actual = image_content_receipt(rows, data_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ValueError(f"{split} image content could not be validated: {exc}") from exc
    if actual != dict(expected):
        raise ValueError(f"{split} image content differs from prepared evidence")


def _sample_core(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row.get(key) for key in ("sample_id", "label", "classname")} for row in rows]


def validate_fewshot_cache(
        cache_path: Path, training: Any, data_root: Path,
        expected_train: list[Mapping[str, Any]], expected_val: list[Mapping[str, Any]],
) -> dict[str, Any]:
    path = Path(cache_path).resolve()
    if not path.is_file():
        raise FileNotFoundError("Future training requires the authenticated existing few-shot cache")
    with path.open("rb") as handle:
        cached = pickle.load(handle)
    if not isinstance(cached, dict) or set(cached) != {"train", "val"}:
        raise ValueError("Few-shot cache schema must contain exactly train/val")
    pets = __import__("datasets.oxford_pets", fromlist=["OxfordPets"]).OxfordPets
    train, val = pets.subsample_classes(cached["train"], cached["val"], subsample="base")
    classnames = list(__import__("dassl.data.datasets", fromlist=["DatasetBase"])
                      .DatasetBase.get_lab2cname(train)[1])
    if not train or not val or not classnames or len(set(classnames)) != len(classnames):
        raise ValueError("Few-shot cache has empty or duplicated Base train/val classes")
    actual_train = probe._sample_rows(train, data_root, classnames)
    actual_val = probe._sample_rows(val, data_root, classnames)
    validate_sample_binding(expected_train, actual_train, "train cache")
    validate_sample_binding(expected_val, actual_val, "val cache")
    return {
        "path": str(path),
        "sha256": probe.sha256_file(path),
        "train": _sample_core(actual_train),
        "val": _sample_core(actual_val),
    }


def _config_args(
        config_path: Path, *, root: str = "", output_dir: str = "", seed: int | None = None,
        opts: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        root=root,
        output_dir=output_dir,
        resume="",
        seed=seed,
        source_domains=None,
        target_domains=None,
        transforms=None,
        trainer="",
        backbone="",
        head="",
        dataset_config_file="",
        config_file=str(config_path),
        opts=list(opts or []),
    )


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            result.update(_flatten(item, name))
        else:
            result[name] = item
    return result


def expected_model_signature(dataset: str, mode: str, template_sha256: str) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise ValueError("Unknown H1 dataset")
    identity = loss_identity(mode)
    return {
        "format": 2,
        "trainer": "CoOp_TextConsistency_V2_FG",
        "n_ctx": 4,
        "aggregation": "hierarchical",
        "feature_adapter": True,
        "n_groups": 4,
        "n_views": 20,
        "ablation_group": None,
        "clip_sha256": probe.CLIP_SHA256,
        "template_sha256": template_sha256,
        "view_order": "contiguous",
        "order_seed": 0,
        "order_version": 1,
        "loss_identity": identity,
    }


def _validate_future_config(cfg: Any, dataset: str, mode: str, source_root: Path) -> None:
    tc = cfg.TRAINER.TEXTCONSISTENCY
    checks = {
        "seed": (int(cfg.SEED), 1),
        "dataset": (cfg.DATASET.NAME, probe.DATASETS[dataset][0]),
        "shots": (int(cfg.DATASET.NUM_SHOTS), 16),
        "subsample": (cfg.DATASET.SUBSAMPLE_CLASSES, "base"),
        "epochs": (int(cfg.OPTIM.MAX_EPOCH), 20),
        "trainer": (cfg.TRAINER.NAME, "CoOp_TextConsistency_V2_FG"),
        "backbone": (cfg.MODEL.BACKBONE.NAME, "ViT-B/16"),
        "precision": (tc.PREC, "fp32"),
        "n_ctx": (int(tc.N_CTX), 4),
        "n_views": (int(tc.N_VIEWS), 20),
        "n_groups": (int(tc.N_GROUPS), 4),
        "aggregation": (tc.AGGREGATION, "hierarchical"),
        "feature_adapter": (bool(tc.USE_FEATURE_ADAPTER), True),
        "kd_weight": (float(tc.KD_WEIGHT), 1.0),
        "kd_temp": (float(tc.KD_TEMP), 2.0),
        "kd_loss_mode": (tc.KD_LOSS_MODE, mode),
        "teacher_ce": (float(tc.ADAPTIVE_CE_WEIGHT), 1.0),
        "diversity": (float(tc.DIVERSITY_WEIGHT), 0.0),
        "contrastive": (float(tc.CONTRASTIVE_WEIGHT), 0.0),
        "train_sampler": (cfg.DATALOADER.TRAIN_X.SAMPLER, "RandomSampler"),
        "train_batch": (int(cfg.DATALOADER.TRAIN_X.BATCH_SIZE), 16),
        "workers": (int(cfg.DATALOADER.NUM_WORKERS), 4),
        "test_split": (cfg.TEST.SPLIT, "val"),
        "no_test": (bool(cfg.TEST.NO_TEST), True),
        "final_model": (cfg.TEST.FINAL_MODEL, "last_step"),
        "test_sampler": (cfg.DATALOADER.TEST.SAMPLER, "SequentialSampler"),
        "cuda": (bool(cfg.USE_CUDA), True),
        "device": (tc.DEVICE, "cuda"),
    }
    differing = sorted(key for key, (actual, expected) in checks.items() if actual != expected)
    clip = Path(tc.CLIP_PATH).expanduser().resolve()
    if clip != (Path(source_root).resolve() / "clip/ViT-B-16.pt").resolve():
        differing.append("clip_path")
    if differing:
        raise ValueError(f"Future H1 configuration mismatch: {differing}")


def build_future_config(
        training: Any, historical_config: Path, data_root: Path, source_root: Path,
        output_placeholder: Path, dataset: str, mode: str,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    if mode not in MODES:
        raise ValueError("Unknown H1 KD loss mode")
    baseline = training.setup_cfg(_config_args(Path(historical_config)))
    opts = [
        "TEST.NO_TEST", "True",
        "TEST.FINAL_MODEL", "last_step",
        "TEST.SPLIT", "val",
        "DATALOADER.TEST.SAMPLER", "SequentialSampler",
        "USE_CUDA", "True",
        "TRAINER.TEXTCONSISTENCY.DEVICE", "cuda",
        "TRAINER.TEXTCONSISTENCY.CLIP_PATH", str(Path(source_root).resolve() / "clip/ViT-B-16.pt"),
        "TRAINER.TEXTCONSISTENCY.KD_LOSS_MODE", mode,
    ]
    cfg = training.setup_cfg(_config_args(
        Path(historical_config), root=str(Path(data_root).resolve()),
        output_dir=str(Path(output_placeholder).resolve()), seed=1, opts=opts))
    baseline_dict = yaml.safe_load(baseline.dump())
    future = yaml.safe_load(cfg.dump())
    allowed = {
        "DATASET.ROOT", "OUTPUT_DIR", "TEST.NO_TEST", "TEST.FINAL_MODEL", "TEST.SPLIT",
        "DATALOADER.TEST.SAMPLER", "USE_CUDA", "TRAINER.TEXTCONSISTENCY.DEVICE",
        "TRAINER.TEXTCONSISTENCY.CLIP_PATH", "TRAINER.TEXTCONSISTENCY.KD_LOSS_MODE",
    }
    old_flat, future_flat = _flatten(baseline_dict), _flatten(future)
    changed = {key for key in set(old_flat) | set(future_flat) if old_flat.get(key) != future_flat.get(key)}
    unexpected = sorted(changed - allowed)
    if unexpected:
        raise ValueError(f"Future config changed unauthorised historical settings: {unexpected}")
    _validate_future_config(cfg, dataset, mode, source_root)
    runtime = __import__("trainers.textconsistency_runtime", fromlist=["ValidatedTrainer"])
    runtime.ValidatedTrainer.check_cfg(None, cfg)
    preserved = {key: future_flat[key] for key in sorted(future_flat) if key not in allowed}
    return cfg, future, {
        "historical_normalized_sha256": probe.json_digest(baseline_dict),
        "preserved_settings_sha256": probe.json_digest(preserved),
        "changed_fields": sorted(changed),
    }


def _prepared_identity(prepared: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": prepared.get("schema_version"),
        "dataset_key": prepared.get("dataset_key"),
        "seed": prepared.get("seed"),
        "mode": prepared.get("mode"),
        "loss_identity": prepared.get("loss_identity"),
        "source_sha256": prepared.get("source_sha256"),
        "assets": prepared.get("assets"),
        "historical": prepared.get("historical"),
        "historical_config": prepared.get("historical_config"),
        "sample_hashes": prepared.get("sample_hashes"),
        "image_content": prepared.get("image_content"),
        "fewshot_cache": prepared.get("fewshot_cache"),
        "future_config_sha256": prepared.get("future_config_sha256"),
        "future_config_payload_sha256": prepared.get("future_config_payload_sha256"),
        "expected_model_signature": prepared.get("expected_model_signature"),
    }


def prepare_case(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve()
    data_root = Path(args.data_root).resolve()
    historical_export = Path(args.historical_export).resolve()
    historical_config = Path(args.historical_config).resolve()
    output = _fresh_output(Path(args.output_dir), (
        source_root, data_root, historical_export, historical_config,
        FIXED_REFERENCE_PATH, FIXED_INVENTORY_PATH))
    dataset, mode = args.dataset, args.mode
    if dataset not in DATASETS or mode not in MODES:
        raise ValueError("Unknown H1 dataset or mode")
    if probe.sha256_file(FIXED_INVENTORY_PATH) != FIXED_INVENTORY_SHA256:
        raise ValueError("Fixed inventory SHA-256 mismatch")
    inventory = _read_json(FIXED_INVENTORY_PATH)
    case = probe.validate_inventory(inventory, dataset, 1)
    reference = load_reference_snapshot()
    reference_case = validate_historical_reference(reference, dataset, 1)
    historical = validate_historical_export(historical_export, dataset, reference_case)
    validate_historical_config(historical_config, dataset)

    source_sha = probe.model_source_digest(source_root)
    if source_sha != CURRENT_SOURCE_SHA256:
        raise ValueError("Current training source SHA-256 mismatch")
    clip_path = source_root / "clip/ViT-B-16.pt"
    template_path = source_root / "templates" / probe.DATASETS[dataset][2]
    if probe.sha256_file(clip_path) != case["clip_sha256"]:
        raise ValueError("Current CLIP asset SHA-256 mismatch")
    if probe.sha256_file(template_path) != case["template_sha256"]:
        raise ValueError("Current template SHA-256 mismatch")

    training = probe._source_modules(source_root)
    future_config_path = output / "future_config.yaml"
    cfg, future_config, config_derivation = build_future_config(
        training, historical_config, data_root, source_root,
        output / "future-model-placeholder", dataset, mode)
    train, val, test, classnames, metadata_paths = probe.rebuild_source_dataset(training, cfg, dataset)
    samples = probe.validate_samples(train, val, test, data_root, classnames)
    sample_hashes = {
        "train": probe.sample_digest(train, data_root),
        "val": probe.sample_digest(val, data_root),
        "test": probe.sample_digest(test, data_root),
    }
    if sample_hashes["train"] != case["train_samples_sha256"] or \
            sample_hashes["val"] != case["val_samples_sha256"]:
        raise ValueError("Cache-free rebuilt train/val samples differ from the authenticated inventory")
    validate_sample_binding(historical["rows"], samples["val"], "historical val")
    if historical["class_order"] != classnames:
        raise ValueError("Historical validation class order differs from current rebuild")
    image_content = {
        "train": image_content_receipt(samples["train"], data_root),
        "val": image_content_receipt(samples["val"], data_root),
    }
    _, folder, _ = probe.DATASETS[dataset]
    cache_path = data_root / folder / "split_fewshot" / "shot_16-seed_1.pkl"
    cache = validate_fewshot_cache(cache_path, training, data_root,
                                   samples["train"], samples["val"])
    loader, transform = probe.build_val_loader(cfg, val)
    decoded_val = probe._decode_evidence(loader)
    source_metadata = [{"path": str(Path(path).resolve()), "sha256": probe.sha256_file(path)}
                       for path in metadata_paths]
    config_text = cfg.dump()
    config_payload_sha = probe.json_digest(future_config)
    model_signature = expected_model_signature(dataset, mode, case["template_sha256"])
    historical_config_record = {
        "path": str(historical_config),
        "sha256": probe.sha256_file(historical_config),
        "kd1_checkpoint_sha256": case["checkpoints"]["kd1"]["sha256"],
    }
    historical_config_record["bundle_sha256"] = probe.json_digest(historical_config_record)
    receipt = {
        "status": "prepared",
        "schema_version": 1,
        "dataset_key": dataset,
        "dataset": probe.DATASETS[dataset][0],
        "seed": 1,
        "mode": mode,
        "loss_identity": loss_identity(mode),
        "protocol": PROTOCOL,
        "source_root": str(source_root),
        "data_root": str(data_root),
        "source_sha256": source_sha,
        "assets": {
            "clip": str(clip_path), "clip_sha256": probe.sha256_file(clip_path),
            "template": str(template_path), "template_sha256": probe.sha256_file(template_path),
        },
        "inventory": {
            "path": str(FIXED_INVENTORY_PATH.resolve()),
            "sha256": FIXED_INVENTORY_SHA256,
            "case": case,
        },
        "reference": {
            "path": str(FIXED_REFERENCE_PATH.resolve()),
            "sha256": FIXED_REFERENCE_SHA256,
            "case": reference_case,
        },
        "historical": historical,
        "historical_config": historical_config_record,
        "samples": samples,
        "sample_hashes": sample_hashes,
        "image_content": image_content,
        "fewshot_cache": cache,
        "source_metadata": source_metadata,
        "future_config_path": str(future_config_path),
        "future_config": future_config,
        "future_config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
        "future_config_payload_sha256": config_payload_sha,
        "config_derivation": config_derivation,
        "expected_model_signature": model_signature,
        "expected_trainable_parameters": 790018,
        "transform": {
            "repr": repr(transform),
            "repr_sha256": hashlib.sha256(repr(transform).encode()).hexdigest(),
            "decoded_val": decoded_val,
        },
        "tool": {
            "h1_validation_sha256": probe.sha256_file(Path(__file__).resolve()),
            "validation_probe_sha256": probe.sha256_file(Path(probe.__file__).resolve()),
            "validation_analysis_sha256": probe.sha256_file(Path(historical_analysis.__file__).resolve()),
        },
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__, "device": "cpu",
        },
        "limitations": ["Prepared without model, optimizer, CUDA, training, or test inference"],
    }
    receipt["prepared_identity_sha256"] = probe.json_digest(_prepared_identity(receipt))
    output.mkdir(parents=True)
    future_config_path.write_text(config_text)
    _atomic_json(output / "prepared.json", receipt)
    return receipt


def load_prepared(path: Path) -> dict[str, Any]:
    prepared_path = Path(path).resolve()
    prepared = _read_json(prepared_path)
    if prepared.get("status") != "prepared" or prepared.get("schema_version") != 1 or \
            prepared.get("protocol") != PROTOCOL:
        raise ValueError("Prepared receipt status/schema/protocol mismatch")
    validate_prepared_identity(prepared)
    if prepared.get("dataset_key") not in DATASETS or prepared.get("dataset") != \
            probe.DATASETS[prepared["dataset_key"]][0]:
        raise ValueError("Prepared dataset identity mismatch")
    if prepared.get("prepared_identity_sha256") != probe.json_digest(_prepared_identity(prepared)):
        raise ValueError("Prepared receipt immutable identity mismatch")
    if probe.sha256_file(FIXED_INVENTORY_PATH) != FIXED_INVENTORY_SHA256:
        raise ValueError("Fixed inventory SHA-256 mismatch")
    case = probe.validate_inventory(_read_json(FIXED_INVENTORY_PATH), prepared["dataset_key"], 1)
    inventory_record = prepared.get("inventory", {})
    if inventory_record.get("path") != str(FIXED_INVENTORY_PATH.resolve()) or \
            inventory_record.get("sha256") != FIXED_INVENTORY_SHA256 or \
            inventory_record.get("case") != case:
        raise ValueError("Prepared inventory binding mismatch")
    reference = load_reference_snapshot()
    reference_case = validate_historical_reference(reference, prepared["dataset_key"], 1)
    if prepared.get("reference", {}).get("path") != str(FIXED_REFERENCE_PATH.resolve()) or \
            prepared.get("reference", {}).get("sha256") != FIXED_REFERENCE_SHA256 or \
            prepared["reference"].get("case") != reference_case:
        raise ValueError("Prepared historical reference binding mismatch")
    historical = validate_historical_export(
        Path(prepared["historical"]["export"]), prepared["dataset_key"], reference_case)
    if historical != prepared["historical"]:
        raise ValueError("Prepared historical export binding changed")
    config = validate_historical_config(
        Path(prepared["historical_config"]["path"]), prepared["dataset_key"])
    del config
    expected_config_record = {
        "path": str(Path(prepared["historical_config"]["path"]).resolve()),
        "sha256": HISTORICAL_CONFIG_SHA256[prepared["dataset_key"]],
        "kd1_checkpoint_sha256": case["checkpoints"]["kd1"]["sha256"],
    }
    expected_config_record["bundle_sha256"] = probe.json_digest(expected_config_record)
    if prepared["historical_config"] != expected_config_record:
        raise ValueError("Historical config/checkpoint bundle binding mismatch")

    source_root, data_root = Path(prepared["source_root"]), Path(prepared["data_root"])
    if probe.model_source_digest(source_root) != CURRENT_SOURCE_SHA256:
        raise ValueError("Prepared current source changed")
    for key in ("clip", "template"):
        asset = Path(prepared["assets"][key])
        if probe.sha256_file(asset) != prepared["assets"][f"{key}_sha256"]:
            raise ValueError(f"Prepared {key} asset changed")
    if prepared["assets"]["clip_sha256"] != case["clip_sha256"] or \
            prepared["assets"]["template_sha256"] != case["template_sha256"]:
        raise ValueError("Prepared assets differ from the fixed inventory")
    for record in prepared.get("source_metadata", []):
        if probe.sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError("Prepared source metadata changed")
    future_config_path = Path(prepared["future_config_path"])
    config_text = future_config_path.read_text()
    if hashlib.sha256(config_text.encode()).hexdigest() != prepared["future_config_sha256"]:
        raise ValueError("Saved future config file changed")
    future_config = yaml.safe_load(config_text)
    if future_config != prepared["future_config"] or \
            probe.json_digest(future_config) != prepared["future_config_payload_sha256"]:
        raise ValueError("Saved future config payload mismatch")

    training = probe._source_modules(source_root)
    cfg = training.setup_cfg(_config_args(future_config_path))
    _validate_future_config(cfg, prepared["dataset_key"], prepared["mode"], source_root)
    runtime = __import__("trainers.textconsistency_runtime", fromlist=["ValidatedTrainer"])
    runtime.ValidatedTrainer.check_cfg(None, cfg)
    train, val, test, classnames, metadata_paths = probe.rebuild_source_dataset(
        training, cfg, prepared["dataset_key"])
    samples = probe.validate_samples(train, val, test, data_root, classnames)
    if samples != prepared["samples"]:
        raise ValueError("Cache-free rebuilt samples/classes differ from prepared receipt")
    sample_hashes = {"train": probe.sample_digest(train, data_root),
                     "val": probe.sample_digest(val, data_root),
                     "test": probe.sample_digest(test, data_root)}
    if sample_hashes != prepared["sample_hashes"] or \
            sample_hashes["train"] != case["train_samples_sha256"] or \
            sample_hashes["val"] != case["val_samples_sha256"]:
        raise ValueError("Prepared sample hashes changed")
    validate_sample_binding(historical["rows"], samples["val"], "historical val")
    for split in ("train", "val"):
        validate_image_content(samples[split], data_root, prepared["image_content"][split], split)
    cache_record = prepared["fewshot_cache"]
    if probe.sha256_file(Path(cache_record["path"])) != cache_record["sha256"]:
        raise ValueError("Few-shot cache file changed")
    cache = validate_fewshot_cache(Path(cache_record["path"]), training, data_root,
                                   samples["train"], samples["val"])
    if cache != cache_record:
        raise ValueError("Few-shot cache sample binding changed")
    current_metadata = [{"path": str(Path(item).resolve()), "sha256": probe.sha256_file(item)}
                        for item in metadata_paths]
    if current_metadata != prepared["source_metadata"]:
        raise ValueError("Source dataset metadata binding changed")
    loader, transform = probe.build_val_loader(cfg, val)
    if hashlib.sha256(repr(transform).encode()).hexdigest() != prepared["transform"]["repr_sha256"]:
        raise ValueError("Validation transform identity changed")
    probe.validate_decoded_transform(probe._decode_evidence(loader),
                                     prepared["transform"]["decoded_val"])
    tool = prepared.get("tool", {})
    expected_tools = {
        "h1_validation_sha256": probe.sha256_file(Path(__file__).resolve()),
        "validation_probe_sha256": probe.sha256_file(Path(probe.__file__).resolve()),
        "validation_analysis_sha256": probe.sha256_file(Path(historical_analysis.__file__).resolve()),
    }
    if tool != expected_tools:
        raise ValueError("Prepared operation tool hashes changed")
    return prepared


def build_train_command(
        prepared: Mapping[str, Any], model_dir: Path, python: str = sys.executable,
) -> list[str]:
    validate_prepared_identity(prepared)
    target = _fresh_output(Path(model_dir), (
        Path(prepared["source_root"]), Path(prepared["data_root"]),
        Path(prepared["future_config_path"])))
    train_py = Path(prepared["source_root"]).resolve() / "train.py"
    if not train_py.is_file():
        raise FileNotFoundError("Current source train.py is missing")
    return [str(python), str(train_py), "--root", str(Path(prepared["data_root"]).resolve()),
            "--output-dir", str(target), "--seed", "1", "--config-file",
            str(Path(prepared["future_config_path"]).resolve())]


def strict_load_current_model(model: torch.nn.Module, checkpoint: Mapping[str, Any]) -> None:
    probe.strict_load_trainable_modules(model, checkpoint)


def export_batches(
        model: torch.nn.Module, batches: Any, description_features: torch.Tensor,
        view_mask: torch.Tensor, classnames: list[str], *, device: Any | None = None,
) -> dict[str, Any]:
    if getattr(model, "visual_adapter", None) is not None:
        raise ValueError("Visual adapter features are outside the historical H1 export protocol")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    descriptions = description_features.float()
    if not torch.isfinite(descriptions).all().item():
        raise ValueError("Description features contain non-finite values")
    if descriptions.ndim != 3:
        raise ValueError("Description features must have shape [C,V,D]")
    mask = view_mask.to(device=descriptions.device, dtype=torch.bool)
    if mask.shape != descriptions.shape[:2] or not mask.any(dim=1).all().item():
        raise ValueError("Description mask shape is invalid or leaves an empty class")
    descriptions = F.normalize(descriptions, dim=-1)
    weights = mask.to(descriptions.dtype) / mask.sum(dim=1, keepdim=True)
    mean_features = F.normalize(torch.einsum("cv,cvd->cd", weights, descriptions), dim=-1)
    if not torch.isfinite(mean_features).all().item():
        raise ValueError("Description mean produced non-finite values")
    if descriptions.shape[0] != len(classnames) or len(set(classnames)) != len(classnames):
        raise ValueError("Description features differ from the class order")
    if device is None:
        device = next(model.parameters()).device
    collected = {name: [] for name in BRANCHES}
    labels, sample_ids = [], []
    digest, decoded_count, decoded_shape, decoded_dtype = hashlib.sha256(), 0, None, None
    precision = set()
    with torch.inference_mode():
        for batch in batches:
            count, decoded_shape, decoded_dtype = probe._update_decode_digest(digest, batch)
            decoded_count += count
            images = batch["img"].to(device)
            student, teacher, image_features, _ = model(images, return_attn=False)
            if teacher is None:
                raise ValueError("Current H1 model omitted the teacher branch")
            scale = model.logit_scale.exp().detach().float()
            mean_logits = scale * F.normalize(image_features.float(), dim=-1) @ mean_features.to(
                image_features.device).t()
            ensemble = 0.5 * (student + teacher)
            values = {"S": student, "T": teacher, "M": mean_logits, "ensemble": ensemble}
            shape = tuple(student.shape)
            if len(shape) != 2 or shape[1] != len(classnames) or \
                    any(tuple(value.shape) != shape for value in values.values()):
                raise ValueError("Export branch shape differs from [N,C] class order")
            if any(not torch.isfinite(value).all().item() for value in values.values()):
                raise ValueError("Export branch contains non-finite logits")
            for name, value in values.items():
                precision.add(str(value.dtype).replace("torch.", ""))
                collected[name].append(value.float().cpu().numpy())
            batch_labels = batch["label"].detach().cpu().numpy().astype(np.int64, copy=False)
            paths = [str(item) for item in batch["impath"]]
            if len(paths) != len(batch_labels):
                raise ValueError("Export batch sample IDs and labels are misaligned")
            labels.append(batch_labels)
            sample_ids.extend(paths)
    if not labels:
        raise ValueError("Export validation loader produced no batches")
    label_array = np.concatenate(labels)
    if np.any(label_array < 0) or np.any(label_array >= len(classnames)) or \
            len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Export labels or sample IDs are invalid")
    return {
        "logits": {name: np.concatenate(values).astype(np.float32, copy=False)
                   for name, values in collected.items()},
        "labels": label_array,
        "sample_ids": sample_ids,
        "classnames": list(classnames),
        "decoded_val": {"count": decoded_count, "tensor_shape": decoded_shape,
                        "tensor_dtype": decoded_dtype,
                        "ordered_transform_sha256": digest.hexdigest()},
        "forward_precision": sorted(precision) if len(precision) != 1 else next(iter(precision)),
    }


def write_export_artifact(
        output_dir: Path, payload: Mapping[str, Any], prepared_path: Path,
        prepared: Mapping[str, Any], checkpoint: Mapping[str, Any], *,
        existing_empty_ok: bool = False,
) -> dict[str, Any]:
    target = Path(output_dir).resolve()
    if target.exists() and (not existing_empty_ok or any(target.iterdir())):
        raise FileExistsError(f"Output must be a fresh path: {target}")
    logits = payload.get("logits", {})
    if set(logits) != set(BRANCHES):
        raise ValueError("Export requires exactly S/T/M/ensemble raw branches")
    labels = np.asarray(payload["labels"])
    classnames = list(payload["classnames"])
    sample_ids = list(payload["sample_ids"])
    if labels.dtype != np.int64 or labels.ndim != 1:
        raise ValueError("Export labels must be one-dimensional int64")
    n, c = len(labels), len(classnames)
    arrays = {name: np.asarray(logits[name]) for name in BRANCHES}
    if n == 0 or c < 2 or len(set(classnames)) != c or \
            any(value.dtype != np.float32 or value.shape != (n, c) or
                not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Export logits must be finite float32 [N,C] arrays")
    if len(sample_ids) != n or len(set(sample_ids)) != n or \
            np.any(labels < 0) or np.any(labels >= c):
        raise ValueError("Export sample IDs/labels are invalid")
    target.mkdir(parents=True, exist_ok=True)
    npz_path = target / "logits.npz"
    np.savez(npz_path, labels=labels, **arrays)
    rows = [{"row": index, "sample_id": str(sample_ids[index]), "label": int(label),
             "classname": classnames[int(label)]}
            for index, label in enumerate(labels)]
    prepared_file = Path(prepared_path).resolve()
    receipt = {
        "status": "passed",
        "schema_version": 1,
        "dataset_key": prepared["dataset_key"],
        "dataset": prepared["dataset"],
        "seed": prepared["seed"],
        "mode": prepared["mode"],
        "loss_identity": copy.deepcopy(prepared["loss_identity"]),
        "protocol": PROTOCOL,
        "branches": list(BRANCHES),
        "npz": "logits.npz",
        "npz_sha256": probe.sha256_file(npz_path),
        "prepared": str(prepared_file),
        "prepared_sha256": probe.sha256_file(prepared_file),
        "prepared_identity_sha256": prepared["prepared_identity_sha256"],
        "checkpoint": copy.deepcopy(dict(checkpoint)),
        "rows": rows,
        "class_order": classnames,
        "sample_hashes": copy.deepcopy(prepared["sample_hashes"]),
        "source_sha256": prepared["source_sha256"],
        "transform": copy.deepcopy(prepared.get("transform")),
        "evaluated_transform": copy.deepcopy(payload["decoded_val"]),
        "forward_precision": payload["forward_precision"],
        "storage_precision": "float32",
        "tool": {"h1_validation_sha256": probe.sha256_file(Path(__file__).resolve())},
    }
    _atomic_json(target / "export.json", receipt)
    return receipt


def _load_npz(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if probe.sha256_file(path) != expected_sha256:
        raise ValueError("Export NPZ SHA-256 mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"labels", *BRANCHES}:
            raise ValueError("Export NPZ schema has missing or extra arrays")
        return {name: archive[name].copy() for name in archive.files}


def _validate_checkpoint_record(record: Mapping[str, Any], prepared: Mapping[str, Any]) -> None:
    required = {"checkpoint_sha256", "manifest_sha256", "effective_config_sha256",
                "loss_identity", "source_sha256"}
    if not required.issubset(record):
        raise ValueError("Export checkpoint binding schema is incomplete")
    if record.get("loss_identity") != prepared["loss_identity"] or \
            record.get("source_sha256") != prepared["source_sha256"]:
        raise ValueError("Export checkpoint source/loss identity mismatch")
    for key in ("checkpoint_sha256", "manifest_sha256", "effective_config_sha256"):
        value = record.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Export checkpoint {key} is invalid")
    for path_key, hash_key in (("checkpoint", "checkpoint_sha256"),
                               ("manifest", "manifest_sha256"),
                               ("effective_config", "effective_config_sha256")):
        if path_key in record and probe.sha256_file(Path(record[path_key])) != record[hash_key]:
            raise ValueError(f"Export checkpoint {path_key} file changed")
    if "sidecar" in record and probe.sha256_file(Path(record["sidecar"])) != record.get("sidecar_sha256"):
        raise ValueError("Export checkpoint sidecar file changed")


def load_export(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    export_path = Path(path).resolve()
    receipt = _read_json(export_path)
    if receipt.get("status") != "passed" or receipt.get("schema_version") != 1 or \
            receipt.get("protocol") != PROTOCOL or receipt.get("branches") != list(BRANCHES):
        raise ValueError("Export status/schema/protocol mismatch")
    prepared_path = Path(receipt["prepared"])
    if not prepared_path.is_absolute():
        prepared_path = export_path.parent / prepared_path
    if probe.sha256_file(prepared_path) != receipt.get("prepared_sha256"):
        raise ValueError("Export prepared receipt SHA-256 mismatch")
    prepared = load_prepared(prepared_path)
    expected_identity = {
        "dataset_key": prepared["dataset_key"], "dataset": prepared["dataset"],
        "seed": prepared["seed"], "mode": prepared["mode"],
        "loss_identity": prepared["loss_identity"], "source_sha256": prepared["source_sha256"],
        "prepared_identity_sha256": prepared["prepared_identity_sha256"],
        "sample_hashes": prepared["sample_hashes"],
    }
    differing = sorted(key for key, value in expected_identity.items() if receipt.get(key) != value)
    if differing:
        raise ValueError(f"Export prepared/source/mode binding mismatch: {differing}")
    checkpoint_record = receipt.get("checkpoint", {})
    _validate_checkpoint_record(checkpoint_record, prepared)
    if "model_dir" not in checkpoint_record:
        raise ValueError("Export checkpoint binding omits model_dir")
    current_checkpoint, _ = validate_new_checkpoint(prepared, Path(checkpoint_record["model_dir"]))
    if current_checkpoint != checkpoint_record:
        raise ValueError("Export checkpoint binding differs from the validated epoch20 bundle")
    npz = Path(receipt["npz"])
    if not npz.is_absolute():
        npz = export_path.parent / npz
    arrays = _load_npz(npz, receipt["npz_sha256"])
    labels = arrays["labels"]
    logits = {name: arrays[name] for name in BRANCHES}
    classes = receipt.get("class_order", [])
    rows = receipt.get("rows", [])
    expected_rows = _sample_core(prepared["samples"]["val"])
    actual_rows = [{key: row.get(key) for key in ("sample_id", "label", "classname")}
                   for row in rows]
    if labels.dtype != np.int64 or labels.ndim != 1 or classes != prepared["samples"]["classes"] or \
            actual_rows != expected_rows or [row.get("row") for row in rows] != list(range(len(rows))) or \
            [row.get("label") for row in rows] != labels.tolist():
        raise ValueError("Export rows/classes/labels differ from prepared validation rows")
    if len({row.get("sample_id") for row in rows}) != len(rows):
        raise ValueError("Export rows contain duplicate sample IDs")
    n, c = len(labels), len(classes)
    for name, value in logits.items():
        if value.dtype != np.float32 or value.shape != (n, c) or not np.isfinite(value).all():
            raise ValueError(f"Export {name} is not finite float32 [N,C]")
    if not np.array_equal(logits["ensemble"],
                          0.5 * (logits["S"] + logits["T"])):
        raise ValueError("Export ensemble differs from the fixed alpha=.5 formula")
    if receipt.get("transform") != prepared.get("transform") or \
            receipt.get("evaluated_transform") != prepared.get("transform", {}).get("decoded_val"):
        raise ValueError("Export transform verification differs from prepared evidence")
    if receipt.get("tool", {}).get("h1_validation_sha256") != \
            probe.sha256_file(Path(__file__).resolve()):
        raise ValueError("Export operation tool hash changed")
    return receipt, arrays


def validate_new_checkpoint(prepared: Mapping[str, Any], model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(model_dir).resolve()
    manifest_path = root / "run_manifest.json"
    config_path = root / "effective_config.yaml"
    checkpoint_path = root / "textconsistency/model.pth.tar-20"
    manifest = _read_json(manifest_path)
    actual_config = yaml.safe_load(config_path.read_text())
    if not isinstance(actual_config, dict):
        raise ValueError("New checkpoint effective config is malformed")
    expected_config = copy.deepcopy(prepared["future_config"])
    expected_config["OUTPUT_DIR"] = str(root)
    if actual_config != expected_config:
        differing = sorted(key for key in set(_flatten(actual_config)) | set(_flatten(expected_config))
                           if _flatten(actual_config).get(key) != _flatten(expected_config).get(key))
        raise ValueError(f"New checkpoint effective config differs from prepared config: {differing}")
    expected_manifest = {
        "dataset": prepared["dataset"],
        "seed": 1,
        "shots": 16,
        "subsample": "base",
        "code_sha256": CURRENT_SOURCE_SHA256,
        "loss_identity": prepared["loss_identity"],
        "effective_losses": {"kd": 1.0, "adaptive_ce": 1.0,
                             "diversity": 0.0, "contrastive": 0.0},
        "model_signature": prepared["expected_model_signature"],
        "train_samples_sha256": prepared["sample_hashes"]["train"],
        "val_samples_sha256": prepared["sample_hashes"]["val"],
        "test_samples_sha256": prepared["sample_hashes"]["test"],
        "trainable_parameters": prepared["expected_trainable_parameters"],
    }
    differing = sorted(key for key, value in expected_manifest.items() if manifest.get(key) != value)
    if differing:
        raise ValueError(f"New checkpoint manifest identity mismatch: {differing}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("epoch") != 20 or \
            checkpoint.get("repro_signature") != prepared["expected_model_signature"]:
        raise ValueError("New checkpoint epoch or model signature mismatch")
    provenance = checkpoint.get("run_provenance", {})
    for key, value in expected_manifest.items():
        if provenance.get(key) != value:
            raise ValueError(f"New checkpoint provenance mismatch: {key}")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state or any(
            not torch.is_tensor(value) or not torch.isfinite(value).all().item()
            for value in state.values()):
        raise ValueError("New checkpoint state_dict is missing or non-finite")
    trainable = {key: value for key, value in state.items()
                 if not key.endswith(("token_prefix", "token_suffix"))}
    if sum(value.numel() for value in trainable.values()) != prepared["expected_trainable_parameters"]:
        raise ValueError("New checkpoint does not contain the complete trainable module state")
    checkpoint_sha = probe.sha256_file(checkpoint_path)
    sidecar = checkpoint_path.with_name(checkpoint_path.name + ".sha256")
    if sidecar.is_file() and sidecar.read_text().strip() != checkpoint_sha:
        raise ValueError("New checkpoint sidecar SHA-256 mismatch")
    record = {
        "model_dir": str(root),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "manifest": str(manifest_path),
        "manifest_sha256": probe.sha256_file(manifest_path),
        "effective_config": str(config_path),
        "effective_config_sha256": probe.sha256_file(config_path),
        "loss_identity": copy.deepcopy(prepared["loss_identity"]),
        "source_sha256": prepared["source_sha256"],
    }
    if sidecar.is_file():
        record.update({"sidecar": str(sidecar), "sidecar_sha256": probe.sha256_file(sidecar)})
    return record, checkpoint


def export_case(args: argparse.Namespace) -> dict[str, Any]:
    prepared_path = Path(args.prepared).resolve()
    model_dir = Path(args.model_dir).resolve()
    unvalidated_prepared = _read_json(prepared_path)
    protected = _export_protected_paths(prepared_path, model_dir, unvalidated_prepared)
    output = _fresh_output(Path(args.output_dir), protected)
    output.mkdir(parents=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for H1 export; CPU fallback is forbidden")
        prepared = load_prepared(prepared_path)
        _validate_output_disjoint(
            output, _export_protected_paths(prepared_path, model_dir, prepared))
        checkpoint_record, checkpoint = validate_new_checkpoint(prepared, model_dir)
        source_root, data_root = Path(prepared["source_root"]), Path(prepared["data_root"])
        training = probe._source_modules(source_root)
        opts = ["DATALOADER.NUM_WORKERS", "0", "DATALOADER.TEST.SAMPLER", "SequentialSampler",
                "TEST.SPLIT", "val", "USE_CUDA", "True",
                "TRAINER.TEXTCONSISTENCY.DEVICE", "cuda"]
        cfg = training.setup_cfg(_config_args(Path(checkpoint_record["effective_config"]), opts=opts))
        runtime = __import__("trainers.textconsistency_runtime", fromlist=["ValidatedTrainer"])
        runtime.ValidatedTrainer.check_cfg(None, cfg)
        if cfg.TEST.SPLIT != "val" or cfg.DATASET.SUBSAMPLE_CLASSES != "base" or \
                cfg.DATALOADER.NUM_WORKERS != 0 or \
                cfg.DATALOADER.TEST.SAMPLER != "SequentialSampler":
            raise ValueError("H1 export config differs from sequential Base validation")
        train, val, test, classnames, _ = probe.rebuild_source_dataset(
            training, cfg, prepared["dataset_key"])
        samples = probe.validate_samples(train, val, test, data_root, classnames)
        if samples != prepared["samples"]:
            raise ValueError("H1 export rebuilt samples/classes differ from prepare")
        loader, transform = probe.build_val_loader(cfg, val)
        if hashlib.sha256(repr(transform).encode()).hexdigest() != prepared["transform"]["repr_sha256"]:
            raise ValueError("H1 export transform identity differs from prepare")
        probe.validate_decoded_transform(probe._decode_evidence(loader),
                                         prepared["transform"]["decoded_val"])
        common = __import__("trainers.textconsistency_common", fromlist=["load_clip_checkpoint"])
        clip_model = common.load_clip_checkpoint(cfg)
        model_cls = __import__("trainers.coop_textconsistency_v2_fg",
                               fromlist=["CustomCLIP"]).CustomCLIP
        model = model_cls(cfg, classnames, clip_model)
        strict_load_current_model(model, checkpoint)
        model = model.to("cuda")
        encoder = model.multiview_encoder
        result = export_batches(model, loader, encoder.get_view_features().to("cuda"),
                                encoder.view_mask.to("cuda"), classnames, device="cuda")
        probe.validate_decoded_transform(result["decoded_val"],
                                         prepared["transform"]["decoded_val"])
        result["sample_ids"] = [str(Path(item).resolve().relative_to(data_root.resolve()))
                                for item in result["sample_ids"]]
        validate_sample_binding(prepared["samples"]["val"], [
            {"sample_id": sample_id, "label": int(label),
             "classname": classnames[int(label)]}
            for sample_id, label in zip(result["sample_ids"], result["labels"])], "export val")
        load_prepared(prepared_path)
        return write_export_artifact(output, result, prepared_path, prepared,
                                     checkpoint_record, existing_empty_ok=True)
    except Exception as exc:
        _atomic_json(output / "failure.json", {
            "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="authenticate inputs and prepare a CPU-only H1 receipt")
    for name in ("source-root", "data-root", "historical-export", "historical-config", "output-dir"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    prepare.add_argument("--dataset", choices=DATASETS, required=True)
    prepare.add_argument("--mode", choices=MODES, required=True)
    export = commands.add_parser("export", help="validate epoch20 and export current-model logits on CUDA")
    export.add_argument("--prepared", type=Path, required=True)
    export.add_argument("--model-dir", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = prepare_case(args) if args.command == "prepare" else export_case(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
