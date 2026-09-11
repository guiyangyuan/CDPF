#!/usr/bin/env python3
"""Prepare and export the frozen validation-only crossed-inference probe.

``prepare`` is CPU-only and reconstructs the historical validation loader without
calling dataset constructors. ``export`` requires CUDA and constructs the frozen
source model classes directly; it never constructs a trainer or optimizer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import platform
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

sys.dont_write_bytecode = True

import numpy as np
import torch
from torch.nn import functional as F


DATASETS = {
    "ucf101": ("UCF101", "ucf101", "ucf_prompts_full.json"),
    "dtd": ("DescribableTextures", "dtd", "texture_prompts_full.json"),
    "fgvc_aircraft": ("FGVCAircraft", "fgvc_aircraft", "airplane_prompts_full.json"),
}
OLD_SOURCE_SHA256 = "d74a9770d0d814c56269486e2492327881633d98314281d0f13a584ee279f17f"
NEW_SOURCE_SHA256 = "9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba"
EXPECTED_SOURCES = {"ucf101": OLD_SOURCE_SHA256, "dtd": NEW_SOURCE_SHA256,
                    "fgvc_aircraft": NEW_SOURCE_SHA256}
CLIP_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
FIXED_INVENTORY_SHA256 = "858bfbf1f64d71e93c5ca89f706bd757ca69b1f6250b2cadb55a521dcda4d185"
BRANCHES = ("C", "S0", "S1", "M", "T0", "T1")
ROLES = ("coop", "kd0", "kd1")
PROTOCOL = {
    "version": 1, "split": "val", "subsample_classes": "base", "shots": 16,
    "val_shots": 4, "epoch": 20, "alpha": 0.5, "sampler": "SequentialSampler",
    "workers": 0, "student_roles": {"C": "coop", "S0": "kd0", "S1": "kd1"},
    "teacher_roles": {"T0": "kd0", "T1": "kd1"},
    "description_mean": "normalize(each valid descriptor), masked mean, normalize(mean)",
    "selection": "all fixed branches and combinations; no fitting or search",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def model_source_digest(source_root: Path) -> str:
    root = Path(source_root).resolve()
    files = [root / "train.py"]
    for folder in ("trainers", "datasets", "clip", "Dassl.pytorch/dassl"):
        files.extend(sorted((root / folder).rglob("*.py")))
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("Frozen source tree is incomplete")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Required JSON is missing: {path}")
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _fresh_output(path: Path, protected: Iterable[Path] = ()) -> Path:
    target = Path(path).resolve()
    for item in protected:
        other = Path(item).resolve()
        if target == other or target in other.parents or other in target.parents:
            raise ValueError(f"Output must be disjoint from immutable input: {other}")
    if target.exists():
        raise FileExistsError(f"Output must be a fresh path: {target}")
    return target


def validate_inventory(inventory: Mapping[str, Any], dataset: str, seed: int) -> dict[str, Any]:
    if dataset not in DATASETS or seed not in (1, 2, 3):
        raise ValueError("Only UCF101/DTD/FGVCAircraft seeds 1/2/3 are allowed")
    cases = [case for case in inventory.get("cases", [])
             if case.get("dataset") == dataset and case.get("seed") == seed]
    if len(cases) != 1:
        raise ValueError("Inventory case is missing or duplicated")
    case = copy.deepcopy(cases[0])
    if case.get("split") != "val" or case.get("candidate_classes") != "base":
        raise ValueError("Inventory must request validation split and Base classes")
    required_case = {"dataset", "seed", "split", "candidate_classes", "train_samples_sha256",
                     "val_samples_sha256", "clip_sha256", "template_sha256",
                     "coop_training_template_sha256", "checkpoints"}
    if not required_case.issubset(case):
        raise ValueError("Inventory case schema is incomplete")
    if set(case["checkpoints"]) != set(ROLES):
        raise ValueError("Inventory role schema must contain exactly coop/kd0/kd1")
    required_role = {"workspace_relative_checkpoint", "sha256", "manifest_sha256",
                     "model_source_sha256", "effective_kd_weight", "epoch"}
    for role, expected_kd in (("coop", 0.0), ("kd0", 0.0), ("kd1", 1.0)):
        record = case["checkpoints"][role]
        if set(record) != required_role:
            raise ValueError(f"Inventory {role} schema has missing or extra fields")
        if record["epoch"] != 20 or record["effective_kd_weight"] != expected_kd:
            raise ValueError(f"Inventory {role} has the wrong epoch or effective loss")
        if record["model_source_sha256"] != EXPECTED_SOURCES[dataset]:
            raise ValueError(f"Inventory {role} has the wrong frozen source")
        for field in ("sha256", "manifest_sha256", "model_source_sha256"):
            if not isinstance(record[field], str) or len(record[field]) != 64:
                raise ValueError(f"Inventory {role} has invalid {field}")
    if case["clip_sha256"] != CLIP_SHA256:
        raise ValueError("Inventory CLIP hash differs from the fixed asset")
    if case["coop_training_template_sha256"] is not None:
        raise ValueError("CoOp training template must remain null")
    return case


def sample_digest(items: Iterable[Any], data_root: Path) -> str:
    root = Path(data_root).resolve()
    rows = []
    for item in items or []:
        path = Path(item.impath).resolve()
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Dataset image is outside data root: {path}") from exc
        rows.append((str(relative), int(item.label), item.classname))
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()


def _sample_rows(items: Iterable[Any], root: Path, classnames: list[str]) -> list[dict[str, Any]]:
    base = Path(root).resolve()
    rows, seen = [], set()
    for item in items or []:
        path = Path(item.impath)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset image is missing: {path}")
        identity = path.resolve()
        try:
            relative = identity.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"Dataset image is outside data root: {identity}") from exc
        label = int(item.label)
        if label < 0 or label >= len(classnames) or item.classname != classnames[label]:
            raise ValueError("Sample label/class order differs from rebuilt classes")
        if identity in seen:
            raise ValueError("Duplicate resolved image identity within a split")
        seen.add(identity)
        stat = identity.stat()
        rows.append({"sample_id": str(relative), "label": label, "classname": item.classname,
                     "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return rows


def validate_samples(train: list[Any], val: list[Any], test: list[Any], data_root: Path,
                     classnames: list[str]) -> dict[str, Any]:
    if not val:
        raise ValueError("Validation split is missing or empty; test fallback is forbidden")
    if not train or not test:
        raise ValueError("Official train/test metadata is missing or empty")
    if not classnames or len(set(classnames)) != len(classnames):
        raise ValueError("Class order is empty or duplicated")
    result = {name: _sample_rows(items, data_root, classnames)
              for name, items in (("train", train), ("val", val), ("test", test))}
    identities = {name: {row["sample_id"] for row in rows} for name, rows in result.items()}
    if identities["train"] & identities["val"]:
        raise ValueError("Resolved train/validation image identities overlap")
    result["classes"] = list(classnames)
    result["inventory_sha256"] = json_digest(result)
    return result


def _expected_model_signature(role: str, case: Mapping[str, Any]) -> dict[str, Any]:
    is_coop = role == "coop"
    signature = {
        "format": 1 if case["dataset"] == "ucf101" else 2,
        "trainer": "CoOp_TextConsistency_Ablation" if is_coop else "CoOp_TextConsistency_V2_FG",
        "n_ctx": 4,
        "aggregation": "none" if is_coop else "hierarchical",
        "feature_adapter": not is_coop,
        "n_groups": None if is_coop else 4,
        "n_views": 20,
        "ablation_group": "baseline" if is_coop else None,
        "clip_sha256": case["clip_sha256"],
        "template_sha256": None if is_coop else case["template_sha256"],
    }
    if signature["format"] == 2:
        signature.update({"view_order": "contiguous", "order_seed": 0, "order_version": 1})
    return signature


def _assert_manifest(role: str, manifest: Mapping[str, Any], case: Mapping[str, Any]) -> None:
    expected_loss = {"coop": 0.0, "kd0": 0.0, "kd1": 1.0}[role]
    fixed = {"dataset": DATASETS[case["dataset"]][0], "seed": case["seed"], "shots": 16,
             "subsample": "base", "code_sha256": EXPECTED_SOURCES[case["dataset"]],
             "train_samples_sha256": case["train_samples_sha256"],
             "val_samples_sha256": case["val_samples_sha256"]}
    for key, value in fixed.items():
        if manifest.get(key) != value:
            raise ValueError(f"{role} manifest mismatch: {key}")
    losses = manifest.get("effective_losses")
    if losses != {"kd": expected_loss, "adaptive_ce": 1.0, "diversity": 0.0, "contrastive": 0.0}:
        raise ValueError(f"{role} manifest has the wrong effective losses")
    signature = manifest.get("model_signature", {})
    expected_signature = _expected_model_signature(role, case)
    if signature != expected_signature:
        differing = sorted(key for key in set(signature) | set(expected_signature)
                           if signature.get(key) != expected_signature.get(key))
        raise ValueError(f"{role} model signature mismatch: {differing}")
    if role == "coop":
        expected_parameters = 2048
    else:
        expected_parameters = 790018
    if manifest.get("trainable_parameters") != expected_parameters:
        raise ValueError(f"{role} trainable parameter count mismatch")


def _checkpoint_state(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint is not a mapping")
    return state


def validate_checkpoint_bundles(case: Mapping[str, Any], workspace_root: Path) -> dict[str, Any]:
    root, receipts, teacher_states = Path(workspace_root).resolve(), {}, {}
    for role in ROLES:
        expected = case["checkpoints"][role]
        checkpoint = (root / expected["workspace_relative_checkpoint"]).resolve()
        try:
            checkpoint.relative_to(root)
        except ValueError as exc:
            raise ValueError("Checkpoint escapes --workspace-root") from exc
        if sha256_file(checkpoint) != expected["sha256"]:
            raise ValueError(f"{role} checkpoint hash mismatch")
        sidecar = checkpoint.with_name(checkpoint.name + ".sha256")
        if sidecar.read_text().strip() != expected["sha256"]:
            raise ValueError(f"{role} checkpoint sidecar mismatch")
        model_dir, manifest_path = checkpoint.parent.parent, checkpoint.parent.parent / "run_manifest.json"
        if sha256_file(manifest_path) != expected["manifest_sha256"]:
            raise ValueError(f"{role} manifest hash mismatch")
        manifest = _read_json(manifest_path); _assert_manifest(role, manifest, case)
        state = _checkpoint_state(checkpoint)
        if state.get("epoch") != 20 or state.get("repro_signature") != manifest["model_signature"]:
            raise ValueError(f"{role} checkpoint epoch/signature mismatch")
        provenance = state.get("run_provenance", {})
        for field in ("dataset", "seed", "shots", "subsample", "code_sha256", "effective_losses",
                      "train_samples_sha256", "val_samples_sha256", "test_samples_sha256",
                      "trainable_parameters", "model_signature"):
            if provenance.get(field) != manifest.get(field):
                raise ValueError(f"{role} checkpoint provenance mismatch: {field}")
        tensors = state.get("state_dict")
        if not isinstance(tensors, dict) or not tensors or any(
                not torch.is_tensor(value) or not torch.isfinite(value).all().item() for value in tensors.values()):
            raise ValueError(f"{role} checkpoint has missing or non-finite tensors")
        trainable = {key: value for key, value in tensors.items()
                     if not key.endswith(("token_prefix", "token_suffix"))}
        expected_count = 2048 if role == "coop" else 790018
        if sum(value.numel() for value in trainable.values()) != expected_count:
            raise ValueError(f"{role} checkpoint omits role parameters")
        if role == "coop" and set(trainable) != {"prompt_learner.ctx"}:
            raise ValueError("CoOp checkpoint has an unexpected parameter schema")
        if role != "coop":
            required = {"prompt_learner.ctx", "view_aggregator.intra_group_temp",
                        "view_aggregator.inter_group_temp"}
            if not set(trainable).issuperset(required):
                raise ValueError(f"{role} checkpoint omits role parameters")
            teacher_states[role] = {key: value for key, value in trainable.items()
                                    if key.startswith("view_aggregator.")}
        config_path = model_dir / "effective_config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"{role} effective config is missing")
        receipts[role] = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": expected["sha256"],
            "sidecar": str(sidecar), "sidecar_sha256": sha256_file(sidecar),
            "manifest": str(manifest_path), "manifest_sha256": expected["manifest_sha256"],
            "effective_config": str(config_path), "effective_config_sha256": sha256_file(config_path),
            "manifest_payload": manifest,
        }
    left, right = teacher_states["kd0"], teacher_states["kd1"]
    receipts["teacher_tensor_exact_equal"] = set(left) == set(right) and all(
        torch.equal(left[key], right[key]) for key in left)
    receipts["teacher_tensor_keys"] = sorted(set(left) | set(right))
    return receipts


def _source_modules(source_root: Path):
    root = Path(source_root).resolve()
    for name, module in list(sys.modules.items()):
        if name in ("train", "clip", "datasets", "trainers", "dassl") or \
                name.startswith(("clip.", "datasets.", "trainers.", "dassl.")):
            filename = getattr(module, "__file__", None)
            if filename and root not in Path(filename).resolve().parents:
                raise RuntimeError(f"Frozen source module contamination: {name} from {filename}")
    sys.path[:0] = [str(root), str(root / "Dassl.pytorch")]
    training = importlib.import_module("train")
    if Path(training.__file__).resolve() != root / "train.py":
        raise RuntimeError(f"Imported train.py from the wrong source: {training.__file__}")
    return training


def _config_args(config_path: Path, data_root: Path, output_dir: Path, seed: int,
                 source_root: Path, cuda: bool):
    return SimpleNamespace(root=str(data_root), output_dir=str(output_dir), resume="", seed=seed,
        source_domains=None, target_domains=None, transforms=None, trainer="", backbone="", head="",
        dataset_config_file="", config_file=str(config_path),
        opts=["TEST.SPLIT", "val", "DATALOADER.NUM_WORKERS", "0",
              "DATALOADER.TEST.SAMPLER", "SequentialSampler", "USE_CUDA", str(cuda),
              "TRAINER.TEXTCONSISTENCY.DEVICE", "cuda" if cuda else "cpu",
              "TRAINER.TEXTCONSISTENCY.CLIP_PATH", str(source_root / "clip/ViT-B-16.pt")])


def build_role_config(training: Any, config_path: Path, data_root: Path, output_dir: Path,
                      seed: int, source_root: Path, *, cuda: bool):
    cfg = training.setup_cfg(_config_args(config_path, data_root, output_dir, seed, source_root, cuda))
    if cfg.TEST.SPLIT != "val" or cfg.DATASET.SUBSAMPLE_CLASSES != "base" or cfg.DATASET.NUM_SHOTS != 16:
        raise ValueError("Source setup_cfg changed the frozen validation/Base protocol")
    if cfg.DATALOADER.NUM_WORKERS != 0 or cfg.DATALOADER.TEST.SAMPLER != "SequentialSampler":
        raise ValueError("Source setup_cfg changed the deterministic loader protocol")
    return cfg


def validate_effective_role_config(role: str, cfg: Any, manifest: Mapping[str, Any],
                                   case: Mapping[str, Any], source_root: Path,
                                   *, cuda: bool) -> dict[str, Any]:
    """Bind every inference/validation-bearing config value to trusted evidence."""
    if role not in ROLES:
        raise ValueError(f"Unknown role effective config: {role}")
    signature = manifest.get("model_signature")
    if signature != _expected_model_signature(role, case):
        raise ValueError(f"{role} effective config is paired with an unauthenticated signature")
    is_coop = role == "coop"
    tc = cfg.TRAINER.TEXTCONSISTENCY
    expected_tc = {
        "ABLATION_GROUP": "baseline" if is_coop else "full",
        "ADAPTIVE_CE_WEIGHT": 1.0,
        "AGGREGATION": "mean" if is_coop else signature["aggregation"],
        "CONSISTENCY_TEMP": 1.0,
        "CONSISTENCY_WEIGHT": 1.0,
        "CONTRASTIVE_WEIGHT": 0.0,
        "DEVICE": "cuda" if cuda else "cpu",
        "DIVERSITY_WEIGHT": 0.0,
        "ENSEMBLE_ALPHA": PROTOCOL["alpha"],
        "INFERENCE_BRANCH": "student" if is_coop else "ensemble",
        "KD_TEMP": 2.0,
        "KD_WEIGHT": (1.0 if is_coop and case["dataset"] == "ucf101" else
                      {"coop": 0.0, "kd0": 0.0, "kd1": 1.0}[role]),
        "N_CTX": signature["n_ctx"],
        "N_GROUPS": 4,
        "N_VIEWS": signature["n_views"],
        "PREC": "fp32",
        "TEXT_BATCH_SIZE": 32,
        "USE_FEATURE_ADAPTER": signature["feature_adapter"],
    }
    if signature["format"] == 2:
        expected_tc.update({"VIEW_ORDER": signature["view_order"],
                            "ORDER_SEED": signature["order_seed"]})
    actual_tc = {key: getattr(tc, key, None) for key in expected_tc}
    if actual_tc != expected_tc:
        differing = sorted(key for key in expected_tc if actual_tc[key] != expected_tc[key])
        raise ValueError(f"{role} effective config mismatch: {differing}")
    expected_losses = manifest["effective_losses"]
    config_losses = {
        "kd": 0.0 if is_coop else float(tc.KD_WEIGHT),
        "adaptive_ce": float(tc.ADAPTIVE_CE_WEIGHT),
        "diversity": float(tc.DIVERSITY_WEIGHT),
        "contrastive": float(tc.CONTRASTIVE_WEIGHT),
    }
    if config_losses != expected_losses:
        raise ValueError(f"{role} effective config loss behavior differs from the manifest")
    fixed = {
        "seed": (int(cfg.SEED), int(case["seed"])),
        "cuda": (bool(cfg.USE_CUDA), cuda),
        "split": (cfg.TEST.SPLIT, PROTOCOL["split"]),
        "dataset": (cfg.DATASET.NAME, DATASETS[case["dataset"]][0]),
        "shots": (int(cfg.DATASET.NUM_SHOTS), PROTOCOL["shots"]),
        "subsample": (cfg.DATASET.SUBSAMPLE_CLASSES, PROTOCOL["subsample_classes"]),
        "workers": (int(cfg.DATALOADER.NUM_WORKERS), PROTOCOL["workers"]),
        "sampler": (cfg.DATALOADER.TEST.SAMPLER, PROTOCOL["sampler"]),
        "batch_size": (int(cfg.DATALOADER.TEST.BATCH_SIZE), 16),
        "backbone": (cfg.MODEL.BACKBONE.NAME, "ViT-B/16"),
        "input_size": (list(cfg.INPUT.SIZE), [224, 224]),
        "interpolation": (cfg.INPUT.INTERPOLATION, "bicubic"),
        "pixel_mean": (list(cfg.INPUT.PIXEL_MEAN), [0.48145466, 0.4578275, 0.40821073]),
        "pixel_std": (list(cfg.INPUT.PIXEL_STD), [0.26862954, 0.26130258, 0.27577711]),
        "no_transform": (bool(cfg.INPUT.NO_TRANSFORM), False),
        "trainer": (cfg.TRAINER.NAME, signature["trainer"]),
    }
    differing = sorted(name for name, (actual, expected) in fixed.items() if actual != expected)
    clip_path = Path(tc.CLIP_PATH).expanduser().resolve()
    expected_clip = (Path(source_root).resolve() / "clip/ViT-B-16.pt").resolve()
    if clip_path != expected_clip:
        differing.append("clip_path")
    if differing:
        raise ValueError(f"{role} effective config protocol mismatch: {differing}")
    return {"model_signature": copy.deepcopy(signature), "textconsistency": actual_tc,
            "losses": config_losses, "protocol": {name: actual for name, (actual, _) in fixed.items()},
            "clip_path": str(clip_path)}


def rebuild_source_dataset(training: Any, cfg: Any, dataset_key: str):
    """Use source readers/samplers directly, never their cache-writing constructors."""
    random.seed(int(cfg.SEED))
    root = Path(cfg.DATASET.ROOT).resolve()
    pets = importlib.import_module("datasets.oxford_pets").OxfordPets
    base_cls = importlib.import_module("dassl.data.datasets").DatasetBase
    _, folder, _ = DATASETS[dataset_key]; dataset_dir = root / folder
    if dataset_key in ("ucf101", "dtd"):
        filename = "split_zhou_UCF101.json" if dataset_key == "ucf101" else "split_zhou_DescribableTextures.json"
        split_path = dataset_dir / filename
        if not split_path.is_file():
            raise FileNotFoundError("Fixed official split metadata is missing; generation is forbidden")
        image_dir = dataset_dir / ("UCF-101-midframes" if dataset_key == "ucf101" else "images")
        train, val, test = pets.read_split(str(split_path), str(image_dir)); metadata_paths = [split_path]
    else:
        aircraft = importlib.import_module("datasets.fgvc_aircraft").FGVCAircraft
        instance = object.__new__(aircraft)
        instance.dataset_dir, instance.image_dir = str(dataset_dir), str(dataset_dir / "images")
        variants = dataset_dir / "variants.txt"
        if not variants.is_file():
            raise FileNotFoundError("Aircraft variants metadata is missing")
        classnames = variants.read_text().splitlines(); cname2lab = {name: i for i, name in enumerate(classnames)}
        files = ["images_variant_train.txt", "images_variant_val.txt", "images_variant_test.txt"]
        train, val, test = [aircraft.read_data(instance, cname2lab, name) for name in files]
        metadata_paths = [variants, *(dataset_dir / name for name in files)]
    sampler = object.__new__(base_cls)
    train = base_cls.generate_fewshot_dataset(sampler, train, num_shots=16)
    val = base_cls.generate_fewshot_dataset(sampler, val, num_shots=4)
    train, val, test = pets.subsample_classes(train, val, test, subsample="base")
    classnames = list(base_cls.get_lab2cname(train)[1])
    if not classnames:
        raise ValueError("Rebuilt Base class order is empty")
    return train, val, test, classnames, metadata_paths


def build_val_loader(cfg: Any, val: list[Any]):
    transforms = importlib.import_module("dassl.data.transforms")
    data = importlib.import_module("dassl.data.data_manager")
    transform = transforms.build_transform(cfg, is_train=False)
    wrapper = data.DatasetWrapper(cfg, val, transform=transform, is_train=False)
    loader = torch.utils.data.DataLoader(wrapper, batch_size=int(cfg.DATALOADER.TEST.BATCH_SIZE),
        sampler=torch.utils.data.SequentialSampler(wrapper), num_workers=0, drop_last=False, pin_memory=False)
    return loader, transform


def _decode_evidence(loader: Any) -> dict[str, Any]:
    digest, count, shape, dtype = hashlib.sha256(), 0, None, None
    for batch in loader:
        batch_count, batch_shape, batch_dtype = _update_decode_digest(digest, batch)
        count += batch_count; shape, dtype = batch_shape, batch_dtype
    if count == 0: raise ValueError("Validation loader decoded zero images")
    return {"count": count, "tensor_shape": shape, "tensor_dtype": dtype,
            "ordered_transform_sha256": digest.hexdigest()}


def _update_decode_digest(digest: Any, batch: Mapping[str, Any]) -> tuple[int, list[int], str]:
    images = batch["img"].detach().cpu().contiguous()
    labels = batch["label"].detach().cpu().contiguous()
    if not torch.is_tensor(images) or not torch.is_tensor(labels) or images.ndim < 2 or labels.ndim != 1:
        raise ValueError("Decoded validation batch tensors are malformed")
    if len(images) != len(labels) or len(batch["impath"]) != len(labels):
        raise ValueError("Decoded validation batch paths/labels/tensors are not aligned")
    if not torch.isfinite(images).all().item():
        raise ValueError("Decoded validation transform produced non-finite pixels")
    digest.update(images.numpy().tobytes()); digest.update(labels.numpy().tobytes())
    for path in batch["impath"]: digest.update(str(path).encode())
    return int(labels.numel()), list(images.shape[1:]), str(images.dtype)


def validate_decoded_transform(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    required = {"count", "tensor_shape", "tensor_dtype", "ordered_transform_sha256"}
    if set(actual) != required or set(expected) != required or dict(actual) != dict(expected):
        raise ValueError("Validation transform digest differs from prepared evidence")


def _metadata_receipt(paths: Iterable[Path]) -> list[dict[str, str]]:
    return [{"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path))} for path in paths]


def prepare_case(args: argparse.Namespace) -> dict[str, Any]:
    inventory_path, workspace_root = Path(args.inventory).resolve(), Path(args.workspace_root).resolve()
    source_root, data_root = Path(args.source_root).resolve(), Path(args.data_root).resolve()
    if sha256_file(inventory_path) != FIXED_INVENTORY_SHA256:
        raise ValueError("Probe inventory differs from the authenticated fixed inventory")
    case = validate_inventory(_read_json(inventory_path), args.dataset, args.seed)
    checkpoint_inputs = [(workspace_root / case["checkpoints"][role]["workspace_relative_checkpoint"]).resolve()
                         for role in ROLES]
    output = _fresh_output(args.output_dir, (inventory_path, source_root, data_root, *checkpoint_inputs))
    source_sha = model_source_digest(source_root)
    if source_sha != EXPECTED_SOURCES[args.dataset]: raise ValueError("Frozen source tree SHA-256 differs")
    clip_path, template_path = source_root / "clip/ViT-B-16.pt", source_root / "templates" / DATASETS[args.dataset][2]
    if sha256_file(clip_path) != case["clip_sha256"] or sha256_file(template_path) != case["template_sha256"]:
        raise ValueError("Frozen CLIP/template asset hash mismatch")
    bundles = validate_checkpoint_bundles(case, workspace_root)
    training = _source_modules(source_root)
    configs = {role: build_role_config(training, Path(bundles[role]["effective_config"]), data_root,
                                       output, args.seed, source_root, cuda=False) for role in ROLES}
    config_bindings = {role: validate_effective_role_config(
        role, configs[role], bundles[role]["manifest_payload"], case, source_root, cuda=False)
        for role in ROLES}
    cfg = configs["coop"]
    train, val, test, classnames, metadata_paths = rebuild_source_dataset(training, cfg, args.dataset)
    samples = validate_samples(train, val, test, data_root, classnames)
    for name, items, expected in (("train", train, case["train_samples_sha256"]),
                                  ("val", val, case["val_samples_sha256"])):
        if sample_digest(items, data_root) != expected: raise ValueError(f"Rebuilt {name} sample hash differs")
    test_sha = sample_digest(test, data_root)
    if any(test_sha != bundles[role]["manifest_payload"]["test_samples_sha256"] for role in ROLES):
        raise ValueError("Rebuilt test metadata hash differs from role manifest")
    loader, transform = build_val_loader(cfg, val); decoded = _decode_evidence(loader)
    import yaml
    effective_configs = {role: yaml.safe_load(configs[role].dump()) for role in ROLES}
    compatible_configs = copy.deepcopy(effective_configs)
    for role in ROLES:
        compatible_configs[role]["SEED"] = "<seed>"
        compatible_configs[role]["OUTPUT_DIR"] = "<output>"
    receipt = {"status": "prepared", "dataset_key": args.dataset, "dataset": DATASETS[args.dataset][0],
        "seed": args.seed, "protocol": PROTOCOL,
        "inventory": {"path": str(inventory_path), "sha256": sha256_file(inventory_path), "case": case},
        "workspace_root": str(workspace_root), "source_root": str(source_root), "data_root": str(data_root),
        "source_sha256": source_sha,
        "assets": {"clip": str(clip_path.resolve()), "clip_sha256": sha256_file(clip_path),
                   "template": str(template_path.resolve()), "template_sha256": sha256_file(template_path)},
        "checkpoints": bundles, "teacher_tensor_exact_equal": bundles["teacher_tensor_exact_equal"],
        "samples": samples, "sample_hashes": {"train": sample_digest(train, data_root),
        "val": sample_digest(val, data_root), "test": test_sha},
        "source_metadata": _metadata_receipt(metadata_paths),
        "effective_config": effective_configs["coop"], "effective_configs": effective_configs,
        "effective_config_bindings": config_bindings,
        "effective_config_sha256": json_digest(effective_configs),
        "config_compatibility_sha256": json_digest(compatible_configs),
        "config_compatibility": {"coop_declared_kd_weight": float(cfg.TRAINER.TEXTCONSISTENCY.KD_WEIGHT),
            "coop_effective_kd_weight": 0.0,
            "legacy_exception": "ucf101 baseline ablation disables declared KD=1" if
                args.dataset == "ucf101" and cfg.TRAINER.TEXTCONSISTENCY.KD_WEIGHT == 1.0 else None},
        "tool": {"validation_probe_sha256": sha256_file(Path(__file__).resolve())},
        "transform": {"repr": repr(transform), "repr_sha256": hashlib.sha256(repr(transform).encode()).hexdigest(),
                      "decoded_val": decoded},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
        "limitations": ["CUDA model construction and forward parity are not exercised by prepare"]}
    output.mkdir(parents=True); _atomic_json(output / "prepared.json", receipt)
    return receipt


def strict_load_trainable_modules(model: torch.nn.Module, checkpoint: Mapping[str, Any]) -> None:
    modules = torch.nn.ModuleDict({"prompt_learner": model.prompt_learner,
                                   "view_aggregator": model.view_aggregator})
    expected, saved = modules.state_dict(), checkpoint.get("state_dict")
    if not isinstance(saved, Mapping): raise ValueError("Checkpoint state_dict is missing")
    missing, unexpected = set(expected) - set(saved), set(saved) - set(expected)
    if missing or unexpected:
        raise ValueError(f"Strict checkpoint mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    for key, value in expected.items():
        actual = saved[key]
        if not torch.is_tensor(actual) or actual.shape != value.shape:
            raise ValueError(f"Checkpoint tensor shape mismatch: {key}")
        if not torch.isfinite(actual).all().item(): raise ValueError(f"Checkpoint tensor is non-finite: {key}")
        if key.endswith(("token_prefix", "token_suffix")) and not torch.equal(actual.cpu(), value.cpu()):
            raise ValueError(f"Checkpoint class-dependent buffer differs: {key}")
    modules.load_state_dict(dict(saved), strict=True)


def _freeze(model: torch.nn.Module) -> torch.nn.Module:
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model


def instantiate_frozen_models(classes: Mapping[str, Any], configs: Mapping[str, Any],
                              classnames: list[str], clip_model: Any,
                              checkpoints: Mapping[str, Mapping[str, Any]], device: Any):
    """Construct every model before moving their shared CLIP modules to a device."""
    if set(classes) != set(ROLES) or set(configs) != set(ROLES) or set(checkpoints) != set(ROLES):
        raise ValueError("Model construction requires complete coop/kd0/kd1 inputs")
    models = {role: classes[role](configs[role], classnames, clip_model) for role in ROLES}
    for role in ROLES:
        strict_load_trainable_modules(models[role], checkpoints[role])
    for role in ROLES:
        models[role] = _freeze(models[role].to(device))
    return models


def export_batches(models: Mapping[str, torch.nn.Module], batches: Iterable[Mapping[str, Any]],
                   description_features: torch.Tensor, view_mask: torch.Tensor,
                   classnames: list[str], *, device: Any | None = None) -> dict[str, Any]:
    if set(models) != set(ROLES): raise ValueError("Export requires exactly coop/kd0/kd1 models")
    for model in models.values(): _freeze(model)
    descriptions = F.normalize(description_features.float(), dim=-1)
    mask = view_mask.to(device=descriptions.device, dtype=torch.bool)
    if descriptions.ndim != 3 or mask.shape != descriptions.shape[:2] or not mask.any(dim=1).all().item():
        raise ValueError("Description features/mask have invalid shapes or empty classes")
    weights = mask.to(descriptions.dtype) / mask.sum(dim=1, keepdim=True)
    mean_features = F.normalize(torch.einsum("cv,cvd->cd", weights, descriptions), dim=-1)
    collected, labels, sample_ids = {name: [] for name in BRANCHES}, [], []
    decoded_digest, decoded_count, decoded_shape, decoded_dtype = hashlib.sha256(), 0, None, None
    teacher_equal, precision = True, set()
    if device is None:
        device = next(models["coop"].parameters()).device
    with torch.inference_mode():
        for batch in batches:
            batch_count, decoded_shape, decoded_dtype = _update_decode_digest(decoded_digest, batch)
            decoded_count += batch_count
            images = batch["img"].to(device)
            outputs = {role: model(images, return_attn=False) for role, model in models.items()}
            coop_s, _, coop_f, _ = outputs["coop"]; s0, t0, f0, _ = outputs["kd0"]; s1, t1, f1, _ = outputs["kd1"]
            if t0 is None or t1 is None: raise ValueError("KD source direct forward omitted a teacher branch")
            if not torch.equal(coop_f, f0) or not torch.equal(f0, f1):
                raise ValueError("Frozen image features differ across role forwards")
            scales = [models[role].logit_scale.exp().detach() for role in ROLES]
            if not all(torch.equal(scales[0], scale) for scale in scales[1:]):
                raise ValueError("Historical CLIP logit scales differ across roles")
            mean_logits = scales[0].float() * F.normalize(coop_f.float(), dim=-1) @ mean_features.t()
            values = {"C": coop_s, "S0": s0, "S1": s1, "M": mean_logits, "T0": t0, "T1": t1}
            expected_shape = tuple(coop_s.shape)
            if expected_shape[1] != len(classnames) or any(tuple(v.shape) != expected_shape for v in values.values()):
                raise ValueError("Branch logits shape/class order mismatch")
            if any(not torch.isfinite(v).all().item() for v in values.values()):
                raise ValueError("Export branch contains non-finite logits")
            teacher_equal = teacher_equal and torch.equal(t0, t1)
            for name, value in values.items():
                precision.add(str(value.dtype).replace("torch.", "")); collected[name].append(value.float().cpu().numpy())
            batch_labels = batch["label"].cpu().numpy().astype(np.int64, copy=False)
            paths = [str(path) for path in batch["impath"]]
            if len(paths) != len(batch_labels): raise ValueError("Batch sample IDs and labels differ in length")
            labels.append(batch_labels); sample_ids.extend(paths)
    if not labels: raise ValueError("Validation loader produced no batches")
    label_array = np.concatenate(labels)
    if np.any(label_array < 0) or np.any(label_array >= len(classnames)):
        raise ValueError("Validation labels are outside the class order")
    return {"logits": {name: np.concatenate(values).astype(np.float32, copy=False)
                       for name, values in collected.items()}, "labels": label_array,
            "sample_ids": sample_ids, "classnames": list(classnames),
            "decoded_val": {"count": decoded_count, "tensor_shape": decoded_shape,
                "tensor_dtype": decoded_dtype, "ordered_transform_sha256": decoded_digest.hexdigest()},
            "teacher_logits_exact_equal": teacher_equal,
            "forward_precision": sorted(precision) if len(precision) != 1 else next(iter(precision))}


def write_export_artifact(output_dir: Path, payload: Mapping[str, Any], metadata: Mapping[str, Any],
                          *, existing_empty_ok: bool = False) -> dict[str, Any]:
    target = Path(output_dir).resolve()
    if target.exists() and (not existing_empty_ok or any(target.iterdir())):
        raise FileExistsError(f"Output must be a fresh path: {target}")
    target.mkdir(parents=True, exist_ok=True)
    logits = payload["logits"]
    if set(logits) != set(BRANCHES): raise ValueError("Export artifact requires all six raw branches")
    labels = np.asarray(payload["labels"]); arrays = {name: np.asarray(logits[name], dtype=np.float32) for name in BRANCHES}
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Export labels must be one-dimensional integers")
    n, c = len(labels), len(payload["classnames"])
    if n == 0 or c < 2 or any(v.shape != (n, c) or not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Export logits must be finite [N,C] arrays")
    if len(payload["sample_ids"]) != n or len(set(payload["sample_ids"])) != n:
        raise ValueError("Export sample IDs must be unique and aligned")
    npz_path = target / "logits.npz"; np.savez(npz_path, labels=labels.astype(np.int64), **arrays)
    rows = [{"row": i, "sample_id": str(sample), "label": int(labels[i]),
             "classname": payload["classnames"][int(labels[i])]}
            for i, sample in enumerate(payload["sample_ids"])]
    receipt = {**copy.deepcopy(dict(metadata)), "status": "passed", "schema_version": 1,
        "branches": list(BRANCHES), "npz": str(npz_path), "npz_sha256": sha256_file(npz_path),
        "rows": rows, "class_order": list(payload["classnames"]),
        "teacher_tensor_exact_equal": bool(metadata.get("teacher_tensor_exact_equal",
                                                        payload.get("teacher_exact_equal", False))),
        "teacher_logits_exact_equal": bool(payload.get("teacher_logits_exact_equal",
                                                        payload.get("teacher_exact_equal", False))),
        "forward_precision": payload["forward_precision"], "storage_precision": "float32"}
    _atomic_json(target / "export.json", receipt); return receipt


def _revalidate_prepared(prepared: Mapping[str, Any]) -> None:
    if prepared.get("status") != "prepared" or prepared.get("protocol") != PROTOCOL:
        raise ValueError("Prepared receipt status/protocol mismatch")
    inventory_record = prepared.get("inventory", {})
    if inventory_record.get("sha256") != FIXED_INVENTORY_SHA256:
        raise ValueError("Prepared receipt is not bound to the authenticated inventory")
    inventory = _read_json(Path(inventory_record["path"]))
    case = validate_inventory(inventory, prepared["dataset_key"], prepared["seed"])
    if case != inventory_record.get("case"):
        raise ValueError("Prepared inventory case differs from the authenticated inventory")
    if prepared.get("source_sha256") != EXPECTED_SOURCES[prepared["dataset_key"]] or \
            prepared.get("assets", {}).get("clip_sha256") != case["clip_sha256"] or \
            prepared.get("assets", {}).get("template_sha256") != case["template_sha256"]:
        raise ValueError("Prepared source/assets differ from the authenticated inventory")
    if prepared.get("sample_hashes", {}).get("train") != case["train_samples_sha256"] or \
            prepared.get("sample_hashes", {}).get("val") != case["val_samples_sha256"]:
        raise ValueError("Prepared sample hashes differ from the authenticated inventory")
    for role in ROLES:
        if prepared["checkpoints"][role].get("checkpoint_sha256") != case["checkpoints"][role]["sha256"] or \
                prepared["checkpoints"][role].get("manifest_sha256") != case["checkpoints"][role]["manifest_sha256"]:
            raise ValueError(f"Prepared {role} bundle differs from the authenticated inventory")
    checks = [(prepared["inventory"]["path"], prepared["inventory"]["sha256"]),
              (prepared["assets"]["clip"], prepared["assets"]["clip_sha256"]),
              (prepared["assets"]["template"], prepared["assets"]["template_sha256"])]
    for role in ROLES:
        row = prepared["checkpoints"][role]
        checks += [(row["checkpoint"], row["checkpoint_sha256"]), (row["sidecar"], row["sidecar_sha256"]),
                   (row["manifest"], row["manifest_sha256"]),
                   (row["effective_config"], row["effective_config_sha256"])]
    checks += [(row["path"], row["sha256"]) for row in prepared["source_metadata"]]
    for path, expected in checks:
        if sha256_file(Path(path)) != expected: raise ValueError(f"Prepared input hash changed: {path}")
    if model_source_digest(Path(prepared["source_root"])) != prepared["source_sha256"]:
        raise ValueError("Prepared frozen source changed")


def export_case(args: argparse.Namespace) -> dict[str, Any]:
    prepared_path = Path(args.prepared).resolve(); prepared = _read_json(prepared_path)
    protected = [prepared_path, Path(prepared["source_root"]), Path(prepared["data_root"]),
                 *(Path(prepared["checkpoints"][role]["checkpoint"]) for role in ROLES)]
    output = _fresh_output(args.output_dir, protected)
    output.mkdir(parents=True)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for export; CPU fallback is forbidden")
        _revalidate_prepared(prepared)
        source_root, data_root = Path(prepared["source_root"]), Path(prepared["data_root"])
        training = _source_modules(source_root)
        configs = {role: build_role_config(training, Path(prepared["checkpoints"][role]["effective_config"]),
                   data_root, output, prepared["seed"], source_root, cuda=True) for role in ROLES}
        for role in ROLES:
            validate_effective_role_config(role, configs[role],
                prepared["checkpoints"][role]["manifest_payload"], prepared["inventory"]["case"],
                source_root, cuda=True)
        train, val, test, classnames, _ = rebuild_source_dataset(training, configs["coop"], prepared["dataset_key"])
        if validate_samples(train, val, test, data_root, classnames) != prepared["samples"]:
            raise ValueError("Rebuilt source samples/classes differ from prepared receipt")
        loader, transform = build_val_loader(configs["coop"], val)
        if hashlib.sha256(repr(transform).encode()).hexdigest() != prepared["transform"]["repr_sha256"]:
            raise ValueError("Validation transform identity differs from prepare")
        validate_decoded_transform(_decode_evidence(loader), prepared["transform"]["decoded_val"])
        common = importlib.import_module("trainers.textconsistency_common")
        clip_model = common.load_clip_checkpoint(configs["coop"])
        baseline_cls = importlib.import_module("trainers.coop_textconsistency_ablation").CustomCLIP
        full_cls = importlib.import_module("trainers.coop_textconsistency_v2_fg").CustomCLIP
        classes = {"coop": baseline_cls, "kd0": full_cls, "kd1": full_cls}
        checkpoints = {role: _checkpoint_state(Path(prepared["checkpoints"][role]["checkpoint"]))
                       for role in ROLES}
        models = instantiate_frozen_models(classes, configs, classnames, clip_model, checkpoints, "cuda")
        encoder = models["kd1"].multiview_encoder
        result = export_batches(models, loader, encoder.get_view_features().to("cuda"),
                                encoder.view_mask.to("cuda"), classnames, device="cuda")
        validate_decoded_transform(result["decoded_val"], prepared["transform"]["decoded_val"])
        root = data_root.resolve(); result["sample_ids"] = [str(Path(p).resolve().relative_to(root)) for p in result["sample_ids"]]
        if result["sample_ids"] != [row["sample_id"] for row in prepared["samples"]["val"]] or \
                result["labels"].tolist() != [row["label"] for row in prepared["samples"]["val"]]:
            raise ValueError("Export row order differs from prepared validation order")
        if validate_samples(train, val, test, data_root, classnames) != prepared["samples"]:
            raise ValueError("Validation samples changed during export")
        post_loader, post_transform = build_val_loader(configs["coop"], val)
        if hashlib.sha256(repr(post_transform).encode()).hexdigest() != prepared["transform"]["repr_sha256"]:
            raise ValueError("Validation transform identity changed during export")
        validate_decoded_transform(_decode_evidence(post_loader), prepared["transform"]["decoded_val"])
        _revalidate_prepared(prepared)
        metadata = {"dataset_key": prepared["dataset_key"], "dataset": prepared["dataset"], "seed": prepared["seed"],
            "protocol": PROTOCOL, "prepared": str(prepared_path), "prepared_sha256": sha256_file(prepared_path),
            "teacher_tensor_exact_equal": prepared["teacher_tensor_exact_equal"],
            "tool": {"validation_probe_sha256": sha256_file(Path(__file__).resolve())},
            "effective_config": prepared["effective_config"],
            "transform": prepared["transform"], "sample_hashes": prepared["sample_hashes"],
            "evaluated_transform": result["decoded_val"],
            "sample_inventory_sha256": prepared["samples"]["inventory_sha256"],
            "provenance": {"source_sha256": prepared["source_sha256"],
                "clip_sha256": prepared["assets"]["clip_sha256"], "template_sha256": prepared["assets"]["template_sha256"],
                "inventory_sha256": prepared["inventory"]["sha256"],
                "inventory_case_sha256": json_digest(prepared["inventory"]["case"]),
                "train_samples_sha256": prepared["sample_hashes"]["train"],
                "val_samples_sha256": prepared["sample_hashes"]["val"],
                "config_sha256": prepared["effective_config_sha256"],
                "config_compatibility_sha256": prepared["config_compatibility_sha256"],
                "checkpoint_sha256": {role: prepared["checkpoints"][role]["checkpoint_sha256"] for role in ROLES},
                "manifest_sha256": {role: prepared["checkpoints"][role]["manifest_sha256"] for role in ROLES}},
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "device": "cuda"}}
        return write_export_artifact(output, result, metadata, existing_empty_ok=True)
    except Exception as exc:
        _atomic_json(output / "failure.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}); raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="CPU-only input and real validation-loader preparation")
    for name in ("inventory", "workspace-root", "source-root", "data-root", "output-dir"):
        prepare.add_argument(f"--{name}", type=Path, required=True)
    prepare.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    prepare.add_argument("--seed", type=int, choices=(1, 2, 3), required=True)
    export = commands.add_parser("export", help="CUDA-only frozen logits export")
    export.add_argument("--prepared", type=Path, required=True); export.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv); result = prepare_case(args) if args.command == "prepare" else export_case(args)
    print(json.dumps(result, indent=2, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
