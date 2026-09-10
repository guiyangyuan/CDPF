#!/usr/bin/env python3
"""Preregister, prepare, execute, or audit the fixed DTD/Aircraft study."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DATASETS = ("dtd", "fgvc_aircraft")
VARIANTS = ("baseline", "hierarchical_adapter", "flat_dual_adapter")
CLIP_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
MODEL_SHA256 = "9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba"
FIXED = {"n_ctx": 4, "n_views": 20, "n_groups": 4, "kd_temp": 2.0,
         "adaptive_ce_weight": 1.0, "diversity_weight": 0.0,
         "contrastive_weight": 0.0, "ensemble_alpha": 0.5}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_module(source_root, relative, name):
    root = str(Path(source_root).resolve())
    for entry in (str(Path(root) / "Dassl.pytorch"), root):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    if relative.startswith("trainers/"):
        module_name = relative[:-3].replace("/", ".")
        return importlib.import_module(module_name)
    return load_file(Path(root) / relative, name)


def model_source_digest(source_root):
    root = Path(source_root).resolve(); h = hashlib.sha256(); paths = [root / "train.py"]
    for folder in ("trainers", "datasets", "clip", "Dassl.pytorch/dassl"):
        paths.extend(sorted((root / folder).rglob("*.py")))
    for path in paths:
        h.update(str(path.relative_to(root)).encode()); h.update(path.read_bytes())
    return h.hexdigest()


def dependency_hashes(source_root):
    root = Path(source_root).resolve()
    paths = [root / "train.py", root / "scripts/repro/run_pilot.py", root / "scripts/repro/preflight.py",
             root / "scripts/repro/summarize.py", root / "trainers/textconsistency_runtime.py",
             root / "trainers/textconsistency_common.py"]
    paths += [root / f"configs/repro/{d}.yaml" for d in DATASETS]
    paths += [root / f"configs/datasets/{d}.yaml" for d in DATASETS]
    paths += [root / "templates/texture_prompts_full.json", root / "templates/airplane_prompts_full.json"]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Required source file is missing: {missing[0]}")
    return {str(p.relative_to(root)): sha256(p) for p in paths}


def build_plan(source_root, data_root, output_root, python=sys.executable):
    source_root, data_root, output_root = map(lambda p: Path(p).resolve(),
                                             (source_root, data_root, output_root))
    runner = source_module(source_root, "scripts/repro/run_pilot.py", "cross_run_pilot")
    stages = []
    for kind, seeds, epochs in (("smoke", [1], 1), ("formal", [1, 2, 3], 20)):
        for dataset in DATASETS:
            commands = runner.build_commands(data_root, output_root / kind / dataset, seeds, epochs,
                                             VARIANTS, python=python, dataset=dataset, kd_weight=1.0)
            stages.append({"name": f"{kind}_{dataset}", "kind": kind, "dataset": dataset,
                           "seeds": seeds, "epochs": epochs, "commands": commands})
    return {"purpose": "fixed_cross_dataset_study", "source_root": str(source_root),
            "data_root": str(data_root), "output_root": str(output_root), "datasets": list(DATASETS),
            "variants": list(VARIANTS), "primary_branch": {"baseline": "student", "full": "ensemble"},
            "shots": 16, "kd_weight": 1.0, "fixed_settings": FIXED,
            "expected_model_sha256": MODEL_SHA256, "observed_model_sha256": model_source_digest(source_root),
            "expected_clip_sha256": CLIP_SHA256, "files_sha256": dependency_hashes(source_root),
            "stages": stages}


def dataset_evidence(source_root, data_root, key):
    module = source_module(source_root, "scripts/repro/preflight.py", f"cross_preflight_{key}")
    checked = module.check(data_root, key)
    spec = module.DATASETS[key]
    per_seed = []
    for seed in (1, 2, 3):
        cfg = module.get_cfg_default(); module.extend_cfg(cfg)
        cfg.SEED = seed; cfg.DATASET.ROOT = str(Path(data_root).resolve()); cfg.DATASET.NUM_SHOTS = 16
        cfg.DATASET.NAME = spec["name"]
        splits = {}
        for subsample in ("base", "new"):
            cfg.DATASET.SUBSAMPLE_CLASSES = subsample; module.set_random_seed(seed)
            loaded = spec["loader"](cfg)
            splits[subsample] = {"classes": list(loaded.classnames),
                                 "train_count": len(loaded.train_x), "val_count": len(loaded.val),
                                 "test_count": len(loaded.test),
                                 "train_sha256": module.sample_digest(loaded.train_x, data_root),
                                 "val_sha256": module.sample_digest(loaded.val, data_root),
                                 "test_sha256": module.sample_digest(loaded.test, data_root)}
        per_seed.append({"seed": seed, "splits": splits})
    checked["loader_evidence"] = per_seed
    return checked


def prepare(plan, preparation_dir, clip_path=None):
    target = Path(preparation_dir).resolve()
    if target.exists():
        raise FileExistsError(f"Choose a fresh preparation directory: {target}")
    source = Path(plan["source_root"])
    actual_model = model_source_digest(source)
    if actual_model != MODEL_SHA256:
        raise ValueError(f"Model source digest mismatch: {actual_model}")
    clip = Path(clip_path or Path.home() / ".cache/clip/ViT-B-16.pt").resolve()
    if not clip.is_file() or sha256(clip) != CLIP_SHA256:
        raise ValueError(f"Missing or incorrect complete CLIP ViT-B/16 checkpoint: {clip}")
    evidence = {"protocol": plan, "model_source_sha256": actual_model,
                "files_sha256": dependency_hashes(source), "clip": {"path": str(clip), "sha256": sha256(clip)},
                "environment": cpu_environment(),
                "datasets": {key: dataset_evidence(source, plan["data_root"], key) for key in DATASETS}}
    target.mkdir(parents=True)
    atomic_json(target / "preparation.json", evidence)
    atomic_json(target / "protocol.json", plan)
    return evidence


def cpu_environment():
    import platform
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "cpu_threads_required": 4}


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2)); temporary.replace(path)


def execute(plan, preparation_dir):
    output = Path(plan["output_root"])
    if output.exists():
        raise FileExistsError(f"Choose a fresh output root: {output}")
    prepared = json.loads((Path(preparation_dir) / "preparation.json").read_text())
    if prepared["protocol"] != plan:
        raise ValueError("Preparation protocol differs from requested execution")
    if dependency_hashes(plan["source_root"]) != prepared["files_sha256"]:
        raise ValueError("Prepared source files changed before execution")
    if sha256(prepared["clip"]["path"]) != prepared["clip"]["sha256"]:
        raise ValueError("Prepared CLIP checkpoint changed before execution")
    current_data = {key: dataset_evidence(plan["source_root"], plan["data_root"], key) for key in DATASETS}
    if current_data != prepared["datasets"]:
        raise ValueError("Dataset preflight/loader evidence changed before execution")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA with at least one visible GPU is required for execution")
    if any(torch.cuda.mem_get_info(i)[0] < 0.9 * torch.cuda.mem_get_info(i)[1]
           for i in range(torch.cuda.device_count())):
        raise RuntimeError("Every visible GPU must be idle (at least 90% memory free) before execution")
    output.mkdir(parents=True)
    atomic_json(output / "protocol.json", plan)
    status = {"state": "running", "stage": None, "pid": os.getpid(), "commands": [], "completed": []}
    atomic_json(output / "status.json", status)
    env = os.environ.copy(); env.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                                         "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4",
                                         "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    env.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
    source = Path(plan["source_root"]); env["PYTHONPATH"] = os.pathsep.join(
        [str(source), str(source / "Dassl.pytorch"), env.get("PYTHONPATH", "")])
    audit = Path(__file__).with_name("cross_dataset_audit.py")
    for stage in plan["stages"]:
        status["stage"] = stage["name"]; log_dir = output / "logs" / stage["name"]; log_dir.mkdir(parents=True)
        for index, command in enumerate(stage["commands"]):
            record = {"stage": stage["name"], "index": index, "command": command, "started": time.time(), "exit_code": None}
            status["commands"].append(record); atomic_json(output / "status.json", status)
            with (log_dir / f"{index:03d}.stdout.log").open("w") as out, (log_dir / f"{index:03d}.stderr.log").open("w") as err:
                result = subprocess.run(command, cwd=source, env=env, stdout=out, stderr=err)
            record.update(ended=time.time(), exit_code=result.returncode); atomic_json(output / "status.json", status)
            if result.returncode:
                status["state"] = "failed"; atomic_json(output / "status.json", status)
                raise subprocess.CalledProcessError(result.returncode, command)
        audit_command = [sys.executable, str(audit), "--source-root", str(source), "--study-root", str(output),
                         "--preparation-dir", str(Path(preparation_dir).resolve()), "--stage", stage["name"]]
        audit_record = {"stage": stage["name"], "index": "audit", "command": audit_command,
                        "started": time.time(), "exit_code": None}
        status["commands"].append(audit_record); atomic_json(output / "status.json", status)
        audit_result = subprocess.run(audit_command, env=env)
        audit_record.update(ended=time.time(), exit_code=audit_result.returncode)
        if audit_result.returncode:
            status["state"] = "failed"; atomic_json(output / "status.json", status)
            raise subprocess.CalledProcessError(audit_result.returncode, audit_command)
        status["completed"].append(stage["name"]); atomic_json(output / "status.json", status)
    final_command = [sys.executable, str(audit), "--source-root", str(source), "--study-root", str(output),
                     "--preparation-dir", str(Path(preparation_dir).resolve()), "--final"]
    final_record = {"stage": "final_audit", "index": "audit", "command": final_command,
                    "started": time.time(), "exit_code": None}
    status["stage"] = "final_audit"; status["commands"].append(final_record); atomic_json(output / "status.json", status)
    final_result = subprocess.run(final_command, env=env)
    final_record.update(ended=time.time(), exit_code=final_result.returncode)
    if final_result.returncode:
        status["state"] = "failed"; atomic_json(output / "status.json", status)
        raise subprocess.CalledProcessError(final_result.returncode, final_command)
    status["completed"].append("final_audit")
    status.update(state="passed", stage=None); atomic_json(output / "status.json", status)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path); parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path); parser.add_argument("--preparation-dir", type=Path)
    parser.add_argument("--clip-path", type=Path); parser.add_argument("--python", default=sys.executable)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true"); mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true"); mode.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv); plan = build_plan(args.source_root, args.data_root, args.output_root, args.python)
    if args.plan_only:
        print(json.dumps(plan, indent=2)); return
    if not args.preparation_dir:
        parser.error("--preparation-dir is required outside plan mode")
    if args.prepare_only:
        prepare(plan, args.preparation_dir, args.clip_path); return
    if args.execute:
        execute(plan, args.preparation_dir); return
    audit = Path(__file__).with_name("cross_dataset_audit.py")
    subprocess.run([sys.executable, str(audit), "--source-root", str(args.source_root.resolve()),
                    "--study-root", str(args.output_root.resolve()), "--preparation-dir",
                    str(args.preparation_dir.resolve()), "--final"], check=True)


if __name__ == "__main__":
    main()
