#!/usr/bin/env python3
"""Plan and execute only the fixed four-train H1 validation study."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import torch

from scripts.repro import h1_analysis
from scripts.repro import h1_validation
from scripts.repro.h1_validation import build_train_command, load_export, load_prepared


ROOT = Path(__file__).resolve().parents[2]
CASE_SPECS = (
    ("anchor-dtd", "dtd", "full_kl", "anchor"),
    ("candidate-ucf101", "ucf101", "weighted_nckd", "candidate"),
    ("candidate-dtd", "dtd", "weighted_nckd", "candidate"),
    ("candidate-fgvc_aircraft", "fgvc_aircraft", "weighted_nckd", "candidate"),
)
PLAN_KEYS = {
    "schema_version", "state", "study", "plan_path", "output_root", "python",
    "training_budget", "protocol", "cases", "stages", "operations",
    "plan_identity_sha256",
}


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def plan_digest(plan: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(plan))
    payload.pop("plan_identity_sha256", None)
    return _canonical_digest(payload)


def _is_overlap(left: Path, right: Path) -> bool:
    a, b = Path(left).resolve(), Path(right).resolve()
    return a == b or a in b.parents or b in a.parents


def _require_fresh_disjoint(target: Path, protected: Sequence[Path]) -> Path:
    resolved = Path(target).resolve()
    if resolved.exists():
        raise FileExistsError(f"Output must be a fresh path: {resolved}")
    for path in protected:
        if _is_overlap(resolved, Path(path)):
            raise ValueError(f"Output must be disjoint from immutable input: {Path(path).resolve()}")
    return resolved


def _require_disjoint(target: Path, protected: Sequence[Path]) -> Path:
    resolved = Path(target).resolve()
    for path in protected:
        if _is_overlap(resolved, Path(path)):
            raise ValueError(f"Output must be disjoint from immutable input: {Path(path).resolve()}")
    return resolved


def _prepared_protected(path: Path, prepared: Mapping[str, Any]) -> list[Path]:
    result = [Path(path).resolve(), Path(path).resolve().parent]
    for key in ("source_root", "data_root", "future_config_path"):
        if prepared.get(key):
            result.append(Path(prepared[key]).resolve())
    for section, keys in (
            ("historical", ("export", "npz")), ("historical_config", ("path",)),
            ("assets", ("clip", "template")), ("inventory", ("path",)),
            ("reference", ("path",)), ("fewshot_cache", ("path",))):
        value = prepared.get(section, {})
        if isinstance(value, Mapping):
            for key in keys:
                if value.get(key):
                    result.append(Path(value[key]).resolve())
    for value in prepared.get("source_metadata", []):
        if isinstance(value, Mapping) and value.get("path"):
            result.append(Path(value["path"]).resolve())
    return result


def _operation_files(prepared: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    source_roots = {str(Path(value["source_root"]).resolve()) for value in prepared}
    if len(source_roots) != 1:
        raise ValueError("All H1 cases must use the same authenticated source root")
    if source_roots != {str(ROOT.resolve())}:
        raise ValueError("Prepared receipts do not bind the current source checkout")
    files = {
        "train": Path(next(iter(source_roots))) / "train.py",
        "h1_validation": Path(h1_validation.__file__).resolve(),
        "h1_analysis": Path(h1_analysis.__file__).resolve(),
        "h1_study": Path(__file__).resolve(),
    }
    if any(not path.is_file() for path in files.values()):
        raise FileNotFoundError("A fixed H1 operation script is missing")
    return {name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in files.items()}


def _case_record(case_id: str, role: str, path: Path,
                 prepared: Mapping[str, Any]) -> dict[str, Any]:
    historical = prepared.get("historical", {})
    historical_export = Path(historical["export"]).resolve()
    historical_npz = Path(historical["npz"]).resolve()
    historical_export_sha = _sha256(historical_export)
    historical_npz_sha = _sha256(historical_npz)
    if historical_export_sha != historical.get("export_sha256") or \
            historical_npz_sha != historical.get("npz_sha256"):
        raise ValueError(f"Authenticated historical input changed for {case_id}")
    return {
        "case_id": case_id,
        "role": role,
        "dataset_key": prepared["dataset_key"],
        "seed": prepared["seed"],
        "mode": prepared["mode"],
        "prepared": str(Path(path).resolve()),
        "prepared_sha256": _sha256(Path(path)),
        "prepared_identity_sha256": prepared["prepared_identity_sha256"],
        "future_config": str(Path(prepared["future_config_path"]).resolve()),
        "future_config_sha256": _sha256(Path(prepared["future_config_path"])),
        "historical_export": str(historical_export),
        "historical_export_sha256": historical_export_sha,
        "historical_npz": str(historical_npz),
        "historical_npz_sha256": historical_npz_sha,
        "source_sha256": prepared["source_sha256"],
        "loss_identity": copy.deepcopy(prepared["loss_identity"]),
    }


def _stage(stage_id: str, case_id: str, kind: str, argv: Sequence[str],
           output_dir: Path, result_path: Path) -> dict[str, Any]:
    command = [str(value) for value in argv]
    return {
        "stage_id": stage_id,
        "case_id": case_id,
        "kind": kind,
        "argv": command,
        "argv_sha256": _canonical_digest(command),
        "output_dir": str(Path(output_dir).resolve()),
        "result_path": str(Path(result_path).resolve()),
    }


def _assemble_plan(prepared_paths: Sequence[Path], output_root: Path,
                   plan_path: Path, python: str, *, saved_plan: bool = False) -> dict[str, Any]:
    if len(prepared_paths) != 4:
        raise ValueError("H1 plan requires exactly four prepared receipts")
    resolved = [Path(path).resolve() for path in prepared_paths]
    if len(set(resolved)) != 4:
        raise ValueError("Each prepared case must occur once; duplicate receipts are forbidden")
    for index, left in enumerate(resolved):
        for right in resolved[index + 1:]:
            if _is_overlap(left.parent, right.parent):
                raise ValueError("Prepared artifact directories may not alias or overlap")
    prepared = [load_prepared(path) for path in resolved]
    for index, ((case_id, dataset, mode, role), receipt) in enumerate(zip(CASE_SPECS, prepared)):
        actual = (receipt.get("dataset_key"), receipt.get("mode"))
        if receipt.get("seed") != 1:
            raise ValueError(f"{case_id} must use fixed seed1")
        if actual != (dataset, mode):
            raise ValueError(f"Prepared arguments have wrong order or mode at position {index + 1}")
    if prepared[0]["historical"] != prepared[2]["historical"]:
        raise ValueError("DTD anchor and candidate must share the identical historical reference")
    protected = []
    for path, receipt in zip(resolved, prepared):
        protected.extend(_prepared_protected(path, receipt))
    root = _require_fresh_disjoint(output_root, protected)
    plan_directory = (_require_disjoint(plan_path.parent, [*protected, root]) if saved_plan
                      else _require_fresh_disjoint(plan_path.parent, [*protected, root]))
    operations = _operation_files(prepared)
    cases = [_case_record(spec[0], spec[3], path, receipt)
             for spec, path, receipt in zip(CASE_SPECS, resolved, prepared)]
    stages = []
    candidate_analyses = []
    for index, (case, receipt) in enumerate(zip(cases, prepared)):
        case_root = root / "cases" / case["case_id"]
        model_dir, export_dir, analysis_dir = (
            case_root / "model", case_root / "export", case_root / "analysis")
        train = build_train_command(receipt, model_dir, python=python)
        stages.append(_stage(f"{len(stages) + 1:02d}-train", case["case_id"], "train",
                             train, model_dir, model_dir / "textconsistency/model.pth.tar-20"))
        export = [python, operations["h1_validation"]["path"], "export",
                  "--prepared", case["prepared"], "--model-dir", str(model_dir),
                  "--output-dir", str(export_dir)]
        stages.append(_stage(f"{len(stages) + 1:02d}-export", case["case_id"], "export",
                             export, export_dir, export_dir / "export.json"))
        command = "anchor" if case["role"] == "anchor" else "candidate"
        analyze = [python, operations["h1_analysis"]["path"], command,
                   "--export", str(export_dir / "export.json"),
                   "--historical", case["historical_export"],
                   "--output-dir", str(analysis_dir)]
        kind = "anchor_audit" if command == "anchor" else "candidate_analysis"
        stages.append(_stage(f"{len(stages) + 1:02d}-{kind}", case["case_id"], kind,
                             analyze, analysis_dir, analysis_dir / "analysis.json"))
        if command == "candidate":
            candidate_analyses.append(str(analysis_dir / "analysis.json"))
    aggregate_dir = root / "aggregate"
    aggregate = [python, operations["h1_analysis"]["path"], "aggregate",
                 "--analysis", *candidate_analyses, "--output-dir", str(aggregate_dir)]
    stages.append(_stage(f"{len(stages) + 1:02d}-aggregate", "all-candidates", "aggregate",
                         aggregate, aggregate_dir, aggregate_dir / "aggregate.json"))
    plan = {
        "schema_version": 1,
        "state": "planned_not_executed",
        "study": "H1 fixed seed1 Base-validation loss comparison",
        "plan_path": str(Path(plan_path).resolve()),
        "output_root": str(root),
        "python": str(python),
        "training_budget": {"maximum": 4, "planned": 4},
        "protocol": copy.deepcopy(h1_validation.PROTOCOL),
        "cases": cases,
        "stages": stages,
        "operations": operations,
    }
    plan["plan_identity_sha256"] = plan_digest(plan)
    return plan


def create_plan(prepared_paths: Sequence[Path], output_root: Path, output_dir: Path,
                *, python: str = sys.executable) -> dict[str, Any]:
    plan_dir = Path(output_dir).resolve()
    plan = _assemble_plan(prepared_paths, output_root, plan_dir / "plan.json", str(python))
    plan_dir.mkdir(parents=True)
    _atomic_json(plan_dir / "plan.json", plan)
    return plan


def _validate_plan_for_run(path: Path) -> dict[str, Any]:
    plan_path = Path(path).resolve()
    plan = json.loads(plan_path.read_text())
    if set(plan) != PLAN_KEYS or plan.get("schema_version") != 1 or \
            plan.get("state") != "planned_not_executed":
        raise ValueError("H1 plan schema/state has missing or arbitrary fields")
    if plan.get("plan_path") != str(plan_path) or plan.get("plan_identity_sha256") != plan_digest(plan):
        raise ValueError("H1 plan identity changed")
    if plan.get("training_budget") != {"maximum": 4, "planned": 4} or \
            plan.get("protocol") != h1_validation.PROTOCOL:
        raise ValueError("H1 plan budget/protocol changed")
    paths = [Path(case["prepared"]) for case in plan.get("cases", [])]
    rebuilt = _assemble_plan(paths, Path(plan["output_root"]), plan_path, plan["python"],
                             saved_plan=True)
    if rebuilt != plan:
        raise ValueError("H1 plan commands, paths, inputs, or operation hashes changed")
    return plan


def _input_snapshot(plan: Mapping[str, Any]) -> dict[str, Any]:
    cases = []
    for expected in plan.get("cases", []):
        prepared_path = Path(expected["prepared"])
        prepared = load_prepared(prepared_path)
        record = _case_record(expected["case_id"], expected["role"], prepared_path, prepared)
        if record != expected:
            raise ValueError(f"Prepared input changed for {expected['case_id']}")
        cases.append(record)
    operations = {}
    for name, expected in plan.get("operations", {}).items():
        path = Path(expected["path"])
        actual = {"path": str(path.resolve()), "sha256": _sha256(path)}
        if actual != expected:
            raise ValueError(f"Operation script changed: {name}")
        operations[name] = actual
    plan_file = None
    if plan.get("plan_path"):
        path = Path(plan["plan_path"])
        saved = json.loads(path.read_text())
        if saved != plan or saved.get("plan_identity_sha256") != plan_digest(saved):
            raise ValueError("Immutable command plan changed during execution")
        plan_file = {"path": str(path.resolve()), "sha256": _sha256(path)}
    payload = {"cases": cases, "operations": operations, "plan_file": plan_file}
    return {**payload, "digest": _canonical_digest(payload)}


def _directory_manifest(path: Path) -> dict[str, Any]:
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"Stage output directory is missing: {root}")
    files = []
    for item in sorted(value for value in root.rglob("*") if value.is_file()):
        files.append({"path": str(item.relative_to(root)), "sha256": _sha256(item),
                      "size": item.stat().st_size})
    if not files:
        raise ValueError(f"Stage output directory is empty: {root}")
    return {"files": files, "tree_sha256": _canonical_digest(files)}


def _validate_stage_output(stage: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    case = next((value for value in plan["cases"]
                 if value["case_id"] == stage["case_id"]), None)
    kind, output = stage["kind"], Path(stage["output_dir"])
    if kind == "train":
        prepared = load_prepared(Path(case["prepared"]))
        checkpoint, _ = h1_validation.validate_new_checkpoint(prepared, output)
        return {"checkpoint": checkpoint, "output": _directory_manifest(output)}
    if kind == "export":
        receipt, _ = load_export(Path(stage["result_path"]))
        if receipt.get("dataset_key") != case["dataset_key"] or receipt.get("mode") != case["mode"]:
            raise ValueError("Export identity differs from the planned case")
        return {"export_sha256": _sha256(Path(stage["result_path"])),
                "output": _directory_manifest(output)}
    payload = json.loads(Path(stage["result_path"]).read_text())
    if payload.get("status") != "passed":
        raise ValueError(f"{kind} did not pass")
    if kind in ("anchor_audit", "candidate_analysis"):
        expected_kind = "anchor_audit" if kind == "anchor_audit" else "candidate"
        if payload.get("analysis_kind") != expected_kind or \
                payload.get("dataset_key") != case["dataset_key"]:
            raise ValueError("Analysis identity differs from the planned case")
    elif kind == "aggregate":
        if payload.get("analysis_kind") != "aggregate" or payload.get("case_count") != 3:
            raise ValueError("Aggregate is incomplete or mixed")
    return {"result_sha256": _sha256(Path(stage["result_path"])),
            "output": _directory_manifest(output)}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def execute_plan(
        plan_path: Path, *, cuda_available: Callable[[], bool] = torch.cuda.is_available,
        plan_validator: Callable[[Path], Mapping[str, Any]] = _validate_plan_for_run,
        stage_validator: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any] | None]
            = _validate_stage_output,
) -> dict[str, Any]:
    plan = dict(plan_validator(Path(plan_path)))
    root = Path(plan["output_root"]).resolve()
    if root.exists():
        raise FileExistsError(f"Study root must be fresh; retry/resume is forbidden: {root}")
    lock = root.with_name(root.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"Study has an existing exclusive lock: {lock}") from exc
    os.close(descriptor)
    root.mkdir(parents=True)
    logs = root / "logs"; logs.mkdir()
    status: dict[str, Any] = {
        "schema_version": 1, "state": "running", "plan": str(Path(plan_path).resolve()),
        "plan_identity_sha256": plan.get("plan_identity_sha256"), "started_at": _now(),
        "commands": [],
    }
    _atomic_json(root / "status.json", status)
    try:
        if not cuda_available():
            raise RuntimeError("CUDA is required for H1 study execution; no command was spawned")
        for stage in plan["stages"]:
            if Path(stage["output_dir"]).exists():
                raise FileExistsError(f"Planned stage output already exists: {stage['output_dir']}")
            before = _input_snapshot(plan)
            started = _now()
            command_log = {
                "stage_id": stage["stage_id"], "case_id": stage["case_id"],
                "kind": stage["kind"], "argv": stage["argv"],
                "argv_sha256": _canonical_digest(stage["argv"]), "started_at": started,
                "input_snapshot_before": before["digest"],
            }
            try:
                completed = subprocess.run(stage["argv"], text=True, capture_output=True,
                                           check=False)
            except Exception as exc:
                command_log.update({
                    "finished_at": _now(), "returncode": None, "stdout": "", "stderr": "",
                    "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                })
                _atomic_json(logs / f"{stage['stage_id']}.json", command_log)
                status["commands"].append({"stage_id": stage["stage_id"], "status": "failed",
                                            "log": str(logs / f"{stage['stage_id']}.json")})
                status["state"] = "failed"
                status["error"] = command_log["error"]
                status["finished_at"] = _now()
                _atomic_json(root / "status.json", status)
                raise
            command_log.update({"finished_at": _now(), "returncode": completed.returncode,
                                "stdout": completed.stdout, "stderr": completed.stderr})
            try:
                after = _input_snapshot(plan)
                command_log["input_snapshot_after"] = after["digest"]
                command_log["inputs_immutable"] = before == after
                if before != after:
                    raise ValueError("Authenticated inputs changed during stage execution")
                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Stage {stage['stage_id']} {stage['case_id']} {stage['kind']} failed "
                        f"with exit {completed.returncode}")
                evidence = stage_validator(stage, plan)
                command_log["evidence"] = evidence
                command_log["status"] = "passed"
            except Exception as exc:
                command_log["status"] = "failed"
                command_log["error"] = f"{type(exc).__name__}: {exc}"
                _atomic_json(logs / f"{stage['stage_id']}.json", command_log)
                status["commands"].append({"stage_id": stage["stage_id"], "status": "failed",
                                            "log": str(logs / f"{stage['stage_id']}.json")})
                status["state"] = ("stopped" if stage["kind"] in
                                   ("anchor_audit", "aggregate") else "failed")
                status["error"] = command_log["error"]
                status["finished_at"] = _now()
                _atomic_json(root / "status.json", status)
                raise
            _atomic_json(logs / f"{stage['stage_id']}.json", command_log)
            status["commands"].append({"stage_id": stage["stage_id"], "status": "passed",
                                        "log": str(logs / f"{stage['stage_id']}.json")})
            _atomic_json(root / "status.json", status)
        status["state"] = "passed"
        status["finished_at"] = _now()
        _atomic_json(root / "status.json", status)
        return status
    except Exception as exc:
        if status["state"] == "running":
            status["state"] = "failed"
            status["error"] = f"{type(exc).__name__}: {exc}"
            status["finished_at"] = _now()
            _atomic_json(root / "status.json", status)
        raise
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="CPU-only validation and immutable planning")
    plan.add_argument("--prepared", type=Path, nargs=4, required=True)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    run = commands.add_parser("run", help="future CUDA-only one-shot execution")
    run.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        result = create_plan(args.prepared, args.output_root, args.output_dir)
    else:
        result = execute_plan(args.plan)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
