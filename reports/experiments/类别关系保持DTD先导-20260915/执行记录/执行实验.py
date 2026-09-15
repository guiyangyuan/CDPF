#!/usr/bin/env python3
"""Execute the frozen DTD relation-preserving pilot without retries."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


PROTOCOL_PATH = Path("/root/autodl-tmp/textconsistency/类别关系保持实验协议.json")
STATE_DIR = Path("/root/autodl-tmp/textconsistency/实验状态/类别关系保持-DTD-20260915")
STATE_PATH = STATE_DIR / "运行状态.json"
RUNNER_COPY = STATE_DIR / "执行实验.py"
REPO_URL = "https://github.com/guiyangyuan/textconsistency_v2_tune.git"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def run_checked(argv: list[str], *, cwd: Path | None = None,
                stdout=None, env: dict[str, str] | None = None) -> None:
    subprocess.run(argv, cwd=cwd, env=env, stdout=stdout,
                   stderr=subprocess.STDOUT if stdout is not None else None,
                   check=True)


def prepare_release(protocol: dict) -> Path:
    release = Path(protocol["服务器路径"]["release"])
    commit = protocol["源码提交"]
    if not release.exists():
        raise RuntimeError(f"pre-deployed frozen source is missing: {release}")
    if (release / ".git").is_dir():
        run_checked(["git", "checkout", "--detach", commit], cwd=release)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=release, text=True).strip()
        if head != commit:
            raise RuntimeError(f"wrong source commit: {head}")
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=release, text=True).strip():
            raise RuntimeError("frozen release checkout is dirty")
    elif not (release / "train.py").is_file():
        raise RuntimeError(f"frozen source archive is incomplete: {release}")
    return release


def verify_inputs(protocol: dict, release: Path) -> dict:
    sys.path[:0] = [str(release), str(release / "Dassl.pytorch")]
    os.chdir(release)
    import train as training
    from dassl.config import get_cfg_default
    from scripts.repro import validation_probe as probe

    actual_files = {}
    for relative, expected in protocol["文件SHA256"].items():
        actual = sha256(release / relative)
        actual_files[relative] = actual
        if actual != expected:
            raise RuntimeError(f"frozen file hash mismatch: {relative}")
    source_digest = probe.model_source_digest(release)
    if source_digest != protocol["源码摘要SHA256"]:
        raise RuntimeError("frozen source digest mismatch")

    clip = Path(protocol["服务器路径"]["clip"])
    clip_digest = sha256(clip)
    if clip_digest != protocol["固定资产"]["CLIP_SHA256"]:
        raise RuntimeError("CLIP checkpoint hash mismatch")

    cfg = get_cfg_default()
    training.extend_cfg(cfg)
    cfg.merge_from_file(str(release / "configs/datasets/dtd.yaml"))
    cfg.merge_from_file(str(release / "configs/repro/类别关系保持/匹配基线.yaml"))
    cfg.SEED = 1
    cfg.DATASET.ROOT = str(Path(protocol["服务器路径"]["data"]).resolve())
    cfg.DATASET.SUBSAMPLE_CLASSES = "base"
    cfg.freeze()
    train, val, test, classes, metadata = probe.rebuild_source_dataset(training, cfg, "dtd")
    sample_hashes = {
        name: probe.sample_digest(items, Path(cfg.DATASET.ROOT))
        for name, items in (("train", train), ("val", val), ("test", test))
    }
    if sample_hashes != protocol["固定资产"]["DTD样本摘要"]:
        raise RuntimeError(f"DTD sample digest mismatch: {sample_hashes}")
    return {
        "source_commit": protocol["源码提交"],
        "source_digest": source_digest,
        "file_sha256": actual_files,
        "clip_sha256": clip_digest,
        "sample_sha256": sample_hashes,
        "sample_counts": {"train": len(train), "val": len(val), "test": len(test)},
        "class_count": len(classes),
        "metadata": [str(path) for path in metadata],
    }


def accuracy(path: Path) -> float:
    value = json.loads(path.read_text())
    candidates = [value.get("accuracy")]
    metrics = value.get("metrics")
    if isinstance(metrics, dict):
        candidates.extend([metrics.get("accuracy"), metrics.get("top1")])
    for candidate in candidates:
        if isinstance(candidate, (int, float)):
            return float(candidate)
    raise RuntimeError(f"accuracy is missing from {path}")


def harmonic_mean(base: float, novel: float) -> float:
    return 0.0 if base + novel == 0 else 2.0 * base * novel / (base + novel)


def archive_outputs(protocol: dict, status: dict) -> tuple[Path, str]:
    output = Path(protocol["服务器路径"]["output"])
    archive_dir = Path("/root/autodl-tmp/textconsistency/backups")
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / "类别关系保持-DTD-20260915-1e9927b.tar.gz"
    temporary = archive.with_name(archive.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with tarfile.open(temporary, "w:gz") as tar:
        if output.exists():
            tar.add(output, arcname=output.name)
        tar.add(STATE_DIR, arcname=STATE_DIR.name)
        tar.add(PROTOCOL_PATH, arcname=PROTOCOL_PATH.name)
    temporary.replace(archive)
    digest = sha256(archive)
    (archive.with_suffix(archive.suffix + ".sha256")).write_text(f"{digest}  {archive.name}\n")
    status["archive"] = {"path": str(archive), "sha256": digest, "bytes": archive.stat().st_size}
    write_json(STATE_PATH, status)
    return archive, digest


def main() -> int:
    started = time.time()
    protocol = json.loads(PROTOCOL_PATH.read_text())
    output = Path(protocol["服务器路径"]["output"])
    if output.exists():
        raise RuntimeError(f"fresh output path already exists: {output}")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, RUNNER_COPY)
    status = {
        "protocol": protocol["协议"],
        "state": "preflight",
        "started_unix": started,
        "stages": [],
        "automatic_retries": 0,
    }
    write_json(STATE_PATH, status)
    release = prepare_release(protocol)
    status["preflight"] = verify_inputs(protocol, release)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(release), str(release / "Dassl.pytorch")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    tests_log = STATE_DIR / "CPU核验.log"
    tests = [
        "tests.repro.test_trainers.TrainerIntegration.test_category_relation_loss_matches_off_diagonal_geometry",
        "tests.repro.test_trainers.TrainerIntegration.test_relation_regularizer_changes_only_training_objective",
    ]
    with tests_log.open("w") as handle:
        run_checked([str(Path(protocol["服务器路径"]["release"]).parent.parent / ".venv/bin/python"),
                     "-m", "unittest", "-q", *tests], cwd=release, stdout=handle, env=env)
    status["preflight"]["focused_tests"] = {"count": 2, "result": "passed", "log": str(tests_log)}
    output.mkdir(parents=True)
    log_dir = output / "运行日志"
    log_dir.mkdir()
    write_json(STATE_PATH, status)
    print("预检通过：源码、CLIP、DTD样本和2项聚焦测试均匹配", flush=True)

    try:
        for stage in protocol["运行顺序"]:
            record = {
                "sequence": stage["序号"], "name": stage["名称"], "type": stage["类型"],
                "state": "running", "started_unix": time.time(),
            }
            status["state"] = "running"
            status["current_stage"] = stage["名称"]
            status["stages"].append(record)
            write_json(STATE_PATH, status)
            print(f"开始 {stage['序号']}/6：{stage['名称']}", flush=True)
            log_path = log_dir / f"{stage['序号']:02d}-{stage['名称']}.log"
            with log_path.open("w") as handle:
                run_checked(stage["argv"], cwd=Path(stage["cwd"]), stdout=handle, env=env)
            record.update(state="passed", ended_unix=time.time(), log=str(log_path),
                          log_sha256=sha256(log_path))
            write_json(STATE_PATH, status)
            print(f"完成 {stage['序号']}/6：{stage['名称']}", flush=True)

        base_root = output / "匹配基线"
        method_root = output / "关系保持"
        baseline_base = accuracy(base_root / "Base测试/metrics.json")
        baseline_novel = accuracy(base_root / "Novel测试/metrics.json")
        method_base = accuracy(method_root / "Base测试/metrics.json")
        method_novel = accuracy(method_root / "Novel测试/metrics.json")
        baseline_hm = harmonic_mean(baseline_base, baseline_novel)
        method_hm = harmonic_mean(method_base, method_novel)
        result = {
            "匹配基线": {"Base": baseline_base, "Novel": baseline_novel, "HM": baseline_hm},
            "关系保持": {"Base": method_base, "Novel": method_novel, "HM": method_hm},
            "差值": {"Base": method_base - baseline_base,
                    "Novel": method_novel - baseline_novel, "HM": method_hm - baseline_hm},
            "通过冻结判据": method_hm > baseline_hm and method_novel >= baseline_novel,
            "判据": "关系保持相对匹配基线的HM严格为正，且Novel准确率不下降",
        }
        write_json(output / "独立复算.json", result)
        status.update(state="completed", current_stage=None, result=result,
                      ended_unix=time.time(), elapsed_seconds=time.time() - started)
        write_json(STATE_PATH, status)
        archive, digest = archive_outputs(protocol, status)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        print(f"实验完成；归档={archive}；SHA256={digest}", flush=True)
        return 0
    except Exception as error:
        status.update(state="failed", error=f"{type(error).__name__}: {error}",
                      ended_unix=time.time(), elapsed_seconds=time.time() - started)
        write_json(STATE_PATH, status)
        try:
            archive, digest = archive_outputs(protocol, status)
            print(f"失败证据已归档；归档={archive}；SHA256={digest}", flush=True)
        finally:
            raise


if __name__ == "__main__":
    raise SystemExit(main())
