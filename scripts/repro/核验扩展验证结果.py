#!/usr/bin/env python3
"""独立复算扩展验证预测、模型与备份清单，并生成三种子审计结果。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tarfile
from pathlib import Path


方法 = ("student", "description_mean", "simple_fusion", "mean_probability_fusion")
方法中文 = {
    "student": "CoOp",
    "description_mean": "冻结均值描述",
    "simple_fusion": "固定logit均值融合",
    "mean_probability_fusion": "固定均值概率融合",
}
预期模型源码摘要 = "b90fa7789983c3ea595a837933d23ee5b4a23e59009cd0c5c42addd99e30d5ef"
预期CLIP摘要 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def harmonic_mean(base: float, novel: float) -> float:
    return 0.0 if base + novel == 0 else 2.0 * base * novel / (base + novel)


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def check(self, name: str, condition: bool, detail: object) -> None:
        self.checks.append({"检查": name, "通过": bool(condition), "详情": detail})

    def require(self, name: str, condition: bool, detail: object) -> None:
        self.check(name, condition, detail)
        if not condition:
            raise ValueError(f"{name}: {detail}")


def server_path_to_local(input_root: Path, server_path: str) -> Path:
    prefix = "/root/autodl-tmp/textconsistency/"
    if not server_path.startswith(prefix):
        raise ValueError(f"清单含未知服务器路径: {server_path}")
    return input_root / server_path[len(prefix):]


def audit_archive(
    input_root: Path,
    audit: Audit,
    archive_name: str,
    expected_sha256: str,
    expected_members: int,
) -> dict:
    archive = input_root / archive_name
    audit.require("本机备份包存在", archive.is_file(), str(archive))
    archive_sha = sha256(archive)
    audit.require("本机备份包SHA256与服务器收据一致", archive_sha == expected_sha256, archive_sha)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
    unsafe = [member.name for member in members if Path(member.name).is_absolute() or ".." in Path(member.name).parts]
    audit.require("备份包成员路径安全", not unsafe, unsafe)
    audit.require("备份包成员数量符合服务器记录", len(members) == expected_members, len(members))
    return {"路径": str(archive), "字节数": archive.stat().st_size, "SHA256": archive_sha, "成员数": len(members)}


def audit_manifest(input_root: Path, audit: Audit, manifest_relative_path: str) -> dict:
    manifest = input_root / manifest_relative_path
    audit.require("服务器文件清单存在", manifest.is_file(), str(manifest))
    raw_lines = manifest.read_bytes().splitlines(keepends=True)
    entries = []
    for index, raw in enumerate(raw_lines):
        line = raw.decode().rstrip("\r\n")
        expected, server_path = line.split("  ", 1)
        local_path = server_path_to_local(input_root, server_path)
        entries.append((index, expected, server_path, local_path))

    missing = [server for _, _, server, local in entries if not local.is_file()]
    audit.require("服务器清单文件均已下载", not missing, missing)
    mismatches = []
    self_entry = None
    manifest_server_path = "/root/autodl-tmp/textconsistency/" + manifest_relative_path
    for index, expected, server_path, local_path in entries:
        if server_path == manifest_server_path:
            self_entry = (index, expected)
            continue
        actual = sha256(local_path)
        if actual != expected:
            mismatches.append({"路径": server_path, "预期": expected, "实际": actual})
    audit.require("服务器实验文件逐项SHA256一致", not mismatches, mismatches)
    audit.require("清单包含自身条目", self_entry is not None, self_entry)
    assert self_entry is not None
    self_index, self_expected = self_entry
    prefix_digest = hashlib.sha256(b"".join(raw_lines[:self_index])).hexdigest()
    audit.require("清单自身条目对应写入前缀SHA256", prefix_digest == self_expected,
                  {"前缀行数": self_index, "SHA256": prefix_digest})
    return {
        "清单条目数": len(entries),
        "普通文件核验数": len(entries) - 1,
        "普通文件不一致数": len(mismatches),
        "清单自身说明": "自身条目记录的是写入该行之前的文件前缀；前缀SHA256已匹配",
    }


def audit_predictions(path: Path, metrics: dict, audit: Audit, seed: int, split: str) -> dict:
    expected_hash = metrics["predictions"]["sha256"]
    actual_hash = sha256(path)
    audit.require(f"seed{seed}-{split}预测文件SHA256", actual_hash == expected_hash, actual_hash)

    correct = {name: 0 for name in 方法}
    seen_paths: set[str] = set()
    count = 0
    class_order = metrics["class_order"]
    identity_errors = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            count += 1
            relative_path = row["relative_path"]
            if relative_path in seen_paths:
                identity_errors.append(f"第{line_number}行路径重复")
            seen_paths.add(relative_path)
            label = int(row["label"])
            if not 0 <= label < len(class_order) or row["label_classname"] != class_order[label]:
                identity_errors.append(f"第{line_number}行真值类别不一致")
            for name in 方法:
                prediction = row["predictions"][name]
                predicted_label = int(prediction["label"])
                if not 0 <= predicted_label < len(class_order) or prediction["classname"] != class_order[predicted_label]:
                    identity_errors.append(f"第{line_number}行{name}类别不一致")
                correct[name] += predicted_label == label

    audit.require(f"seed{seed}-{split}预测身份一致", not identity_errors, identity_errors[:20])
    audit.require(f"seed{seed}-{split}预测数量", count == metrics["predictions"]["count"], count)
    recomputed = {}
    for name in 方法:
        expected = metrics["metrics"][name]
        accuracy = 100.0 * correct[name] / count
        valid = (correct[name] == expected["correct"] and count == expected["total"]
                 and close(accuracy, float(expected["accuracy"])))
        audit.require(f"seed{seed}-{split}-{name}独立准确率", valid,
                      {"correct": correct[name], "total": count, "accuracy": accuracy})
        recomputed[name] = {"correct": correct[name], "total": count, "accuracy": accuracy}
    return recomputed


def audit_split(run_root: Path, seed: int, split: str, audit: Audit, dataset_key: str) -> tuple[dict, dict]:
    result_dir = run_root / "融合评估" / f"seed{seed}" / split
    metrics_path = result_dir / "metrics.json"
    metrics = load_json(metrics_path)
    audit.require(f"seed{seed}-{split}评估完成", metrics.get("status") == "passed", metrics.get("status"))
    audit.require(f"seed{seed}-{split}结果身份",
                  (metrics.get("dataset_key"), metrics.get("seed"), metrics.get("subsample")) == (dataset_key, seed, split),
                  {key: metrics.get(key) for key in ("dataset_key", "seed", "subsample")})
    audit.require(f"seed{seed}-{split}固定alpha", close(float(metrics["fixed_protocol"]["alpha"]), 0.5),
                  metrics["fixed_protocol"]["alpha"])
    audit.require(f"seed{seed}-{split}冻结模型源码",
                  metrics["source"]["model_sha256"] == metrics["source"]["expected_model_sha256"] == 预期模型源码摘要,
                  metrics["source"])
    audit.require(f"seed{seed}-{split}冻结CLIP", metrics["assets"]["clip_sha256"] == 预期CLIP摘要,
                  metrics["assets"]["clip_sha256"])
    audit.require(f"seed{seed}-{split}输入文件未被评估修改",
                  metrics["input_hashes_before"] == metrics["input_hashes_after"], metrics["input_hashes_after"])

    checkpoint_dir = run_root / "基线训练" / "baseline" / f"seed{seed}" / "textconsistency"
    checkpoint = checkpoint_dir / "model.pth.tar-20"
    sidecar = checkpoint.with_name(checkpoint.name + ".sha256")
    checkpoint_sha = sha256(checkpoint)
    audit.require(f"seed{seed}-{split}检查点SHA256",
                  checkpoint_sha == metrics["checkpoint"]["sha256"] == metrics["input_hashes_before"]["checkpoint"], checkpoint_sha)
    audit.require(f"seed{seed}-{split}检查点旁路摘要", sidecar.read_text().strip() == checkpoint_sha,
                  sidecar.read_text().strip())

    eval_name = "eval_base_student" if split == "base" else "eval_new_student"
    historical_dir = run_root / "基线训练" / "baseline" / f"seed{seed}" / eval_name
    historical_metrics_path = historical_dir / "metrics.json"
    historical_log_path = historical_dir / "log.txt"
    audit.require(f"seed{seed}-{split}历史指标SHA256",
                  sha256(historical_metrics_path) == metrics["input_hashes_before"]["historical_metrics"],
                  sha256(historical_metrics_path))
    audit.require(f"seed{seed}-{split}历史日志SHA256",
                  sha256(historical_log_path) == metrics["input_hashes_before"]["historical_log"],
                  sha256(historical_log_path))
    historical = load_json(historical_metrics_path)
    audit.require(f"seed{seed}-{split}CoOp重载结果一致",
                  close(float(historical["metrics"]["accuracy"]), float(metrics["metrics"]["student"]["accuracy"])),
                  metrics["metrics"]["student"]["accuracy"])
    run_manifest = load_json(historical_dir / "run_manifest.json")
    audit.require(f"seed{seed}-{split}训练运行签名", run_manifest["code_sha256"] == 预期模型源码摘要,
                  run_manifest["code_sha256"])
    audit.require(f"seed{seed}-{split}运行设备", run_manifest["device"] == "cuda" and "4090" in run_manifest["gpu"],
                  {"device": run_manifest["device"], "gpu": run_manifest["gpu"]})

    recomputed = audit_predictions(result_dir / "predictions.jsonl", metrics, audit, seed, split)
    metadata = {
        "checkpoint_sha256": checkpoint_sha,
        "model_source_sha256": metrics["source"]["model_sha256"],
        "clip_sha256": metrics["assets"]["clip_sha256"],
        "evaluator_sha256": metrics["source"]["evaluator_sha256"],
        "train_samples_sha256": metrics["samples"]["train_sha256"],
        "test_samples_sha256": metrics["samples"]["test_sha256"],
        "predictions_sha256": metrics["predictions"]["sha256"],
    }
    return recomputed, metadata


def audit_runs(input_root: Path, audit: Audit, dataset_key: str, dataset_name: str) -> tuple[list[dict], dict]:
    run_roots = sorted(path for path in (input_root / "runs").iterdir() if path.is_dir())
    audit.require("只包含两段冻结执行", len(run_roots) == 2, [path.name for path in run_roots])
    seed_to_root: dict[int, Path] = {}
    protocols = []
    for run_root in run_roots:
        protocol = load_json(run_root / "实验协议.json")
        status = load_json(run_root / "执行状态.json")
        protocols.append(protocol)
        audit.require(f"{run_root.name}命令全部完成",
                      status["状态"] == "完成" and status["已完成命令"] == status["总命令"] == protocol["命令数"], status)
        audit.require(f"{run_root.name}冻结协议",
                      (protocol["数据集键"], protocol["数据集"], protocol["训练轮数"], protocol["样本数"], protocol["骨干"],
                       protocol["实际模型源码SHA256"]) ==
                      (dataset_key, dataset_name, 20, 16, "ViT-B/16", 预期模型源码摘要),
                      {key: protocol[key] for key in ("数据集键", "数据集", "训练轮数", "样本数", "骨干", "实际模型源码SHA256")})
        for seed in protocol["种子"]:
            audit.require(f"seed{seed}只执行一次", seed not in seed_to_root, str(run_root))
            seed_to_root[seed] = run_root
    audit.require("三种子结果齐全", sorted(seed_to_root) == [1, 2, 3], sorted(seed_to_root))

    rows = []
    metadata_by_seed = {}
    for seed in sorted(seed_to_root):
        root = seed_to_root[seed]
        base, base_metadata = audit_split(root, seed, "base", audit, dataset_key)
        novel, novel_metadata = audit_split(root, seed, "new", audit, dataset_key)
        audit.require(f"seed{seed}两分支使用同一检查点", base_metadata["checkpoint_sha256"] == novel_metadata["checkpoint_sha256"],
                      {"base": base_metadata["checkpoint_sha256"], "novel": novel_metadata["checkpoint_sha256"]})
        metadata_by_seed[str(seed)] = {"base": base_metadata, "novel": novel_metadata}
        method_values = {}
        for name in 方法:
            base_accuracy = base[name]["accuracy"]
            novel_accuracy = novel[name]["accuracy"]
            method_values[name] = {
                "Base": base_accuracy,
                "Novel": novel_accuracy,
                "HM": harmonic_mean(base_accuracy, novel_accuracy),
            }
        rows.append({
            "种子": seed,
            "方法": method_values,
            "融合相对CoOp": {
                "Base增益pp": method_values["mean_probability_fusion"]["Base"] - method_values["student"]["Base"],
                "Novel增益pp": method_values["mean_probability_fusion"]["Novel"] - method_values["student"]["Novel"],
                "HM增益pp": method_values["mean_probability_fusion"]["HM"] - method_values["student"]["HM"],
            },
        })

    train_hashes = [metadata_by_seed[str(seed)]["base"]["train_samples_sha256"] for seed in (1, 2, 3)]
    checkpoint_hashes = [metadata_by_seed[str(seed)]["base"]["checkpoint_sha256"] for seed in (1, 2, 3)]
    audit.require("三种子训练采样互异", len(set(train_hashes)) == 3, train_hashes)
    audit.require("三种子检查点互异", len(set(checkpoint_hashes)) == 3, checkpoint_hashes)
    evaluator_hashes = {
        metadata_by_seed[str(seed)][split]["evaluator_sha256"]
        for seed in (1, 2, 3) for split in ("base", "novel")
    }
    audit.require("六次融合评估器完全一致", len(evaluator_hashes) == 1, sorted(evaluator_hashes))
    return rows, metadata_by_seed


def summarize(rows: list[dict], audit: Audit) -> dict:
    averages = {}
    for name in 方法:
        base = sum(row["方法"][name]["Base"] for row in rows) / len(rows)
        novel = sum(row["方法"][name]["Novel"] for row in rows) / len(rows)
        averages[name] = {
            "Base均值": base,
            "Novel均值": novel,
            "均值的HM": harmonic_mean(base, novel),
            "种子HM均值": sum(row["方法"][name]["HM"] for row in rows) / len(rows),
        }
    fusion = averages["mean_probability_fusion"]
    student = averages["student"]
    positive_count = sum(row["融合相对CoOp"]["HM增益pp"] > 0 for row in rows)
    deltas = {
        "Base增益pp": fusion["Base均值"] - student["Base均值"],
        "Novel增益pp": fusion["Novel均值"] - student["Novel均值"],
        "HM增益pp": fusion["均值的HM"] - student["均值的HM"],
        "正HM增益种子数": positive_count,
    }
    gates = {
        "融合HM严格高于CoOp": deltas["HM增益pp"] > 0,
        "融合Novel相对CoOp不低于-0.5pp": deltas["Novel增益pp"] >= -0.5,
        "至少三分之二种子HM正增益": positive_count >= 2,
    }
    audit.require("冻结三种子门槛全部通过", all(gates.values()), gates)
    return {"逐种子": rows, "均值": averages, "融合相对CoOp": deltas, "门槛": gates, "通过": True}


def write_csv(path: Path, result: dict) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["种子", "方法", "Base", "Novel", "HM", "相对CoOp的HM增益pp"])
        for row in result["逐种子"]:
            for name in 方法:
                values = row["方法"][name]
                delta = row["融合相对CoOp"]["HM增益pp"] if name == "mean_probability_fusion" else ""
                writer.writerow([row["种子"], 方法中文[name], f"{values['Base']:.6f}", f"{values['Novel']:.6f}",
                                 f"{values['HM']:.6f}", "" if delta == "" else f"{delta:.6f}"])
        for name in 方法:
            values = result["均值"][name]
            delta = result["融合相对CoOp"]["HM增益pp"] if name == "mean_probability_fusion" else ""
            writer.writerow(["三种子均值", 方法中文[name], f"{values['Base均值']:.6f}",
                             f"{values['Novel均值']:.6f}", f"{values['均值的HM']:.6f}",
                             "" if delta == "" else f"{delta:.6f}"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-key", default="eurosat")
    parser.add_argument("--dataset-name", default="EuroSAT")
    parser.add_argument("--archive-name", default="EuroSAT扩展验证-seed123-20260915-8f54900.tar.gz")
    parser.add_argument("--expected-archive-sha256",
                        default="3f1c46edb9ecebeb51872b3d8779d0990de6aa8394cb3ef2a3b0004cada9ef2e")
    parser.add_argument("--expected-member-count", type=int, default=121)
    parser.add_argument("--manifest-relative-path",
                        default="setup-logs/扩展验证-20260915-8f54900/服务器文件清单.txt")
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = Audit()
    backup = audit_archive(input_root, audit, args.archive_name, args.expected_archive_sha256,
                           args.expected_member_count)
    manifest = audit_manifest(input_root, audit, args.manifest_relative_path)
    rows, metadata = audit_runs(input_root, audit, args.dataset_key, args.dataset_name)
    result = summarize(rows, audit)
    payload = {
        "状态": "通过" if all(item["通过"] for item in audit.checks) else "失败",
        "数据集": args.dataset_name,
        "协议": "冻结均值概率融合扩展验证-v1",
        "输入根目录": str(input_root),
        "备份": backup,
        "服务器文件清单": manifest,
        "结果": result,
        "运行元数据": metadata,
        "检查项数量": len(audit.checks),
        "失败检查项数量": sum(not item["通过"] for item in audit.checks),
        "检查项": audit.checks,
    }
    (output_dir / "独立复算.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    write_csv(output_dir / "三种子结果.csv", result)
    print(json.dumps({
        "状态": payload["状态"],
        "检查项数量": payload["检查项数量"],
        "失败检查项数量": payload["失败检查项数量"],
        "融合相对CoOp": result["融合相对CoOp"],
        "门槛": result["门槛"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
