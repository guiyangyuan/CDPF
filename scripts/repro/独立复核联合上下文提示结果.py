#!/usr/bin/env python3
"""独立复核 CoCoOp 阶段结果、原始日志、检查点与文件清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


准确率模式 = re.compile(r"^\* accuracy:\s*([0-9.]+)%$", re.MULTILINE)
正确数模式 = re.compile(r"^\* correct:\s*([0-9,]+)$", re.MULTILINE)
总数模式 = re.compile(r"^\* total:\s*([0-9,]+)$", re.MULTILINE)
阶段数据集 = {
    "核心三数据集": ["UCF101", "DTD", "FGVC-Aircraft"],
    "扩展两数据集": ["EuroSAT", "OxfordFlowers"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"数值不一致：{actual} != {expected}")


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    accuracy = 准确率模式.findall(text)
    correct = 正确数模式.findall(text)
    total = 总数模式.findall(text)
    if len(accuracy) != 1 or len(correct) != 1 or len(total) != 1:
        raise ValueError(f"评估日志指标数量异常：{path}")
    correct_value = int(correct[0].replace(",", ""))
    total_value = int(total[0].replace(",", ""))
    recomputed = 100.0 * correct_value / total_value
    close(float(accuracy[0]), recomputed, 0.00051)
    return {
        "打印准确率": float(accuracy[0]),
        "正确数": correct_value,
        "总数": total_value,
        "复算准确率": recomputed,
        "SHA256": sha256(path),
    }


def verify_manifest(stage_root: Path) -> int:
    manifest = stage_root / "文件清单.txt"
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = stage_root / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"文件清单不一致：{relative}")
        entries.append(relative)
    actual = sorted(
        path.relative_to(stage_root).as_posix()
        for path in stage_root.rglob("*")
        if path.is_file() and path != manifest
    )
    if sorted(entries) != actual:
        raise ValueError("文件清单存在缺项、重复或额外文件")
    return len(entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.stage_root.resolve()
    plan = json.loads((root / "执行计划.json").read_text(encoding="utf-8"))
    status = json.loads((root / "执行状态.json").read_text(encoding="utf-8"))
    result = json.loads((root / "阶段结果.json").read_text(encoding="utf-8"))
    stage = result["阶段"]
    datasets = 阶段数据集[stage]
    expected_runs = len(datasets) * 3
    checks = []

    if plan["协议"] != "投稿CoCoOp五数据集标准基线-v1" or plan["运行数"] != 15:
        raise ValueError("执行计划身份错误")
    if status["状态"] != "完成" or result["状态"] != "通过":
        raise ValueError("阶段未完成")
    if status["已完成命令"] != expected_runs * 3 or len(status["命令"]) != expected_runs * 3:
        raise ValueError("命令数量错误")
    if any(record["退出码"] != 0 for record in status["命令"]):
        raise ValueError("存在失败命令")
    checks.append(f"{expected_runs * 3}条命令全部成功")

    manifest_count = verify_manifest(root)
    checks.append(f"文件清单中的{manifest_count}个文件逐SHA256一致且无缺项")

    rows = result["逐种子"]
    identities = {(row["数据集"], row["种子"]) for row in rows}
    expected_identities = {(dataset, seed) for dataset in datasets for seed in (1, 2, 3)}
    if len(rows) != expected_runs or identities != expected_identities:
        raise ValueError("逐种子身份不完整或重复")

    audited = []
    for row in rows:
        run_root = root / row["数据集"] / f"种子{row['种子']}"
        checkpoint = run_root / "训练/prompt_learner/model.pth.tar-10"
        checkpoint_sha = sha256(checkpoint)
        if checkpoint_sha != row["检查点SHA256"]:
            raise ValueError(f"检查点摘要不一致：{row['数据集']} seed{row['种子']}")
        base = parse_log(run_root / "基础测试/log.txt")
        novel = parse_log(run_root / "新类别测试/log.txt")
        close(base["复算准确率"], row["Base"])
        close(novel["复算准确率"], row["Novel"])
        hm = 2.0 * row["Base"] * row["Novel"] / (row["Base"] + row["Novel"])
        close(hm, row["HM"])
        if base["SHA256"] != row["基础测试日志SHA256"] or novel["SHA256"] != row["新类别测试日志SHA256"]:
            raise ValueError("评估日志摘要不一致")
        audited.append(
            {
                "数据集": row["数据集"],
                "种子": row["种子"],
                "Base": base,
                "Novel": novel,
                "HM": hm,
                "检查点SHA256": checkpoint_sha,
            }
        )
    checks.append(f"{expected_runs}个检查点和{expected_runs * 2}份原始评估日志通过复算")

    summary_by_dataset = {item["数据集"]: item for item in result["数据集汇总"]}
    for dataset in datasets:
        group = [row for row in rows if row["数据集"] == dataset]
        base_values = [row["Base"] for row in group]
        novel_values = [row["Novel"] for row in group]
        hm_values = [row["HM"] for row in group]
        expected = summary_by_dataset[dataset]
        base_mean = statistics.mean(base_values)
        novel_mean = statistics.mean(novel_values)
        close(base_mean, expected["Base均值"])
        close(statistics.stdev(base_values), expected["Base样本标准差"])
        close(novel_mean, expected["Novel均值"])
        close(statistics.stdev(novel_values), expected["Novel样本标准差"])
        close(2.0 * base_mean * novel_mean / (base_mean + novel_mean), expected["HM_均值计算"])
        close(statistics.mean(hm_values), expected["种子HM均值"])
        close(statistics.stdev(hm_values), expected["种子HM样本标准差"])
    checks.append(f"{len(datasets)}个数据集的均值、样本标准差和HM独立复算一致")

    payload = {
        "状态": "通过",
        "阶段": stage,
        "检查项": checks,
        "计划SHA256": sha256(root / "执行计划.json"),
        "阶段结果SHA256": sha256(root / "阶段结果.json"),
        "文件清单SHA256": sha256(root / "文件清单.txt"),
        "逐种子复核": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"状态": "通过", "阶段": stage, "逐种子": len(audited), "输出": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
