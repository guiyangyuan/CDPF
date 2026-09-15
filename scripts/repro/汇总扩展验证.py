#!/usr/bin/env python3
"""汇总扩展验证的 Base/Novel 结果并执行冻结门槛。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


方法 = ("student", "description_mean", "simple_fusion", "mean_probability_fusion")
方法中文 = {
    "student": "CoOp",
    "description_mean": "冻结均值描述",
    "simple_fusion": "固定logit均值融合",
    "mean_probability_fusion": "固定均值概率融合",
}


def hm(base: float, novel: float) -> float:
    return 0.0 if base + novel == 0 else 2.0 * base * novel / (base + novel)


def 读取指标(path: Path, dataset: str, seed: int, split: str) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "passed":
        raise ValueError(f"融合评估未通过: {path}")
    if (payload.get("dataset_key"), payload.get("seed"), payload.get("subsample")) != (dataset, seed, split):
        raise ValueError(f"结果身份不匹配: {path}")
    stats = payload.get("metrics", {})
    if any(name not in stats for name in 方法):
        raise ValueError(f"结果缺少冻结分支: {path}")
    values = {name: float(stats[name]["accuracy"]) for name in 方法}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError(f"结果包含非有限值: {path}")
    return values


def 汇总(root: Path) -> dict:
    root = root.resolve()
    protocol = json.loads((root / "实验协议.json").read_text())
    dataset = protocol["数据集键"]
    seeds = protocol["种子"]
    rows = []
    for seed in seeds:
        base = 读取指标(root / "融合评估" / f"seed{seed}" / "base" / "metrics.json", dataset, seed, "base")
        novel = 读取指标(root / "融合评估" / f"seed{seed}" / "new" / "metrics.json", dataset, seed, "new")
        row = {"种子": seed, "方法": {}}
        for name in 方法:
            row["方法"][name] = {"Base": base[name], "Novel": novel[name], "HM": hm(base[name], novel[name])}
        row["融合相对CoOp"] = {
            "Base增益pp": base["mean_probability_fusion"] - base["student"],
            "Novel增益pp": novel["mean_probability_fusion"] - novel["student"],
            "HM增益pp": row["方法"]["mean_probability_fusion"]["HM"] - row["方法"]["student"]["HM"],
        }
        rows.append(row)

    averages = {}
    for name in 方法:
        base = sum(row["方法"][name]["Base"] for row in rows) / len(rows)
        novel = sum(row["方法"][name]["Novel"] for row in rows) / len(rows)
        averages[name] = {
            "Base均值": base,
            "Novel均值": novel,
            "均值的HM": hm(base, novel),
            "种子HM均值": sum(row["方法"][name]["HM"] for row in rows) / len(rows),
        }
    hm_delta = averages["mean_probability_fusion"]["均值的HM"] - averages["student"]["均值的HM"]
    novel_delta = averages["mean_probability_fusion"]["Novel均值"] - averages["student"]["Novel均值"]
    positive = sum(row["融合相对CoOp"]["HM增益pp"] > 0 for row in rows)
    required = 1 if len(rows) == 1 else math.ceil(2 * len(rows) / 3)
    checks = {
        "融合HM严格高于CoOp": hm_delta > 0,
        "融合Novel相对CoOp不低于-0.5pp": novel_delta >= -0.5,
        "达到正HM增益种子数": positive >= required,
    }
    passed = all(checks.values())
    result = {
        "协议": protocol["协议"],
        "数据集": protocol["数据集"],
        "种子": seeds,
        "逐种子": rows,
        "均值": averages,
        "融合相对CoOp": {"Novel增益pp": novel_delta, "HM增益pp": hm_delta},
        "门槛": checks,
        "通过": passed,
        "下一步": ("单种子通过：扩展seed2/3" if passed and len(seeds) == 1 else
                  "三种子通过：可纳入扩展证据" if passed else "停止该数据集，不修改融合公式"),
    }
    return result


def 写入报告(root: Path, result: dict) -> None:
    rows = []
    for name in 方法:
        value = result["均值"][name]
        rows.append(f"| {方法中文[name]} | {value['Base均值']:.3f} | {value['Novel均值']:.3f} | {value['均值的HM']:.3f} |")
    lines = [
        f"# {result['数据集']} 扩展验证结果", "",
        f"**判定：{'通过' if result['通过'] else '未通过'}。** {result['下一步']}", "",
        "| 方法 | Base均值 | Novel均值 | HM |", "|---|---:|---:|---:|", *rows, "",
        f"固定均值概率融合相对CoOp：Novel {result['融合相对CoOp']['Novel增益pp']:+.3f}pp，"
        f"HM {result['融合相对CoOp']['HM增益pp']:+.3f}pp。", "", "## 冻结门槛", "",
    ]
    lines.extend(f"- {'通过' if value else '未通过'}：{name}" for name, value in result["门槛"].items())
    lines += ["", "本实验固定alpha=0.5，不按结果搜索权重、温度、描述、轮数或阈值。"
              "该数据集有历史调参或论文成绩，本轮只能作为未参与当前融合规则选择的扩展验证，不能称严格盲测。", ""]
    (root / "结果报告.md").write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = 汇总(args.root)
    (args.root / "结果汇总.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    写入报告(args.root, result)
    print(json.dumps({"通过": result["通过"], "下一步": result["下一步"],
                      "融合相对CoOp": result["融合相对CoOp"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
