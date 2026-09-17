#!/usr/bin/env python3
"""汇总十数据集类别名称对照和全局固定权重稳健性。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.repro.evaluate_frozen_fusion import ALPHA_GRID, alpha_branch_name


数据集顺序 = [
    "UCF101", "DTD", "FGVCAircraft", "EuroSAT", "OxfordFlowers",
    "Caltech101", "OxfordPets", "Food101", "StanfordCars", "SUN397",
]
方法 = {
    "CoOp": "student",
    "类别名称": "classname",
    "CoOp+类别名称": "classname_probability_fusion",
    "类别描述": "description_mean",
    "TextConsistency": alpha_branch_name("description_probability_alpha", 0.5),
}


def 调和平均(base: float, novel: float) -> float:
    return 0.0 if base + novel == 0 else 2.0 * base * novel / (base + novel)


def 读取输入(root: Path, datasets: list[str]) -> list[dict]:
    rows = []
    expected = {
        (dataset, seed, split)
        for dataset in datasets for seed in (1, 2, 3) for split in ("base", "new")
    }
    actual = set()
    for path in sorted(root.rglob("metrics.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        identity = (value.get("dataset"), value.get("seed"), value.get("subsample"))
        if identity not in expected:
            continue
        if identity in actual:
            raise ValueError(f"重复的数据集/种子/分支：{identity}")
        if value.get("status") != "passed" or value.get("split") != "test":
            raise ValueError(f"评估未通过：{path}")
        metrics = value.get("metrics", {})
        required = set(方法.values()) | {
            alpha_branch_name("description_probability_alpha", alpha) for alpha in ALPHA_GRID
        } | {"mean_probability_fusion"}
        missing = sorted(required - set(metrics))
        if missing:
            raise ValueError(f"评估缺少分支 {missing}：{path}")
        for name, branch in (
            ("alpha=0", alpha_branch_name("description_probability_alpha", 0.0)),
            ("alpha=0.5", alpha_branch_name("description_probability_alpha", 0.5)),
            ("alpha=1", alpha_branch_name("description_probability_alpha", 1.0)),
        ):
            reference = {
                "alpha=0": "student",
                "alpha=0.5": "mean_probability_fusion",
                "alpha=1": "description_mean",
            }[name]
            if metrics[branch] != metrics[reference]:
                raise ValueError(f"{name}端点/主方法不与原始分支一致：{path}")
        for branch in required:
            row = metrics[branch]
            if (set(row) != {"correct", "total", "accuracy"} or int(row["total"]) <= 0 or
                    int(row["correct"]) < 0 or int(row["correct"]) > int(row["total"])):
                raise ValueError(f"指标结构非法：{path} {branch}")
            expected_accuracy = 100.0 * int(row["correct"]) / int(row["total"])
            if not math.isclose(float(row["accuracy"]), expected_accuracy, abs_tol=1e-10):
                raise ValueError(f"准确率不可复算：{path} {branch}")
        rows.append({
            "数据集": identity[0], "种子": int(identity[1]), "分支": identity[2],
            "指标": metrics, "来源": str(path),
        })
        actual.add(identity)
    if actual != expected:
        raise ValueError(f"输入不完整：缺少 {sorted(expected - actual)}")
    return rows


def 构造逐种子(rows: list[dict], datasets: list[str]) -> list[dict]:
    index = {(row["数据集"], row["种子"], row["分支"]): row for row in rows}
    output = []
    for dataset in datasets:
        for seed in (1, 2, 3):
            for method, branch in 方法.items():
                base = float(index[(dataset, seed, "base")]["指标"][branch]["accuracy"])
                novel = float(index[(dataset, seed, "new")]["指标"][branch]["accuracy"])
                output.append({"数据集": dataset, "种子": seed, "方法": method,
                               "Base": base, "Novel": novel, "HM": 调和平均(base, novel)})
    return output


def 汇总方法(seed_rows: list[dict], datasets: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row["数据集"], row["方法"])].append(row)
    output = []
    for dataset in datasets:
        for method in 方法:
            group = sorted(grouped[(dataset, method)], key=lambda item: item["种子"])
            base = [row["Base"] for row in group]
            novel = [row["Novel"] for row in group]
            hms = [row["HM"] for row in group]
            base_mean, novel_mean = statistics.mean(base), statistics.mean(novel)
            output.append({
                "数据集": dataset, "方法": method, "种子数": 3,
                "Base均值": base_mean, "Base标准差": statistics.stdev(base),
                "Novel均值": novel_mean, "Novel标准差": statistics.stdev(novel),
                "HM_均值计算": 调和平均(base_mean, novel_mean),
                "种子HM均值": statistics.mean(hms), "种子HM标准差": statistics.stdev(hms),
            })
    return output


def 汇总权重(rows: list[dict], datasets: list[str]) -> list[dict]:
    index = {(row["数据集"], row["种子"], row["分支"]): row for row in rows}
    output = []
    for alpha in ALPHA_GRID:
        branch = alpha_branch_name("description_probability_alpha", alpha)
        dataset_hm = []
        dataset_base = []
        dataset_novel = []
        for dataset in datasets:
            base = statistics.mean(
                float(index[(dataset, seed, "base")]["指标"][branch]["accuracy"])
                for seed in (1, 2, 3))
            novel = statistics.mean(
                float(index[(dataset, seed, "new")]["指标"][branch]["accuracy"])
                for seed in (1, 2, 3))
            dataset_base.append(base)
            dataset_novel.append(novel)
            dataset_hm.append(调和平均(base, novel))
        output.append({
            "alpha": alpha, "数据集数": len(datasets),
            "宏平均Base": statistics.mean(dataset_base),
            "宏平均Novel": statistics.mean(dataset_novel),
            "宏平均HM": statistics.mean(dataset_hm),
            "数据集HM标准差": statistics.stdev(dataset_hm),
        })
    return output


def 写csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def 生成结果(input_root: Path, output_dir: Path, datasets: list[str] | None = None) -> dict:
    datasets = list(数据集顺序 if datasets is None else datasets)
    if not datasets or len(set(datasets)) != len(datasets) or any(item not in 数据集顺序 for item in datasets):
        raise ValueError("数据集列表必须是非空、无重复的固定十数据集子集")
    rows = 读取输入(input_root, datasets)
    seed_rows = 构造逐种子(rows, datasets)
    summaries = 汇总方法(seed_rows, datasets)
    alpha_rows = 汇总权重(rows, datasets)
    summary_index = {(row["数据集"], row["方法"]): row for row in summaries}
    macro = {
        method: statistics.mean(summary_index[(dataset, method)]["HM_均值计算"]
                                for dataset in datasets)
        for method in 方法
    }
    description_wins = sum(
        summary_index[(dataset, "TextConsistency")]["HM_均值计算"] >
        summary_index[(dataset, "CoOp+类别名称")]["HM_均值计算"]
        for dataset in datasets
    )
    paired_wins = sum(
        row["HM"] > next(item["HM"] for item in seed_rows
                         if item["数据集"] == row["数据集"] and
                         item["种子"] == row["种子"] and item["方法"] == "CoOp+类别名称")
        for row in seed_rows if row["方法"] == "TextConsistency"
    )
    result = {
        "状态": "通过", "协议": "投稿关键对照-v1", "数据集": datasets,
        "种子": [1, 2, 3], "全局alpha网格": list(ALPHA_GRID),
        "方法宏平均HM": macro,
        "核心比较": {
            "TextConsistency减CoOp+类别名称_HM_pp": macro["TextConsistency"] - macro["CoOp+类别名称"],
            "TextConsistency胜出数据集数": description_wins,
            "TextConsistency胜出配对种子数": paired_wins,
            "配对种子总数": len(datasets) * 3,
        },
        "权重稳健性": alpha_rows,
        "选择边界": "alpha网格仅用于事后稳健性报告，不按数据集选最优值，主方法仍固定alpha=0.5。",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    写csv(output_dir / "关键对照逐种子.csv", seed_rows)
    写csv(output_dir / "关键对照数据集汇总.csv", summaries)
    写csv(output_dir / "统一权重稳健性.csv", alpha_rows)
    (output_dir / "结果.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# 类别名称与权重稳健性对照",
        "",
        f"- TextConsistency 宏平均 HM：{macro['TextConsistency']:.2f}",
        f"- CoOp+类别名称 宏平均 HM：{macro['CoOp+类别名称']:.2f}",
        f"- 描述语义净增益：{result['核心比较']['TextConsistency减CoOp+类别名称_HM_pp']:+.2f} pp",
        f"- 胜出数据集：{description_wins}/{len(datasets)}；胜出配对种子：{paired_wins}/{len(datasets) * 3}",
        "",
        "## 全局权重网格",
        "",
        "| alpha | Base | Novel | HM |",
        "|---:|---:|---:|---:|",
    ]
    report.extend(
        f"| {row['alpha']:.2f} | {row['宏平均Base']:.2f} | {row['宏平均Novel']:.2f} | {row['宏平均HM']:.2f} |"
        for row in alpha_rows
    )
    report.extend(["", result["选择边界"]])
    (output_dir / "结果报告.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=数据集顺序, default=None,
                        help="可选的固定数据集子集；默认严格要求十数据集全部齐全")
    args = parser.parse_args()
    print(json.dumps(生成结果(args.input_root, args.output_dir, args.datasets), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
