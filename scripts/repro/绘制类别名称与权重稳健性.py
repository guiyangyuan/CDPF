#!/usr/bin/env python3
"""从投稿关键对照的中文 CSV 生成论文图。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


数据集顺序 = [
    "UCF101", "DTD", "FGVCAircraft", "EuroSAT", "OxfordFlowers",
    "Caltech101", "OxfordPets", "Food101", "StanfordCars", "SUN397",
]


def 读取数据(summary_dir: Path) -> tuple[list[float], dict[str, list[float]]]:
    with (summary_dir / "关键对照数据集汇总.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    index = {
        (row["数据集"], row["方法"]): float(row["HM_均值计算"])
        for row in rows
    }
    gains = [
        index[(dataset, "TextConsistency")] - index[(dataset, "CoOp+类别名称")]
        for dataset in 数据集顺序
    ]

    with (summary_dir / "统一权重稳健性.csv").open(encoding="utf-8-sig") as stream:
        alpha_rows = list(csv.DictReader(stream))
    curves = {
        "alpha": [float(row["alpha"]) for row in alpha_rows],
        "Base": [float(row["宏平均Base"]) for row in alpha_rows],
        "Novel": [float(row["宏平均Novel"]) for row in alpha_rows],
        "HM": [float(row["宏平均HM"]) for row in alpha_rows],
    }
    return gains, curves


def 绘图(summary_dir: Path, output_base: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gains, curves = 读取数据(summary_dir)
    short_names = ["UCF", "DTD", "Aircraft", "EuroSAT", "Flowers", "Caltech", "Pets", "Food", "Cars", "SUN"]
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.45), constrained_layout=True)

    colors = ["#2f6f9f" if value >= 0 else "#b24745" for value in gains]
    axes[0].bar(range(len(gains)), gains, color=colors, width=0.68)
    axes[0].axhline(0, color="#444444", linewidth=0.8)
    axes[0].set_xticks(range(len(gains)), short_names, rotation=28, ha="right")
    axes[0].set_ylabel(r"HM gain over CoOp + class name (pp)")
    axes[0].set_title("(a) Description-specific contribution")
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)
    for index, value in enumerate(gains):
        axes[0].text(index, value + (0.13 if value >= 0 else -0.13), f"{value:+.2f}",
                     ha="center", va="bottom" if value >= 0 else "top", fontsize=7.5)

    styles = {"Base": ("#577590", "o"), "Novel": ("#f8961e", "s"), "HM": ("#2a9d8f", "D")}
    for label in ("Base", "Novel", "HM"):
        color, marker = styles[label]
        axes[1].plot(curves["alpha"], curves[label], label=label, color=color,
                     marker=marker, linewidth=1.8, markersize=4.8)
    axes[1].axvline(0.5, color="#555555", linestyle="--", linewidth=0.9, alpha=0.7)
    axes[1].set_xticks(curves["alpha"])
    axes[1].set_xlabel(r"Global description weight $\alpha$")
    axes[1].set_ylabel("Ten-dataset macro average (%)")
    axes[1].set_title("(b) One global weight for all datasets")
    axes[1].grid(alpha=0.25, linewidth=0.6)
    axes[1].legend(frameon=False, ncol=3, loc="lower center")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_base.with_suffix(".pdf"), bbox_inches="tight",
        metadata={"Creator": "TextConsistency", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        output_base.with_suffix(".png"), dpi=220, bbox_inches="tight",
        metadata={"Software": "TextConsistency"},
    )
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    args = parser.parse_args()
    绘图(args.summary_dir, args.output_base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
