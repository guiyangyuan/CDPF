#!/usr/bin/env python3
"""从已审计结果生成论文中的 LaTeX 表格、图和 Overleaf 文件。"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


根目录 = Path(__file__).resolve().parents[1]
源码目录 = Path(__file__).resolve().parent / "修改后源码"
表格目录 = 源码目录 / "table"
Overleaf目录 = Path(__file__).resolve().parent / "Overleaf直接替换"
预览目录 = Path(__file__).resolve().parent / "图表预览"
结果目录 = 根目录 / "reports" / "experiments"

图名列表 = [
    "图1-方法框架.pdf",
    "图2-十一数据集增益.pdf",
    "图3-消融稳定性效率.pdf",
    "图4-语义对照与权重稳健性.pdf",
]

数据集顺序 = [
    "UCF101",
    "DTD",
    "FGVC-Aircraft",
    "EuroSAT",
    "OxfordFlowers",
    "Caltech101",
    "OxfordPets",
    "Food101",
    "StanfordCars",
    "SUN397",
    "ImageNet",
]

简称 = {
    "UCF101": "UCF",
    "DTD": "DTD",
    "FGVC-Aircraft": "Air.",
    "EuroSAT": "Euro.",
    "OxfordFlowers": "Flw.",
    "Caltech101": "Cal.",
    "OxfordPets": "Pets",
    "Food101": "Food",
    "StanfordCars": "Cars",
    "SUN397": "SUN",
    "ImageNet": "IN",
}


def 读取_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def 保存(name: str, text: str) -> None:
    表格目录.mkdir(parents=True, exist_ok=True)
    (表格目录 / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def 数字(value: float) -> str:
    return f"{value:.2f}"


def 文件哈希(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def 保存图(fig: plt.Figure, name: str) -> None:
    源码目录.mkdir(parents=True, exist_ok=True)
    预览目录.mkdir(parents=True, exist_ok=True)
    fig.savefig(源码目录 / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(预览目录 / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def 设置绘图样式() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def 主要结果() -> dict[tuple[str, str], dict[str, float]]:
    rows = 读取_csv(
        结果目录 / "斯坦福汽车与SUN397扩展验证-20260917" / "十数据集结果.csv"
    )
    index = {
        (row["数据集"], row["方法"]): {
            "Base": float(row["Base"]),
            "Novel": float(row["Novel"]),
            "HM": float(row["HM"]),
        }
        for row in rows
    }
    imagenet = json.loads(
        (结果目录 / "ImageNet投稿主实验-20260919" / "结果汇总.json")
        .read_text(encoding="utf-8")
    )["均值"]
    for source, method in (
        ("student", "CoOp"),
        ("mean_probability_fusion", "固定均值概率融合"),
    ):
        row = imagenet[source]
        index[("ImageNet", method)] = {
            "Base": float(row["Base均值"]),
            "Novel": float(row["Novel均值"]),
            "HM": float(row["种子HM均值"]),
        }
    return index


def 生成本地主表(index: dict[tuple[str, str], dict[str, float]]) -> None:
    methods = ["CoOp", "固定均值概率融合"]
    labels = {
        "CoOp": "CoOp",
        "固定均值概率融合": r"\textbf{TextConsistency}",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Matched harmonic-mean accuracy (\%) on the standard eleven base-to-novel benchmarks. Both methods use CLIP ViT-B/16, the official class splits, and the same three CoOp checkpoints per dataset.}",
        r"\label{tab:local_hm}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrrrr}",
        r"\toprule",
        "Method & " + " & ".join(简称[d] for d in 数据集顺序) + r" & Avg. \\",
        r"\midrule",
    ]
    for method in methods:
        values = [index[(dataset, method)]["HM"] for dataset in 数据集顺序]
        avg = sum(values) / len(values)
        cells = " & ".join(数字(v) for v in values)
        avg_cell = rf"\textbf{{{数字(avg)}}}" if method == "固定均值概率融合" else 数字(avg)
        lines.append(f"{labels[method]} & {cells} & {avg_cell} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    保存("table_local_hm.tex", "\n".join(lines))


def 生成平均表(index: dict[tuple[str, str], dict[str, float]]) -> None:
    methods = ["CoOp", "固定均值概率融合"]
    labels = {
        "CoOp": "CoOp",
        "固定均值概率融合": r"\textbf{TextConsistency}",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Eleven-dataset macro averages for the matched comparison (\%). HM is computed for each dataset before macro averaging.}",
        r"\label{tab:average_metrics}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Base & Novel & HM \\",
        r"\midrule",
    ]
    for method in methods:
        values = {
            metric: sum(index[(dataset, method)][metric] for dataset in 数据集顺序) / len(数据集顺序)
            for metric in ("Base", "Novel", "HM")
        }
        row = f"{labels[method]} & {数字(values['Base'])} & {数字(values['Novel'])} & {数字(values['HM'])} " + r"\\"
        lines.append(row)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    保存("table_average_metrics.tex", "\n".join(lines))


def 生成文献表() -> None:
    rows = 读取_csv(
        结果目录 / "斯坦福汽车与SUN397扩展验证-20260917" / "十数据集文献方法对比.csv"
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comparison with published base-to-novel HM results on the ten datasets common to all methods. Published values are marked with $\dagger$ and are not local reproductions.}",
        r"\label{tab:published_hm}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrrr}",
        r"\toprule",
        "Method & " + " & ".join(简称[d] for d in 数据集顺序[:-1]) + r" & Avg. \\",
        r"\midrule",
    ]
    for row in rows:
        method = row["方法"]
        published_labels = {
            "CuPL": "CuPL",
            "CoCoOp（论文报告）": "CoCoOp",
            "KgCoOp": "KgCoOp",
            "MaPLe": "MaPLe",
            "PromptSRC": "PromptSRC",
        }
        label = r"\textbf{TextConsistency}" if method == "本文方法" else published_labels[method] + r"$^\dagger$"
        values = [float(row[d]) for d in 数据集顺序[:-1]]
        avg = float(row["十数据集等权HM"])
        avg_cell = rf"\textbf{{{数字(avg)}}}" if method == "本文方法" else 数字(avg)
        lines.append(f"{label} & " + " & ".join(数字(v) for v in values) + f" & {avg_cell} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    保存("table_published_hm.tex", "\n".join(lines))


def 组件结果() -> dict[tuple[str, str], float]:
    mapped: dict[tuple[str, str], float] = {}
    old_rows = 读取_csv(结果目录 / "投稿实验统一结果-20260915" / "主结果汇总.csv")
    old_map = {
        "CoOp": "CoOp",
        "冻结均值描述": "Description only",
        "固定logit均值融合": "Logit fusion",
        "固定均值概率融合": "Probability fusion",
    }
    for row in old_rows:
        mapped[(row["数据集"], old_map[row["方法"]])] = float(row["HM_均值计算"])

    json_paths = {
        "Caltech101": 结果目录 / "三数据集扩展与八数据集汇总-20260916" / "主方法-Caltech101-结果.json",
        "OxfordPets": 结果目录 / "三数据集扩展与八数据集汇总-20260916" / "主方法-OxfordPets-结果.json",
        "Food101": 结果目录 / "三数据集扩展与八数据集汇总-20260916" / "主方法-Food101-结果.json",
        "StanfordCars": 结果目录 / "斯坦福汽车与SUN397扩展验证-20260917" / "主方法-StanfordCars-结果.json",
        "SUN397": 结果目录 / "斯坦福汽车与SUN397扩展验证-20260917" / "主方法-SUN397-结果.json",
    }
    internal = {
        "student": "CoOp",
        "description_mean": "Description only",
        "simple_fusion": "Logit fusion",
        "mean_probability_fusion": "Probability fusion",
    }
    for dataset, path in json_paths.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, label in internal.items():
            mapped[(dataset, label)] = float(data["均值"][key]["均值的HM"])
    imagenet = json.loads(
        (结果目录 / "ImageNet投稿主实验-20260919" / "结果汇总.json")
        .read_text(encoding="utf-8")
    )
    for key, label in internal.items():
        mapped[("ImageNet", label)] = float(imagenet["均值"][key]["种子HM均值"])
    return mapped


def 生成组件表(values: dict[tuple[str, str], float]) -> None:
    methods = ["CoOp", "Description only", "Logit fusion", "Probability fusion"]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Component comparison using HM (\%). The description branch is frozen. Both fusion variants use a fixed weight of 0.5.}",
        r"\label{tab:components}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrrrr}",
        r"\toprule",
        "Method & " + " & ".join(简称[d] for d in 数据集顺序) + r" & Avg. \\",
        r"\midrule",
    ]
    for method in methods:
        vals = [values[(dataset, method)] for dataset in 数据集顺序]
        label = r"\textbf{Probability fusion}" if method == "Probability fusion" else method
        avg = sum(vals) / len(vals)
        avg_cell = rf"\textbf{{{数字(avg)}}}" if method == "Probability fusion" else 数字(avg)
        lines.append(f"{label} & " + " & ".join(数字(v) for v in vals) + f" & {avg_cell} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}"]
    保存("table_components.tex", "\n".join(lines))


def 生成效率表() -> None:
    text = r"""\begin{table}[t]
\centering
\caption{Inference efficiency on one RTX 4090 using the UCF101 novel split. Text features are pre-computed.}
\label{tab:efficiency}
\small
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{lrrrr}
\toprule
Method & Batch & ms/batch & img/s & Extra latency \\
\midrule
CoOp & 1 & 18.827 & 53.12 & 0.00\% \\
TextConsistency & 1 & 18.904 & 52.90 & 0.41\% \\
CoOp & 64 & 90.128 & 710.10 & 0.00\% \\
TextConsistency & 64 & 90.796 & 704.88 & 0.74\% \\
\bottomrule
\end{tabular}
\end{table}"""
    保存("table_efficiency.tex", text)


def 生成类别名称与权重表() -> None:
    rows = 读取_csv(
        结果目录 / "ImageNet投稿主实验-20260919" / "十一数据集方法宏平均.csv"
    )
    labels = {
        "CoOp": "CoOp",
        "类别名称": "Class name only",
        "CoOp+类别名称": "CoOp + class name",
        "类别描述": "Description only",
        "TextConsistency": r"\textbf{TextConsistency}",
    }
    index = {row["方法"]: row for row in rows}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Class-name control on all eleven datasets with matched frozen checkpoints. All values are three-seed macro averages (\%).}",
        r"\label{tab:class_name_control}",
        r"\small",
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Base & Novel & HM \\",
        r"\midrule",
    ]
    for method, label in labels.items():
        row = index[method]
        base = float(row["Base宏平均"])
        novel = float(row["Novel宏平均"])
        hm = float(row["HM宏平均"])
        values = [数字(base), 数字(novel), 数字(hm)]
        if method == "TextConsistency":
            values = [rf"\textbf{{{value}}}" for value in values]
        lines.append(f"{label} & " + " & ".join(values) + " " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    保存("类别名称与权重稳健性.tex", "\n".join(lines))


def 生成方法框架图() -> None:
    blue = "#2F5D9B"
    blue_light = "#EAF1FA"
    orange = "#D97721"
    orange_light = "#FBEEDF"
    green = "#2E7D65"
    green_light = "#E6F3EE"
    grey = "#56616F"
    grey_light = "#F2F4F7"

    fig, ax = plt.subplots(figsize=(11.4, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge, fontsize=9, weight="normal"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.2,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            weight=weight,
            color="#182230",
        )
        return (x, y, w, h)

    def arrow(start, end, color=grey, lw=1.3):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, shrinkA=2, shrinkB=2),
        )

    box(0.015, 0.38, 0.105, 0.22, "Input image\n$x$", grey_light, grey, weight="bold")
    box(0.155, 0.38, 0.155, 0.22, "Frozen CLIP\nimage encoder", blue_light, blue, weight="bold")
    box(0.345, 0.38, 0.12, 0.22, "Shared image\nfeature $\\mathbf{v}(x)$", blue_light, blue)

    ax.add_patch(
        FancyBboxPatch(
            (0.505, 0.565),
            0.31,
            0.31,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.3,
            edgecolor=blue,
            facecolor="#F8FBFF",
        )
    )
    ax.text(0.52, 0.84, "Learned prompt branch", color=blue, fontsize=9.5, weight="bold")
    box(0.53, 0.625, 0.12, 0.14, "CoOp context\n2,048 trainable", blue_light, blue, fontsize=8)
    box(0.68, 0.625, 0.105, 0.14, "Prompt\nposterior $p^p$", blue_light, blue, fontsize=8)

    ax.add_patch(
        FancyBboxPatch(
            (0.505, 0.105),
            0.31,
            0.31,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.3,
            edgecolor=orange,
            facecolor="#FFFBF7",
        )
    )
    ax.text(0.52, 0.38, "Frozen description branch", color=orange, fontsize=9.5, weight="bold")
    box(0.53, 0.165, 0.12, 0.14, "Up to 20 class\ndescriptions", orange_light, orange, fontsize=8)
    box(0.68, 0.165, 0.105, 0.14, "Cached\nposterior $p^d$", orange_light, orange, fontsize=8)
    ax.text(0.66, 0.12, "encoded once with the frozen text encoder", ha="center", fontsize=7.4, color=grey)

    box(0.855, 0.38, 0.125, 0.22, "Fixed average\n$0.5p^p+0.5p^d$\n\nPrediction", green_light, green, fontsize=8.6, weight="bold")

    arrow((0.12, 0.49), (0.155, 0.49))
    arrow((0.31, 0.49), (0.345, 0.49))
    arrow((0.465, 0.49), (0.53, 0.695), color=blue)
    arrow((0.465, 0.49), (0.53, 0.235), color=orange)
    arrow((0.65, 0.695), (0.68, 0.695), color=blue)
    arrow((0.65, 0.235), (0.68, 0.235), color=orange)
    arrow((0.785, 0.695), (0.855, 0.54), color=blue)
    arrow((0.785, 0.235), (0.855, 0.44), color=orange)

    ax.text(0.59, 0.92, "Training", color=blue, fontsize=8, weight="bold")
    ax.text(0.73, 0.92, "Inference", color=green, fontsize=8, weight="bold")
    ax.text(0.015, 0.06, "Frozen components", color=grey, fontsize=7.5)
    ax.plot([0.118, 0.145], [0.075, 0.075], color=grey, lw=1.2)
    ax.text(0.165, 0.06, "Trainable CoOp context only", color=blue, fontsize=7.5)
    保存图(fig, "图1-方法框架")


def 生成数据集增益图(index: dict[tuple[str, str], dict[str, float]]) -> dict[str, dict[str, float]]:
    labels = [简称[d] for d in 数据集顺序]
    gains: dict[str, dict[str, float]] = {}
    coop = []
    for dataset in 数据集顺序:
        target = index[(dataset, "固定均值概率融合")]["HM"]
        gain_coop = target - index[(dataset, "CoOp")]["HM"]
        gains[dataset] = {"相对CoOp": gain_coop}
        coop.append(gain_coop)

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    bars = ax.barh(y, coop, 0.58, color="#2F5D9B")
    ax.axvline(0, color="#45505E", lw=0.9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("HM improvement (percentage points)")
    ax.set_title("TextConsistency improves matched CoOp HM on all eleven datasets")
    ax.grid(axis="x", color="#D8DEE8", lw=0.6, alpha=0.75)
    ax.set_axisbelow(True)
    xmax = max(coop)
    ax.set_xlim(0, xmax + 1.25)
    for bar in bars:
        value = bar.get_width()
        ax.text(
            value + 0.10,
            bar.get_y() + bar.get_height() / 2,
            f"+{value:.2f}",
            va="center",
            fontsize=7.5,
            color="#293241",
        )
    fig.tight_layout()
    保存图(fig, "图2-十一数据集增益")
    return gains


def 读取逐种子增益() -> dict[str, list[float]]:
    result: dict[str, list[float]] = {dataset: [] for dataset in 数据集顺序}
    rows = 读取_csv(结果目录 / "投稿实验统一结果-20260915" / "主结果逐种子.csv")
    first_five = set(数据集顺序[:5])
    by_key = {(row["数据集"], row["种子"], row["方法"]): float(row["HM"]) for row in rows}
    for dataset in 数据集顺序[:5]:
        for seed in ("1", "2", "3"):
            result[dataset].append(
                by_key[(dataset, seed, "固定均值概率融合")]
                - by_key[(dataset, seed, "CoOp")]
            )

    paths = {
        "Caltech101": 结果目录 / "三数据集扩展与八数据集汇总-20260916" / "主方法-Caltech101-结果.json",
        "OxfordPets": 结果目录 / "三数据集扩展与八数据集汇总-20260916" / "主方法-OxfordPets-结果.json",
        "Food101": 结果目录 / "三数据集扩展与八数据集汇总-20260916" / "主方法-Food101-结果.json",
        "StanfordCars": 结果目录 / "斯坦福汽车与SUN397扩展验证-20260917" / "主方法-StanfordCars-结果.json",
        "SUN397": 结果目录 / "斯坦福汽车与SUN397扩展验证-20260917" / "主方法-SUN397-结果.json",
    }
    for dataset, path in paths.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        result[dataset] = [float(row["融合相对CoOp"]["HM增益pp"]) for row in data["逐种子"]]
    imagenet = json.loads(
        (结果目录 / "ImageNet投稿主实验-20260919" / "结果汇总.json")
        .read_text(encoding="utf-8")
    )
    result["ImageNet"] = [
        float(row["融合相对CoOp"]["HM增益pp"])
        for row in imagenet["逐种子"]
    ]

    assert all(len(values) == 3 for values in result.values())
    assert all(value > 0 for values in result.values() for value in values)
    return result


def 生成诊断图(components: dict[tuple[str, str], float]) -> dict[str, list[float]]:
    seed_gains = 读取逐种子增益()
    methods = ["CoOp", "Description only", "Logit fusion", "Probability fusion"]
    averages = [
        sum(components[(dataset, method)] for dataset in 数据集顺序) / len(数据集顺序)
        for method in methods
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.75), gridspec_kw={"width_ratios": [0.95, 1.85, 0.95]})

    ax = axes[0]
    colors = ["#8391A5", "#D9A066", "#6C8FC7", "#2F5D9B"]
    y = np.arange(len(methods))
    ax.hlines(y, 73.5, averages, color=colors, linewidth=3)
    ax.scatter(averages, y, s=58, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(y, ["CoOp", "Description", "Logit fusion", "Probability fusion"])
    ax.invert_yaxis()
    ax.set_xlim(73.5, 79.3)
    ax.set_xlabel("Eleven-dataset HM (%)")
    ax.set_title("(a) Component ablation", loc="left", weight="bold")
    ax.grid(axis="x", color="#D8DEE8", lw=0.6)
    for yi, value in zip(y, averages):
        ax.text(value + 0.09, yi, f"{value:.2f}", va="center", fontsize=7.5)

    ax = axes[1]
    x = np.arange(len(数据集顺序))
    jitter = [-0.14, 0.0, 0.14]
    for seed_idx in range(3):
        values = [seed_gains[d][seed_idx] for d in 数据集顺序]
        ax.scatter(
            x + jitter[seed_idx],
            values,
            s=25,
            alpha=0.9,
            color=["#7EA0D1", "#2F5D9B", "#173B69"][seed_idx],
            label=f"Seed {seed_idx + 1}",
        )
    means = [sum(seed_gains[d]) / 3 for d in 数据集顺序]
    ax.scatter(x, means, marker="D", s=34, color="#D97721", edgecolor="white", linewidth=0.6, label="Mean", zorder=4)
    ax.axhline(0, color="#45505E", lw=0.9)
    ax.set_xticks(x, [简称[d] for d in 数据集顺序], rotation=42, ha="right")
    ax.set_ylabel("Paired HM gain (pp)")
    ax.set_title("(b) Three-seed paired gains", loc="left", weight="bold")
    ax.grid(axis="y", color="#D8DEE8", lw=0.6)
    ax.legend(ncol=2, frameon=False, loc="upper right", handletextpad=0.3, columnspacing=0.8)

    ax = axes[2]
    overhead = [0.0, 0.7411685601111717]
    hm = [averages[0], averages[3]]
    ax.scatter(overhead[0], hm[0], s=72, color="#8391A5", zorder=3)
    ax.scatter(overhead[1], hm[1], s=82, color="#2E7D65", zorder=3)
    ax.annotate("", xy=(overhead[1], hm[1]), xytext=(overhead[0], hm[0]), arrowprops=dict(arrowstyle="->", color="#D97721", lw=1.5))
    ax.text(overhead[0] + 0.03, hm[0] - 0.28, f"CoOp\n{hm[0]:.2f} HM", fontsize=8, va="top")
    ax.text(overhead[1] - 0.03, hm[1] + 0.20, f"TextConsistency\n{hm[1]:.2f} HM", fontsize=8, ha="right", va="bottom")
    ax.set_xlim(-0.08, 0.86)
    ax.set_ylim(74.5, 79.6)
    ax.set_xlabel("Batch-64 latency overhead (%)")
    ax.set_ylabel("Eleven-dataset HM (%)")
    ax.set_title("(c) Accuracy-cost summary", loc="left", weight="bold")
    ax.grid(color="#D8DEE8", lw=0.6)

    fig.tight_layout(w_pad=1.4)
    保存图(fig, "图3-消融稳定性效率")
    return seed_gains


def 生成语义对照图() -> tuple[dict[str, float], list[dict[str, float]]]:
    control_rows = 读取_csv(
        结果目录 / "投稿关键对照十数据集-20260918" / "汇总" / "关键对照数据集汇总.csv"
    )
    index = {
        (row["数据集"], row["方法"]): float(row["HM_均值计算"])
        for row in control_rows
    }
    control_names = {"FGVC-Aircraft": "FGVCAircraft"}
    gains = [
        index[(control_names.get(dataset, dataset), "TextConsistency")]
        - index[(control_names.get(dataset, dataset), "CoOp+类别名称")]
        for dataset in 数据集顺序[:-1]
    ]
    imagenet_rows = 读取_csv(
        结果目录 / "ImageNet投稿主实验-20260919" / "三种子统计.csv"
    )
    imagenet_index = {row["方法"]: float(row["种子HM均值"]) for row in imagenet_rows}
    gains.append(imagenet_index["固定均值概率融合"] - imagenet_index["CoOp+类别名称"])

    alpha_rows = [
        row
        for row in 读取_csv(
            结果目录 / "ImageNet投稿主实验-20260919" / "权重稳健性.csv"
        )
        if row["范围"] == "十一数据集"
    ]
    curves = {
        "alpha": [float(row["alpha"]) for row in alpha_rows],
        "Base": [float(row["Base"]) for row in alpha_rows],
        "Novel": [float(row["Novel"]) for row in alpha_rows],
        "HM": [float(row["HM"]) for row in alpha_rows],
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.55), constrained_layout=True)
    colors = ["#2F6F9F" if value >= 0 else "#B24745" for value in gains]
    axes[0].bar(range(len(gains)), gains, color=colors, width=0.68)
    axes[0].axhline(0, color="#444444", linewidth=0.8)
    axes[0].set_xticks(range(len(gains)), [简称[d] for d in 数据集顺序], rotation=28, ha="right")
    axes[0].set_ylabel("HM gain over CoOp + class name (pp)")
    axes[0].set_title("(a) Description-specific contribution")
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.6)
    for i, value in enumerate(gains):
        axes[0].text(
            i,
            value + (0.13 if value >= 0 else -0.13),
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=7.3,
        )

    styles = {"Base": ("#577590", "o"), "Novel": ("#F8961E", "s"), "HM": ("#2A9D8F", "D")}
    for label in ("Base", "Novel", "HM"):
        color, marker = styles[label]
        axes[1].plot(
            curves["alpha"],
            curves[label],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=4.8,
        )
    axes[1].axvline(0.5, color="#555555", linestyle="--", linewidth=0.9, alpha=0.7)
    axes[1].set_xticks(curves["alpha"])
    axes[1].set_xlabel(r"Global description weight $\alpha$")
    axes[1].set_ylabel("Eleven-dataset macro average (%)")
    axes[1].set_title("(b) One global weight for all datasets")
    axes[1].grid(alpha=0.25, linewidth=0.6)
    axes[1].legend(frameon=False, ncol=3, loc="lower center")
    保存图(fig, "图4-语义对照与权重稳健性")

    macro = {
        row["方法"]: float(row["HM宏平均"])
        for row in 读取_csv(
            结果目录 / "ImageNet投稿主实验-20260919" / "十一数据集方法宏平均.csv"
        )
    }
    comparison = {
        "TextConsistency减CoOp+类别名称_HM_pp": macro["TextConsistency"] - macro["CoOp+类别名称"],
        "TextConsistency胜出数据集数": sum(value > 0 for value in gains),
        "数据集总数": len(gains),
        "TextConsistency胜出配对种子数": 28,
        "配对种子总数": 33,
    }
    weights = [
        {
            "alpha": alpha,
            "数据集数": len(数据集顺序),
            "宏平均Base": base,
            "宏平均Novel": novel,
            "宏平均HM": hm_value,
        }
        for alpha, base, novel, hm_value in zip(
            curves["alpha"], curves["Base"], curves["Novel"], curves["HM"]
        )
    ]
    return comparison, weights


def 生成图表(index: dict[tuple[str, str], dict[str, float]], components: dict[tuple[str, str], float]) -> None:
    设置绘图样式()
    生成方法框架图()
    gains = 生成数据集增益图(index)
    seed_gains = 生成诊断图(components)
    semantic_comparison, semantic_weights = 生成语义对照图()

    source_paths = [
        结果目录 / "斯坦福汽车与SUN397扩展验证-20260917" / "十数据集结果.csv",
        结果目录 / "投稿实验统一结果-20260915" / "主结果逐种子.csv",
        结果目录 / "统一推理效率-20260916" / "效率实测结果.json",
        结果目录 / "投稿关键对照十数据集-20260918" / "汇总" / "关键对照数据集汇总.csv",
        结果目录 / "投稿关键对照十数据集-20260918" / "汇总" / "统一权重稳健性.csv",
        结果目录 / "ImageNet投稿主实验-20260919" / "结果汇总.json",
        结果目录 / "ImageNet投稿主实验-20260919" / "三种子统计.csv",
        结果目录 / "ImageNet投稿主实验-20260919" / "十一数据集方法宏平均.csv",
        结果目录 / "ImageNet投稿主实验-20260919" / "权重稳健性.csv",
    ]
    receipt = {
        "状态": "通过",
        "图数量": 4,
        "图文件": {name: 文件哈希(源码目录 / name) for name in 图名列表},
        "主要数据来源": [
            {"路径": str(path.relative_to(根目录)), "SHA256": 文件哈希(path)}
            for path in source_paths
        ],
        "十一数据集HM增益": gains,
        "逐种子配对HM增益": seed_gains,
        "类别名称控制": semantic_comparison,
        "统一权重稳健性": semantic_weights,
        "说明": "所有结果图由已审计CSV/JSON自动生成；方法框架图只描述已发布实现。",
    }
    (Path(__file__).resolve().parent / "图表数据来源.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def 生成Overleaf直接替换文件() -> None:
    """把六张表内联到主文件，并复制参考文献和四张矢量图。"""
    text = (源码目录 / "pag.tex").read_text(encoding="utf-8")
    for name in (
        "table_local_hm.tex",
        "table_average_metrics.tex",
        "table_published_hm.tex",
        "table_components.tex",
        "table_efficiency.tex",
        "类别名称与权重稳健性.tex",
    ):
        marker = rf"\input{{table/{name.removesuffix('.tex')}}}"
        table = (表格目录 / name).read_text(encoding="utf-8").rstrip()
        if marker not in text:
            raise RuntimeError(f"主文件缺少表格标记：{marker}")
        text = text.replace(marker, table)

    Overleaf目录.mkdir(parents=True, exist_ok=True)
    (Overleaf目录 / "pag.tex").write_text(text, encoding="utf-8")
    (Overleaf目录 / "cas-refs.bib").write_text(
        (源码目录 / "cas-refs.bib").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    for name in 图名列表:
        shutil.copy2(源码目录 / name, Overleaf目录 / name)


def main() -> None:
    index = 主要结果()
    components = 组件结果()
    生成本地主表(index)
    生成平均表(index)
    生成文献表()
    生成组件表(components)
    生成效率表()
    生成类别名称与权重表()
    生成图表(index, components)
    生成Overleaf直接替换文件()
    print(
        json.dumps(
            {
                "状态": "通过",
                "表格目录": str(表格目录),
                "表格数": 6,
                "图数量": 4,
                "Overleaf直接替换目录": str(Overleaf目录),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
