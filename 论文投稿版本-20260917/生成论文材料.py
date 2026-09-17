#!/usr/bin/env python3
"""从已审计结果生成论文中的 LaTeX 表格。"""

from __future__ import annotations

import csv
import json
from pathlib import Path


根目录 = Path(__file__).resolve().parents[1]
源码目录 = Path(__file__).resolve().parent / "修改后源码"
表格目录 = 源码目录 / "table"
Overleaf目录 = Path(__file__).resolve().parent / "Overleaf直接替换"
结果目录 = 根目录 / "reports" / "experiments"

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
}


def 读取_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def 保存(name: str, text: str) -> None:
    表格目录.mkdir(parents=True, exist_ok=True)
    (表格目录 / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def 数字(value: float) -> str:
    return f"{value:.2f}"


def 主要结果() -> dict[tuple[str, str], dict[str, float]]:
    rows = 读取_csv(
        结果目录 / "斯坦福汽车与SUN397扩展验证-20260917" / "十数据集结果.csv"
    )
    return {
        (row["数据集"], row["方法"]): {
            "Base": float(row["Base"]),
            "Novel": float(row["Novel"]),
            "HM": float(row["HM"]),
        }
        for row in rows
    }


def 生成本地主表(index: dict[tuple[str, str], dict[str, float]]) -> None:
    methods = ["Zero-shot CLIP", "CoOp", "CoCoOp", "固定均值概率融合"]
    labels = {
        "Zero-shot CLIP": "Zero-shot CLIP",
        "CoOp": "CoOp",
        "CoCoOp": "CoCoOp",
        "固定均值概率融合": r"\textbf{TextConsistency}",
    }
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Locally evaluated harmonic-mean accuracy (\%) on ten base-to-novel benchmarks. All methods use CLIP ViT-B/16 and the official class splits. CoOp, CoCoOp, and TextConsistency are averaged over three seeds.}",
        r"\label{tab:local_hm}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrrr}",
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
    methods = ["Zero-shot CLIP", "CoOp", "CoCoOp", "固定均值概率融合"]
    labels = {
        "Zero-shot CLIP": "Zero-shot CLIP",
        "CoOp": "CoOp",
        "CoCoOp": "CoCoOp",
        "固定均值概率融合": r"\textbf{TextConsistency}",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ten-dataset macro averages (\%). HM is computed for each dataset before macro averaging.}",
        r"\label{tab:average_metrics}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Base & Novel & HM \\",
        r"\midrule",
    ]
    for method in methods:
        values = {
            metric: sum(index[(dataset, method)][metric] for dataset in 数据集顺序) / 10
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
        "Method & " + " & ".join(简称[d] for d in 数据集顺序) + r" & Avg. \\",
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
        values = [float(row[d]) for d in 数据集顺序]
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
        r"\begin{tabular}{lrrrrrrrrrrr}",
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


def 生成Overleaf直接替换文件() -> None:
    """把五张表内联到主文件，便于直接覆盖现有 Overleaf 项目的两个文件。"""
    text = (源码目录 / "pag.tex").read_text(encoding="utf-8")
    for name in (
        "table_local_hm.tex",
        "table_average_metrics.tex",
        "table_published_hm.tex",
        "table_components.tex",
        "table_efficiency.tex",
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


def main() -> None:
    index = 主要结果()
    生成本地主表(index)
    生成平均表(index)
    生成文献表()
    生成组件表(组件结果())
    生成效率表()
    生成Overleaf直接替换文件()
    print(
        json.dumps(
            {
                "状态": "通过",
                "表格目录": str(表格目录),
                "表格数": 5,
                "Overleaf直接替换目录": str(Overleaf目录),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
