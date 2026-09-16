#!/usr/bin/env python3
"""用论文原表数值生成八数据集文献定位表。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


根目录 = Path(__file__).resolve().parent
数据集 = [
    "UCF101",
    "DTD",
    "FGVC-Aircraft",
    "EuroSAT",
    "OxfordFlowers",
    "Caltech101",
    "OxfordPets",
    "Food101",
]

提示源论文 = "https://arxiv.org/abs/2307.06948"
文本监督论文 = "https://ojs.aaai.org/index.php/AAAI/article/download/32444/34599"
榜单地址 = "https://github.com/zhengli97/Awesome-Prompt-Adapter-Learning-for-VLMs-CLIP"

# HM 数值来自对应论文的 Base-to-Novel 表，而不是本仓库复跑。
外部结果 = [
    {
        "方法": "CuPL",
        "会议": "ICCV 2023",
        "来源": "ProText（AAAI 2025）Table 3 对 CuPL 的实现",
        "url": 文本监督论文,
        "HM": [76.40, 61.30, 33.20, 61.80, 76.10, 95.60, 95.80, 90.60],
        "备注": "Base使用LLM描述；Novel沿用zero-shot CLIP。",
    },
    {
        "方法": "CoCoOp（论文报告）",
        "会议": "CVPR 2022",
        "来源": "PromptSRC（ICCV 2023）Table 1",
        "url": 提示源论文,
        "HM": [77.64, 64.85, 27.74, 71.21, 81.71, 95.84, 96.43, 90.99],
        "备注": "论文汇总值，非本仓库CoCoOp复跑值。",
    },
    {
        "方法": "KgCoOp",
        "会议": "CVPR 2023",
        "来源": "PromptSRC（ICCV 2023）Table 1",
        "url": 提示源论文,
        "HM": [79.65, 64.35, 34.83, 73.48, 83.65, 96.03, 96.18, 91.09],
        "备注": "论文报告值。",
    },
    {
        "方法": "MaPLe",
        "会议": "CVPR 2023",
        "来源": "PromptSRC（ICCV 2023）Table 1",
        "url": 提示源论文,
        "HM": [80.77, 68.16, 36.50, 82.35, 82.56, 96.02, 96.58, 91.38],
        "备注": "论文报告值。",
    },
    {
        "方法": "PromptSRC",
        "会议": "ICCV 2023",
        "来源": "PromptSRC（ICCV 2023）Table 1",
        "url": 提示源论文,
        "HM": [82.74, 71.75, 40.15, 82.32, 85.95, 96.02, 96.30, 91.10],
        "备注": "论文报告值。",
    },
]


def 写_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    audit = json.loads((根目录 / "独立复核结果.json").read_text(encoding="utf-8"))
    own_map = {
        row["数据集"]: row["HM"]
        for row in audit["八数据集逐数据集结果"]
        if row["方法"] == "固定均值概率融合"
    }
    assert set(own_map) == set(数据集)
    methods = []
    for item in 外部结果:
        assert len(item["HM"]) == len(数据集)
        methods.append({**item, "八数据集等权HM": mean(item["HM"])})
    own_values = [own_map[name] for name in 数据集]
    own_mean = mean(own_values)
    methods.append({
        "方法": "本文方法",
        "会议": "本仓库实跑",
        "来源": "三数据集扩展与八数据集独立复核",
        "url": "独立复核结果.json",
        "HM": own_values,
        "八数据集等权HM": own_mean,
        "备注": "ViT-B/16、16-shot、seed 1/2/3；固定0.5概率融合。",
    })

    delta = {
        item["方法"]: own_mean - item["八数据集等权HM"]
        for item in methods
        if item["方法"] != "本文方法"
    }
    promptsrc = next(item for item in methods if item["方法"] == "PromptSRC")
    vs_promptsrc = [
        {
            "数据集": name,
            "本文HM": own_values[index],
            "PromptSRC HM": promptsrc["HM"][index],
            "差值pp": own_values[index] - promptsrc["HM"][index],
        }
        for index, name in enumerate(数据集)
    ]
    result = {
        "状态": "通过",
        "用途": "八数据集论文定位；论文报告值与本地实跑值严格分开",
        "数据集顺序": 数据集,
        "结果": methods,
        "本文方法相对论文平均HM差值": delta,
        "本文方法相对PromptSRC逐数据集": vs_promptsrc,
        "当前Awesome榜单背景": {
            "口径": "11数据集平均HM，不能与本文八数据集平均直接比较",
            "CoCoOp": 75.83,
            "KgCoOp": 77.00,
            "MaPLe": 78.55,
            "PromptSRC": 79.97,
            "PromptKD": 83.73,
            "来源": 榜单地址,
        },
        "结论": "本文八数据集平均HM高于CuPL、论文CoCoOp和KgCoOp，与MaPLe基本持平，低于PromptSRC 1.46pp；当前方向已有PromptKD等更强方法，不能宣称SOTA。",
        "限制": [
            "外部方法为论文原表或论文中的基线复现值，不是本仓库同环境复跑。",
            "CuPL数值来自ProText对CuPL的Base-to-Novel实现，Novel沿用zero-shot CLIP。",
            "八数据集是11数据集标准集合的子集，不能把八数据集平均与Awesome榜单11数据集平均直接排序。",
        ],
    }
    写_json(根目录 / "八数据集文献方法对比.json", result)

    csv_path = 根目录 / "八数据集文献方法对比.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["方法", *数据集, "八数据集等权HM", "本文相对该方法pp", "来源", "url", "备注"]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in methods:
            row = {"方法": item["方法"]}
            row.update(dict(zip(数据集, item["HM"])))
            row.update({
                "八数据集等权HM": item["八数据集等权HM"],
                "本文相对该方法pp": 0.0 if item["方法"] == "本文方法" else delta[item["方法"]],
                "来源": item["来源"],
                "url": item["url"],
                "备注": item["备注"],
            })
            writer.writerow(row)

    md_lines = [
        "# 八数据集论文方法对比",
        "",
        "外部方法均为论文报告值；本文方法为本仓库独立复核的本地实跑值。表内只比较八个共同数据集的 HM。",
        "",
        "| 方法 | UCF101 | DTD | Aircraft | EuroSAT | Flowers | Caltech101 | OxfordPets | Food101 | 八数据集平均 | 本文差值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in methods:
        values = " | ".join(f"{value:.2f}" for value in item["HM"])
        own_delta = 0.0 if item["方法"] == "本文方法" else delta[item["方法"]]
        label = f"**{item['方法']}**" if item["方法"] == "本文方法" else item["方法"]
        md_lines.append(f"| {label} | {values} | **{item['八数据集等权HM']:.2f}** | {own_delta:+.2f} |")
    md_lines.extend([
        "",
        "## 结论",
        "",
        f"本文八数据集平均 HM 为 **{own_mean:.2f}**，高于 CuPL **{delta['CuPL']:+.2f}pp**、论文 CoCoOp **{delta['CoCoOp（论文报告）']:+.2f}pp**、KgCoOp **{delta['KgCoOp']:+.2f}pp**，与 MaPLe 基本持平（**{delta['MaPLe']:+.2f}pp**），低于 PromptSRC **{-delta['PromptSRC']:.2f}pp**。本文在 Caltech101、OxfordPets、Food101 上分别比 PromptSRC 高 0.07、0.33、0.20pp，但原五数据集均未超过 PromptSRC。",
        "",
        "这能支撑的表述是：固定概率融合以零新增可训练参数达到 MaPLe 水平，并缩小与 PromptSRC 的差距。不能写成达到或超过 PromptSRC，也不能写成当前 SOTA。",
        "",
        "## 与当前 Awesome 榜单的关系",
        "",
        "Awesome 仓库当前列出的11数据集平均 HM 包括 CoCoOp 75.83、KgCoOp 77.00、MaPLe 78.55、PromptSRC 79.97 和 PromptKD 83.73。本文还缺 ImageNet、StanfordCars、SUN397，因此 **79.33 的八数据集平均不能与这些11数据集平均直接比较**。PromptKD 等更新方法也说明本文不处于当前准确率最高位置。",
        "",
        "## 来源与限制",
        "",
        f"- CoCoOp、KgCoOp、MaPLe、PromptSRC：PromptSRC ICCV 2023 Table 1，{提示源论文}",
        f"- CuPL：ProText AAAI 2025 Table 3 对 CuPL 的实现，{文本监督论文}",
        f"- 当前方向榜单：{榜单地址}",
        "- 外部数字不是本仓库同环境复跑；论文中必须用 † 标明，并与本地受控对比表分开。",
    ])
    (根目录 / "八数据集文献方法对比.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comparison with published base-to-novel results on the eight datasets shared by all methods. Published values are marked with $\dagger$ and are not local reproductions.}",
        r"\label{tab:eight-dataset-literature}",
        r"\small",
        r"\setlength{\tabcolsep}{3.0pt}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Method & UCF & DTD & Air. & Euro. & Flw. & Cal. & Pets & Food & Avg. \\",
        r"\midrule",
    ]
    for item in methods:
        suffix = "" if item["方法"] == "本文方法" else r"$^\dagger$"
        label = "Ours" if item["方法"] == "本文方法" else item["方法"].replace("（论文报告）", "")
        values = " & ".join(f"{value:.2f}" for value in item["HM"])
        avg = f"\\textbf{{{item['八数据集等权HM']:.2f}}}" if item["方法"] == "本文方法" else f"{item['八数据集等权HM']:.2f}"
        tex_lines.append(f"{label}{suffix} & {values} & {avg} \\\\")
    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    (根目录 / "论文八数据集文献对比表.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    print(json.dumps({"状态": "通过", "本文八数据集HM": own_mean, "相对PromptSRC": delta["PromptSRC"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
