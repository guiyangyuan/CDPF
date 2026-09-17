#!/usr/bin/env python3
"""根据独立复核与文献对比结果生成中文实验报告和论文主表。"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


根目录 = Path(__file__).resolve().parent
新增数据集 = ["StanfordCars", "SUN397"]
方法顺序 = ["Zero-shot CLIP", "CoOp", "CoCoOp", "固定均值概率融合"]


def 读取(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    audit = 读取(根目录 / "独立复核结果.json")
    literature = 读取(根目录 / "十数据集文献方法对比.json")
    rows = audit["十数据集逐数据集结果"]
    by_key = {(row["数据集"], row["方法"]): row for row in rows}
    summaries = {(row["范围"], row["方法"]): row for row in audit["范围汇总"]}
    deltas = literature["本文方法相对论文平均HM差值"]
    comparisons = {row["数据集"]: row for row in audit["主方法相对CoCoOp"]}
    paired = audit["主方法新增两数据集逐种子比较"]
    positive_seeds = sum(row["HM增益pp"] > 0 for row in paired)
    positive_datasets = sum(row["主方法相对CoCoOp HM增益pp"] > 0 for row in audit["主方法相对CoCoOp"])

    lines = [
        "# StanfordCars与SUN397扩展验证报告",
        "",
        "## 结论",
        "",
    ]
    new_main = summaries[("新增两数据集", "固定均值概率融合")]["HM等权均值"]
    new_cocoop = summaries[("新增两数据集", "CoCoOp")]["HM等权均值"]
    ten_main = summaries[("全部十数据集", "固定均值概率融合")]["HM等权均值"]
    ten_cocoop = summaries[("全部十数据集", "CoCoOp")]["HM等权均值"]
    ten_coop = summaries[("全部十数据集", "CoOp")]["HM等权均值"]
    ten_zero = summaries[("全部十数据集", "Zero-shot CLIP")]["HM等权均值"]
    lines.extend([
        f"StanfordCars、SUN397 的 Zero-shot CLIP、主方法和本地 CoCoOp 均已执行完成并通过独立复核。固定均值概率融合在新增两数据集上的等权平均 HM 为 **{new_main:.2f}**，本地 CoCoOp 为 **{new_cocoop:.2f}**，差值为 **{new_main-new_cocoop:+.2f}pp**；相对同批 CoOp 的 6 个种子中有 **{positive_seeds}/6** 个为正增益。",
        "",
        f"合并已有八数据集后，主方法十数据集等权平均 HM 为 **{ten_main:.2f}**，高于本地 CoCoOp **{ten_cocoop:.2f}**（{ten_main-ten_cocoop:+.2f}pp）、同批 CoOp **{ten_coop:.2f}**（{ten_main-ten_coop:+.2f}pp）和 Zero-shot CLIP **{ten_zero:.2f}**（{ten_main-ten_zero:+.2f}pp）。主方法在 **{positive_datasets}/10** 个数据集上高于本地 CoCoOp。",
        "",
        f"按十个共同数据集重算论文报告值，本文高于 CuPL **{deltas['CuPL']:+.2f}pp**、论文 CoCoOp **{deltas['CoCoOp（论文报告）']:+.2f}pp** 和 KgCoOp **{deltas['KgCoOp']:+.2f}pp**，与 MaPLe 差值 **{deltas['MaPLe']:+.2f}pp**，低于 PromptSRC **{-deltas['PromptSRC']:.2f}pp**。当前证据适合支撑低成本、稳定改善和接近 MaPLe 的普通期刊主张，不能写成达到 PromptSRC 或 SOTA。",
        "",
        "## 新增两数据集结果",
        "",
        "表中 HM 由三个种子的 Base/Novel 均值计算；Zero-shot CLIP 为冻结模型的一次 Base/Novel 评估。",
        "",
        "| 数据集 | 方法 | Base | Novel | HM | 相对本地CoCoOp |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for dataset in 新增数据集:
        cocoop_hm = by_key[(dataset, "CoCoOp")]["HM"]
        for method in 方法顺序:
            row = by_key[(dataset, method)]
            delta = row["HM"] - cocoop_hm
            label = f"**{method}**" if method == "固定均值概率融合" else method
            delta_text = "—" if method == "CoCoOp" else f"{delta:+.2f}"
            lines.append(f"| {dataset} | {label} | {row['Base']:.2f} | {row['Novel']:.2f} | {row['HM']:.2f} | {delta_text} |")

    lines.extend(["", "主方法相对同批 CoOp 的逐种子 HM 增益为：", ""])
    for dataset in 新增数据集:
        values = [row["HM增益pp"] for row in paired if row["数据集"] == dataset]
        lines.append(f"- {dataset}：" + "、".join(f"{value:+.3f}pp" for value in values) + "。")

    lines.extend([
        "",
        "## 十数据集总体结果",
        "",
        "| 数据集 | Zero-shot HM | CoOp HM | CoCoOp HM | 固定均值概率融合 HM | 融合相对CoCoOp |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    dataset_order = []
    for row in rows:
        if row["数据集"] not in dataset_order:
            dataset_order.append(row["数据集"])
    for dataset in dataset_order:
        values = {method: by_key[(dataset, method)]["HM"] for method in 方法顺序}
        lines.append(
            f"| {dataset} | {values['Zero-shot CLIP']:.2f} | {values['CoOp']:.2f} | {values['CoCoOp']:.2f} | "
            f"**{values['固定均值概率融合']:.2f}** | {values['固定均值概率融合']-values['CoCoOp']:+.2f} |"
        )
    lines.append(f"| **十数据集等权平均** | **{ten_zero:.2f}** | **{ten_coop:.2f}** | **{ten_cocoop:.2f}** | **{ten_main:.2f}** | **{ten_main-ten_cocoop:+.2f}** |")

    lines.extend([
        "",
        "## 执行、恢复与复核",
        "",
        "- Zero-shot CLIP：4/4 条评估命令成功，从原始日志的 correct/total 复算。",
        "- 主方法：2/2 个数据集、6 次训练完成，从 predictions.jsonl 逐样本复算四种预测；冻结源码在执行前后保持一致。",
        "- CoCoOp：6 次训练与 12 次 Base/Novel 独立重载全部成功，6 个检查点及 12 份评估日志的 SHA-256 与汇总一致。",
        "- CoCoOp 第一个检查点首次重载遇到 PyTorch 2.6 默认 weights_only 兼容错误。恢复前用同一检查点独立复现并验证官方 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 开关，随后只恢复失败评估和后续命令；没有修改源码、配置、数据、种子或结果。首次失败日志与恢复记录均保留在完整归档中。",
        f"- 完整归档大小 {audit['归档']['字节数']:,} 字节，SHA256 `{audit['归档']['SHA256']}`。",
        "- 完整归档下载、独立复核和 GitHub 结果推送完成后，AutoDL 实例 `a5904787c3-600682b1` 已由页面确认“已关机”。",
        "",
        "## 协议与边界",
        "",
        "- 所有本地方法使用 CLIP ViT-B/16、相同 Zhou 固定划分、16-shot 和 seed 1/2/3。",
        "- 主方法是 CoOp 学习提示与冻结均值描述分支的固定 0.5/0.5 概率融合，不按数据集搜索权重；相对 CoOp 不增加可训练参数。",
        "- 主方法沿用 20 epochs 配方；CoCoOp 使用其标准 10 epochs、batch size 1 配方。两者不是完全相同训练预算的消融。",
        "- 外部论文数字不是本仓库同环境复跑，应在论文中用 † 标记并与本地受控对比表分开。",
        "- 当前仍缺 ImageNet，十数据集平均不能直接与 Awesome 榜单的十一数据集平均排名。",
        "",
        "## 投稿判断",
        "",
        "这两组新实验补齐了除 ImageNet 外的标准 Base-to-Novel 数据集，并继续支持方法对新类别泛化的作用。StanfordCars 的提升幅度较小，应作为限制如实报告；SUN397 的收益更明显。现在优先级应转向论文主表、消融、效率和局限性叙事，不再继续堆叠未经验证的新模块。",
        "",
        "机器可读证据见[独立复核结果](独立复核结果.json)、[新增两数据集结果](新增两数据集结果.csv)、[十数据集结果](十数据集结果.csv)、[十数据集文献对比](十数据集文献方法对比.json)和[完整实验归档](完整实验归档.tar.gz)。",
    ])
    (根目录 / "结果报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Base-to-novel generalization on ten datasets with CLIP ViT-B/16 and 16 shots.}",
        r"\label{tab:ten-dataset-main}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrrrrrrr}",
        r"\toprule",
        r"Method & UCF & DTD & Air. & Euro. & Flw. & Cal. & Pets & Food & Cars & SUN & Avg. \\",
        r"\midrule",
    ]
    for method in 方法顺序:
        values = [by_key[(dataset, method)]["HM"] for dataset in dataset_order]
        average = mean(values)
        label = "Ours" if method == "固定均值概率融合" else method
        cells = " & ".join(f"{value:.2f}" for value in values)
        average_cell = rf"\textbf{{{average:.2f}}}" if method == "固定均值概率融合" else f"{average:.2f}"
        tex.append(f"{label} & {cells} & {average_cell} \\\\")
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (根目录 / "论文十数据集主表.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(json.dumps({"状态": "通过", "十数据集主方法HM": ten_main}, ensure_ascii=False))


if __name__ == "__main__":
    main()
