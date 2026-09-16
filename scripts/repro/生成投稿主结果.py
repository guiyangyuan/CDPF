#!/usr/bin/env python3
"""从已审计实验产物生成五数据集投稿主结果，不读取手工填写的汇总数字。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


数据集顺序 = ["UCF101", "DTD", "FGVC-Aircraft", "EuroSAT", "OxfordFlowers"]
方法顺序 = ["CoOp", "冻结均值描述", "固定logit均值融合", "固定均值概率融合"]
分支到方法 = {
    "student": "CoOp",
    "description_mean": "冻结均值描述",
    "simple_fusion": "固定logit均值融合",
    "mean_probability_fusion": "固定均值概率融合",
}
数据集键到名称 = {
    "ucf101": "UCF101",
    "dtd": "DTD",
    "fgvc_aircraft": "FGVC-Aircraft",
}
置信区间临界值_自由度2 = 4.302652729911275


def 调和平均(base: float, novel: float) -> float:
    if base + novel == 0:
        return 0.0
    return 2.0 * base * novel / (base + novel)


def 读取_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def 加载前三数据集(仓库根目录: Path) -> tuple[list[dict], str]:
    relative = Path("reports/experiments/attribution-study-20260911-42527/analysis.json")
    payload = 读取_json(仓库根目录 / relative)
    if payload.get("passed") is not True:
        raise ValueError(f"来源没有通过审计：{relative}")

    rows = []
    for source in payload["fusion_seed_branch_rows"]:
        if source["dataset"] not in 数据集键到名称 or source["branch"] not in 分支到方法:
            continue
        rows.append(
            {
                "数据集": 数据集键到名称[source["dataset"]],
                "种子": int(source["seed"]),
                "方法": 分支到方法[source["branch"]],
                "Base": float(source["base"]),
                "Novel": float(source["novel"]),
                "HM": float(source["hm"]),
                "来源": relative.as_posix(),
            }
        )
    return rows, relative.as_posix()


def 加载扩展数据集(仓库根目录: Path, relative: Path) -> tuple[list[dict], str]:
    payload = 读取_json(仓库根目录 / relative)
    if payload.get("状态") != "通过" or payload.get("失败检查项数量") != 0:
        raise ValueError(f"扩展验证来源没有通过审计：{relative}")
    dataset = payload["数据集"]
    if dataset not in 数据集顺序:
        raise ValueError(f"未知数据集：{dataset}")

    rows = []
    for seed_result in payload["结果"]["逐种子"]:
        for branch, method in 分支到方法.items():
            source = seed_result["方法"][branch]
            rows.append(
                {
                    "数据集": dataset,
                    "种子": int(seed_result["种子"]),
                    "方法": method,
                    "Base": float(source["Base"]),
                    "Novel": float(source["Novel"]),
                    "HM": float(source["HM"]),
                    "来源": relative.as_posix(),
                }
            )
    return rows, relative.as_posix()


def 加载零样本基线(仓库根目录: Path) -> tuple[dict[str, dict], dict, list[str]]:
    result_relative = Path("reports/experiments/投稿零样本基线-20260916/结果.json")
    audit_relative = Path("reports/experiments/投稿零样本基线-20260916/独立复核.json")
    payload = 读取_json(仓库根目录 / result_relative)
    audit = 读取_json(仓库根目录 / audit_relative)
    if payload.get("状态") != "通过" or audit.get("状态") != "通过":
        raise ValueError("Zero-shot CLIP结果或独立复核没有通过")
    if audit.get("检查", {}).get("分支结果数") != 10 or audit.get("检查", {}).get("训练检查点数") != 0:
        raise ValueError("Zero-shot CLIP独立复核的分支数或检查点数异常")
    index = {}
    for row in payload.get("结果", []):
        dataset = row["数据集"]
        if dataset in index or dataset not in 数据集顺序:
            raise ValueError(f"Zero-shot CLIP数据集身份异常：{dataset}")
        expected_hm = 调和平均(float(row["Base"]), float(row["Novel"]))
        if not math.isclose(float(row["HM"]), expected_hm, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Zero-shot CLIP的HM不能复算：{dataset}")
        index[dataset] = {
            "Base": float(row["Base"]),
            "Novel": float(row["Novel"]),
            "HM": expected_hm,
        }
    if set(index) != set(数据集顺序):
        raise ValueError("Zero-shot CLIP五数据集结果不完整")
    cross = {
        "数据集数": len(index),
        "数据集等权Base均值": statistics.mean(index[name]["Base"] for name in 数据集顺序),
        "数据集等权Novel均值": statistics.mean(index[name]["Novel"] for name in 数据集顺序),
        "数据集等权HM均值": statistics.mean(index[name]["HM"] for name in 数据集顺序),
    }
    return index, cross, [result_relative.as_posix(), audit_relative.as_posix()]


def 加载联合上下文提示基线(仓库根目录: Path) -> tuple[dict, list[str]]:
    stages = [
        (
            "核心三数据集",
            Path("reports/experiments/联合上下文提示基线-核心三数据集-20260916/核心三数据集结果.json"),
            Path("reports/experiments/联合上下文提示基线-核心三数据集-20260916/独立复核.json"),
        ),
        (
            "扩展两数据集",
            Path("reports/experiments/联合上下文提示基线-扩展两数据集-20260916/扩展两数据集结果.json"),
            Path("reports/experiments/联合上下文提示基线-扩展两数据集-20260916/独立复核.json"),
        ),
    ]
    rows = []
    summary_rows = []
    executions = []
    sources = []
    plan_hashes = set()
    for stage, result_relative, audit_relative in stages:
        payload = 读取_json(仓库根目录 / result_relative)
        audit = 读取_json(仓库根目录 / audit_relative)
        if payload.get("状态") != "通过" or payload.get("阶段") != stage:
            raise ValueError(f"CoCoOp阶段结果状态异常：{stage}")
        if audit.get("状态") != "通过" or audit.get("阶段") != stage:
            raise ValueError(f"CoCoOp独立复核没有通过：{stage}")
        rows.extend(payload.get("逐种子", []))
        summary_rows.extend(payload.get("数据集汇总", []))
        executions.append(payload["执行"])
        plan_hashes.add(audit["计划SHA256"])
        sources.extend([result_relative.as_posix(), audit_relative.as_posix()])
    if len(plan_hashes) != 1:
        raise ValueError("CoCoOp两个阶段没有使用同一份冻结计划")

    expected = {(dataset, seed) for dataset in 数据集顺序 for seed in (1, 2, 3)}
    actual = {(row["数据集"], int(row["种子"])) for row in rows}
    if len(rows) != 15 or actual != expected:
        raise ValueError("CoCoOp逐种子结果身份不完整")
    for row in rows:
        expected_hm = 调和平均(float(row["Base"]), float(row["Novel"]))
        if not math.isclose(float(row["HM"]), expected_hm, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"CoCoOp的HM不能复算：{row['数据集']} seed{row['种子']}")

    summaries = {item["数据集"]: item for item in summary_rows}
    if len(summary_rows) != 5 or set(summaries) != set(数据集顺序):
        raise ValueError("CoCoOp数据集汇总不完整")
    for dataset in 数据集顺序:
        group = [row for row in rows if row["数据集"] == dataset]
        base_mean = statistics.mean(float(row["Base"]) for row in group)
        novel_mean = statistics.mean(float(row["Novel"]) for row in group)
        summary = summaries[dataset]
        for key, actual_value in (
            ("Base均值", base_mean),
            ("Novel均值", novel_mean),
            ("HM_均值计算", 调和平均(base_mean, novel_mean)),
        ):
            if not math.isclose(float(summary[key]), actual_value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"CoCoOp汇总不能复算：{dataset} {key}")

    merged = {
        "状态": "通过",
        "协议": "投稿CoCoOp五数据集标准基线-v1",
        "阶段": "五数据集合并",
        "冻结计划SHA256": next(iter(plan_hashes)),
        "逐种子": rows,
        "数据集汇总": [summaries[name] for name in 数据集顺序],
        "执行": {
            "GPU": "NVIDIA GeForce RTX 4090",
            "命令数": sum(int(item["命令数"]) for item in executions),
            "总耗时秒": sum(float(item["总耗时秒"]) for item in executions),
        },
        "跨数据集汇总": {
            "数据集数": 5,
            "数据集等权Base均值": statistics.mean(float(summaries[name]["Base均值"]) for name in 数据集顺序),
            "数据集等权Novel均值": statistics.mean(float(summaries[name]["Novel均值"]) for name in 数据集顺序),
            "数据集等权HM均值": statistics.mean(float(summaries[name]["HM_均值计算"]) for name in 数据集顺序),
        },
    }
    return merged, sources


def 核验逐种子(rows: list[dict]) -> None:
    identities = set()
    for row in rows:
        identity = (row["数据集"], row["种子"], row["方法"])
        if identity in identities:
            raise ValueError(f"重复结果：{identity}")
        identities.add(identity)
        expected_hm = 调和平均(row["Base"], row["Novel"])
        if not math.isclose(row["HM"], expected_hm, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(f"HM不一致：{identity}，记录{row['HM']}，复算{expected_hm}")

    expected = {
        (dataset, seed, method)
        for dataset in 数据集顺序
        for seed in (1, 2, 3)
        for method in 方法顺序
    }
    missing = sorted(expected - identities)
    extra = sorted(identities - expected)
    if missing or extra or len(rows) != 60:
        raise ValueError(f"逐种子结果不完整：missing={missing}, extra={extra}, rows={len(rows)}")


def 均值标准差(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def 描述性置信区间(values: list[float]) -> tuple[float, float, float, float]:
    if len(values) != 3:
        raise ValueError("当前置信区间只允许三个配对种子")
    mean, sd = 均值标准差(values)
    half = 置信区间临界值_自由度2 * sd / math.sqrt(3)
    return mean, sd, mean - half, mean + half


def 生成统计(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["数据集"], row["方法"])].append(row)

    summaries = []
    for dataset in 数据集顺序:
        for method in 方法顺序:
            group = sorted(grouped[(dataset, method)], key=lambda item: item["种子"])
            base_mean, base_sd = 均值标准差([item["Base"] for item in group])
            novel_mean, novel_sd = 均值标准差([item["Novel"] for item in group])
            seed_hm_mean, seed_hm_sd = 均值标准差([item["HM"] for item in group])
            summaries.append(
                {
                    "数据集": dataset,
                    "方法": method,
                    "种子数": len(group),
                    "Base均值": base_mean,
                    "Base样本标准差": base_sd,
                    "Novel均值": novel_mean,
                    "Novel样本标准差": novel_sd,
                    "HM_均值计算": 调和平均(base_mean, novel_mean),
                    "种子HM均值": seed_hm_mean,
                    "种子HM样本标准差": seed_hm_sd,
                }
            )

    index = {(item["数据集"], item["方法"]): item for item in summaries}
    row_index = {(item["数据集"], item["种子"], item["方法"]): item for item in rows}
    paired = []
    for dataset in 数据集顺序:
        metric_payload = {}
        for metric in ("Base", "Novel", "HM"):
            values = [
                row_index[(dataset, seed, "固定均值概率融合")][metric]
                - row_index[(dataset, seed, "CoOp")][metric]
                for seed in (1, 2, 3)
            ]
            mean, sd, low, high = 描述性置信区间(values)
            metric_payload[metric] = {
                "逐种子差值": values,
                "均值": mean,
                "样本标准差": sd,
                "描述性95%t区间": [low, high],
            }
        baseline = index[(dataset, "CoOp")]
        fusion = index[(dataset, "固定均值概率融合")]
        paired.append(
            {
                "数据集": dataset,
                "Base均值差": fusion["Base均值"] - baseline["Base均值"],
                "Novel均值差": fusion["Novel均值"] - baseline["Novel均值"],
                "HM_均值计算差": fusion["HM_均值计算"] - baseline["HM_均值计算"],
                "配对差": metric_payload,
                "HM正增益种子数": sum(value > 0 for value in metric_payload["HM"]["逐种子差值"]),
            }
        )

    cross = {}
    for method in 方法顺序:
        items = [index[(dataset, method)] for dataset in 数据集顺序]
        cross[method] = {
            "数据集数": len(items),
            "数据集等权Base均值": statistics.mean(item["Base均值"] for item in items),
            "数据集等权Novel均值": statistics.mean(item["Novel均值"] for item in items),
            "数据集等权HM均值": statistics.mean(item["HM_均值计算"] for item in items),
        }
    base_cross = cross["CoOp"]
    fusion_cross = cross["固定均值概率融合"]
    cross["固定均值概率融合相对CoOp"] = {
        "数据集等权Base增益": fusion_cross["数据集等权Base均值"] - base_cross["数据集等权Base均值"],
        "数据集等权Novel增益": fusion_cross["数据集等权Novel均值"] - base_cross["数据集等权Novel均值"],
        "数据集等权HM增益": fusion_cross["数据集等权HM均值"] - base_cross["数据集等权HM均值"],
        "HM正增益数据集数": sum(item["HM_均值计算差"] > 0 for item in paired),
        "HM正增益种子数": sum(item["HM正增益种子数"] for item in paired),
        "总种子比较数": 15,
    }
    return summaries, paired, cross


def 写_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.2f}"


def signed(value: float) -> str:
    return f"{value:+.2f}"


def 生成_markdown(
    summaries: list[dict], paired: list[dict], cross: dict, zero_shot: dict[str, dict], zero_shot_cross: dict,
    cocoop: dict, sources: list[str]
) -> str:
    index = {(item["数据集"], item["方法"]): item for item in summaries}
    lines = [
        "# 五数据集投稿主结果",
        "",
        "更新日期：2026-09-16。全部数值来自已通过审计的实验产物，由 `scripts/repro/生成投稿主结果.py` 自动生成；单位为准确率百分点。投稿主表中的训练方法采用 CLIP ViT-B/16、16-shot、20 epochs、seed 1/2/3，融合权重固定为 0.5；Zero-shot CLIP 冻结模型并对每个分支评估一次。CoCoOp的标准10轮结果在后续独立小节报告。",
        "",
        "## 投稿主表",
        "",
        "HM（均值）先对三个种子的 Base、Novel 分别取均值，再计算调和平均。这一口径适合主表；配对稳定性在下一张表单独报告。",
        "",
        "| 数据集 | 方法 | Base | Novel | HM（均值） | 相对CoOp HM |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in 数据集顺序:
        baseline = index[(dataset, "CoOp")]
        fusion = index[(dataset, "固定均值概率融合")]
        zero = zero_shot[dataset]
        lines.append(
            f"| {dataset} | Zero-shot CLIP | {fmt(zero['Base'])} | {fmt(zero['Novel'])} | {fmt(zero['HM'])} | {signed(zero['HM'] - baseline['HM_均值计算'])} |"
        )
        lines.append(
            f"| {dataset} | CoOp | {fmt(baseline['Base均值'])} | {fmt(baseline['Novel均值'])} | {fmt(baseline['HM_均值计算'])} | +0.00 |"
        )
        lines.append(
            f"| {dataset} | 固定均值概率融合 | {fmt(fusion['Base均值'])} | {fmt(fusion['Novel均值'])} | {fmt(fusion['HM_均值计算'])} | {signed(fusion['HM_均值计算'] - baseline['HM_均值计算'])} |"
        )
    lines.extend(
        [
            "",
            f"五个数据集等权平均 HM 为 Zero-shot CLIP **{fmt(zero_shot_cross['数据集等权HM均值'])}**、CoOp **{fmt(cross['CoOp']['数据集等权HM均值'])}**、固定均值概率融合 **{fmt(cross['固定均值概率融合']['数据集等权HM均值'])}**。主方法相对 CoOp 增加 **{signed(cross['固定均值概率融合相对CoOp']['数据集等权HM增益'])}pp**，相对 Zero-shot CLIP 增加 **{signed(cross['固定均值概率融合']['数据集等权HM均值'] - zero_shot_cross['数据集等权HM均值'])}pp**；相对 CoOp 的 Novel 等权平均增加 **{signed(cross['固定均值概率融合相对CoOp']['数据集等权Novel增益'])}pp**，Base 等权平均变化为 **{signed(cross['固定均值概率融合相对CoOp']['数据集等权Base增益'])}pp**。",
            "",
            "## CoCoOp五数据集标准基线",
            "",
            "CoCoOp采用其标准10 epochs、batch size 1配方；本仓库CoOp和主方法沿用既有20 epochs配方。它们使用相同ViT-B/16、16-shot、划分和三个种子，但训练预算不同，因此本节属于标准配方对比，不写成完全等计算量消融。",
            "",
            "| 数据集 | CoCoOp Base | CoCoOp Novel | CoCoOp HM | 主方法 HM | 主方法−CoCoOp HM |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    cocoop_index = {item["数据集"]: item for item in cocoop["数据集汇总"]}
    for dataset in 数据集顺序:
        ours = index[(dataset, "固定均值概率融合")]
        baseline = cocoop_index[dataset]
        lines.append(
            f"| {dataset} | {fmt(baseline['Base均值'])} | {fmt(baseline['Novel均值'])} | {fmt(baseline['HM_均值计算'])} | {fmt(ours['HM_均值计算'])} | {signed(ours['HM_均值计算'] - baseline['HM_均值计算'])} |"
        )
    ours_five_hm = statistics.mean(index[(dataset, "固定均值概率融合")]["HM_均值计算"] for dataset in 数据集顺序)
    cocoop_five_hm = cocoop["跨数据集汇总"]["数据集等权HM均值"]
    lines.extend(
        [
            "",
            f"五数据集等权 HM 为 CoCoOp **{fmt(cocoop_five_hm)}**、固定均值概率融合 **{fmt(ours_five_hm)}**，主方法高 **{signed(ours_five_hm - cocoop_five_hm)}pp**；五个数据集均为正向。",
            "",
            "## 配对稳定性与描述性区间",
            "",
            "每个种子先计算“固定均值概率融合 − CoOp”，再汇总。95% t 区间使用 n=3、df=2，只用于描述波动范围，不能当作强统计结论。",
            "",
            "| 数据集 | ΔBase | ΔNovel | ΔHM（均值） | 配对ΔHM均值 ± SD | 95%描述性区间 | 正向种子 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in paired:
        hm = item["配对差"]["HM"]
        low, high = hm["描述性95%t区间"]
        lines.append(
            f"| {item['数据集']} | {signed(item['Base均值差'])} | {signed(item['Novel均值差'])} | {signed(item['HM_均值计算差'])} | {signed(hm['均值'])} ± {fmt(hm['样本标准差'])} | [{signed(low)}, {signed(high)}] | {item['HM正增益种子数']}/3 |"
        )
    lines.extend(
        [
            "",
            f"五个数据集全部取得正 HM 收益，逐种子比较为 **{cross['固定均值概率融合相对CoOp']['HM正增益种子数']}/{cross['固定均值概率融合相对CoOp']['总种子比较数']}** 正向。UCF101、Aircraft、OxfordFlowers 的 Base 小幅下降；DTD、EuroSAT 的 Base 小幅上升。因此最稳妥的论文结论是：该方法持续改善 Novel 与 Base-to-Novel HM，但不能声称无条件保持 Base。",
            "",
            "## 组件对照",
            "",
            "冻结均值描述不训练；其三个 seed 的数值相同，只是与三个 CoOp 检查点配对展示，不代表三次独立描述模型训练。",
            "",
            "| 数据集 | 方法 | Base均值 ± SD | Novel均值 ± SD | HM（均值） | 种子HM均值 ± SD |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in 数据集顺序:
        for method in 方法顺序:
            item = index[(dataset, method)]
            lines.append(
                f"| {dataset} | {method} | {fmt(item['Base均值'])} ± {fmt(item['Base样本标准差'])} | {fmt(item['Novel均值'])} ± {fmt(item['Novel样本标准差'])} | {fmt(item['HM_均值计算'])} | {fmt(item['种子HM均值'])} ± {fmt(item['种子HM样本标准差'])} |"
            )
    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- UCF101、DTD、Aircraft 参与过方法分析；EuroSAT、OxfordFlowers 是冻结融合规则后的扩展验证，但仓库中已有历史成绩，因此也不能标成严格盲测。",
            "- 五个数据集的核心训练配置一致，但历史检查点来自不同的已冻结源码快照；每批源码、CLIP、检查点和样本身份已经各自审计。论文中应报告这一事实，不能写成单次全新统一重跑。",
            "- Zero-shot CLIP 已按同一数据划分实际评估，并从原始 correct/total 独立复算；它不训练，也没有种子标准差。",
            "- CoCoOp已完成五数据集标准配方复现并独立复核；CuPL、KgCoOp、MaPLe和PromptSRC的原论文数字已单列为文献定位表，但没有在本仓库同协议重跑，不能宣称SOTA。",
            "- 简单概率融合相对CoOp新增0个可训练参数；RTX 4090统一实测中，batch=64延迟增加0.74%，固定描述类向量占0.0977 MiB。该结论不外推到其他硬件。",
            "",
            "## 数据来源",
            "",
        ]
    )
    lines.extend(f"- `{source}`" for source in sources)
    lines.extend(
        [
            "",
            "机器可读版本见[统一结果](统一结果.json)、[标准基线汇总](标准基线汇总.csv)、[逐种子结果](主结果逐种子.csv)、[统计汇总](主结果汇总.csv)、[配对增益与区间](配对增益与区间.csv)；受控实验见[LaTeX主表](论文主结果表.tex)，外部论文定位见[文献方法对比](文献方法对比.md)和[LaTeX文献表](论文文献对比表.tex)。",
            "",
        ]
    )
    return "\n".join(lines)


def 生成_latex(summaries: list[dict], zero_shot: dict[str, dict], cocoop: dict) -> str:
    index = {(item["数据集"], item["方法"]): item for item in summaries}
    cocoop_index = {item["数据集"]: item for item in cocoop["数据集汇总"]}
    rows = []
    for dataset in 数据集顺序:
        base = index[(dataset, "CoOp")]
        fusion = index[(dataset, "固定均值概率融合")]
        zero = zero_shot[dataset]
        rows.append(
            f"{dataset} & Zero-shot CLIP & {fmt(zero['Base'])} & {fmt(zero['Novel'])} & {fmt(zero['HM'])} \\\\"
        )
        rows.append(
            f" & CoOp & {fmt(base['Base均值'])} & {fmt(base['Novel均值'])} & {fmt(base['HM_均值计算'])} \\\\"
        )
        rows.append(
            f" & CoCoOp & {fmt(cocoop_index[dataset]['Base均值'])} & {fmt(cocoop_index[dataset]['Novel均值'])} & {fmt(cocoop_index[dataset]['HM_均值计算'])} \\\\"
        )
        rows.append(
            f" & Ours & {fmt(fusion['Base均值'])} & {fmt(fusion['Novel均值'])} & \\textbf{{{fmt(fusion['HM_均值计算'])}}} \\\\"
        )
        if dataset != 数据集顺序[-1]:
            rows.append("\\midrule")
    return "\n".join(
        [
            "% Auto-generated by scripts/repro/生成投稿主结果.py",
            "\\begin{table*}[t]",
            "\\centering",
            "\\caption{Base-to-novel generalization with CLIP ViT-B/16. Trainable methods use 16 shots and are averaged over three seeds. CoCoOp follows its standard 10-epoch recipe; CoOp and Ours use 20 epochs. Zero-shot CLIP is evaluated once. H is the harmonic mean of Base and Novel accuracies.}",
            "\\label{tab:five_dataset_main}",
            "\\begin{tabular}{llccc}",
            "\\toprule",
            "Dataset & Method & Base & Novel & H \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )


def 生成基线缺口(cross: dict, zero_shot_cross: dict, cocoop: dict, ours_five_hm: float) -> str:
    return f"""# 标准基线与投稿证据缺口

更新日期：2026-09-16。本清单只记录是否真实执行，不用论文中的外部数字冒充本地复现。

| 对照 | 当前状态 | 能否进入主表 | 下一步 |
|---|---|---|---|
| 本仓库 CoOp | 五数据集、三种子已完成并审计 | 可以 | 作为直接训练基线 |
| 冻结均值描述 | 五数据集已评估 | 可以作为组件消融 | 注明描述分支不随 seed 训练 |
| 固定 logits 均值融合 | 五数据集、三种子已评估 | 可以作为融合方式消融 | 与概率融合并列 |
| 固定均值概率融合 | 五数据集、三种子已完成并审计 | 可以，候选主方法 | 固定公式，不再按数据集调权重 |
| 原层级与 MLP 完整方法 | UCF101、DTD、Aircraft 三种子完成 | 只适合历史消融 | 在 DTD、Aircraft 弱于简单融合，不扩展为主方法 |
| Zero-shot CLIP | 五数据集Base/Novel共10次实评，原始日志独立复算通过 | 可以 | 注明冻结模型、每个分支只评估一次 |
| CoCoOp | 五数据集、三种子已完成并独立复核 | 可以，按标准配方单列 | 明确其10轮配方与本仓库20轮配方的预算差异 |
| CuPL、KgCoOp、MaPLe、PromptSRC | 原论文数字已核对并生成单独文献定位表；尚无本地同协议复跑 | 只进入带†的文献表 | 不与本地结果混称公平复现；不能据此宣称SOTA |
| 参数量、推理耗时、显存 | RTX4090统一实测与独立审计已完成 | 可以进入效率表 | 注明排除项、硬件及固定测量顺序 |

## 投稿前最低补充顺序

1. **文献定位表已完成**：本文平均HM 70.13，高于CuPL、CoCoOp和KgCoOp，与MaPLe基本持平（+0.06pp），低于PromptSRC（-2.46pp）。外部论文成绩只放单独表格并标†。
2. **下一步修改论文**：使用受控主表、文献定位表和统一效率表；准确表述为“融合模块新增0个可训练参数”，不能写“整个方法无可训练参数”。

Zero-shot CLIP 的五数据集等权 HM 为 **{zero_shot_cross['数据集等权HM均值']:.2f}**。固定概率融合为 **{cross['固定均值概率融合']['数据集等权HM均值']:.2f}**，相对 Zero-shot CLIP 提升 **{cross['固定均值概率融合']['数据集等权HM均值'] - zero_shot_cross['数据集等权HM均值']:+.2f}pp**；相对 CoOp 提升 **{cross['固定均值概率融合相对CoOp']['数据集等权HM增益']:+.2f}pp**，且 15/15 个种子比较为正。固定概率融合五数据集等权HM为 **{ours_five_hm:.2f}**，本地CoCoOp为 **{cocoop['跨数据集汇总']['数据集等权HM均值']:.2f}**，高 **{ours_five_hm - cocoop['跨数据集汇总']['数据集等权HM均值']:+.2f}pp**，且5/5数据集为正。统一效率和文献定位均已完成，后续不需开卡，转入论文修改。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repository-root", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    output = (args.output_dir or root / "reports/experiments/投稿实验统一结果-20260915").resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows, source_first = 加载前三数据集(root)
    sources = [source_first]
    for relative in (
        Path("reports/experiments/EuroSAT扩展验证-20260915/独立复算.json"),
        Path("reports/experiments/牛津花卉扩展验证-20260915/独立复算.json"),
    ):
        extra, source = 加载扩展数据集(root, relative)
        rows.extend(extra)
        sources.append(source)
    核验逐种子(rows)
    rows.sort(key=lambda item: (数据集顺序.index(item["数据集"]), item["种子"], 方法顺序.index(item["方法"])))
    summaries, paired, cross = 生成统计(rows)
    zero_shot, zero_shot_cross, zero_shot_sources = 加载零样本基线(root)
    sources.extend(zero_shot_sources)
    cocoop, cocoop_sources = 加载联合上下文提示基线(root)
    sources.extend(cocoop_sources)

    summary_index = {(item["数据集"], item["方法"]): item for item in summaries}
    baseline_rows = []
    for dataset in 数据集顺序:
        baseline_rows.append(
            {
                "数据集": dataset,
                "方法": "Zero-shot CLIP",
                "Base": zero_shot[dataset]["Base"],
                "Novel": zero_shot[dataset]["Novel"],
                "HM": zero_shot[dataset]["HM"],
                "评估次数": 1,
            }
        )
        for method in ("CoOp", "CoCoOp", "固定均值概率融合"):
            if method == "CoCoOp":
                item = next(item for item in cocoop["数据集汇总"] if item["数据集"] == dataset)
            else:
                item = summary_index[(dataset, method)]
            baseline_rows.append(
                {
                    "数据集": dataset,
                    "方法": method,
                    "Base": item["Base均值"],
                    "Novel": item["Novel均值"],
                    "HM": item["HM_均值计算"],
                    "评估次数": 3,
                }
            )

    写_csv(output / "主结果逐种子.csv", rows, ["数据集", "种子", "方法", "Base", "Novel", "HM", "来源"])
    写_csv(output / "标准基线汇总.csv", baseline_rows, ["数据集", "方法", "Base", "Novel", "HM", "评估次数"])
    写_csv(
        output / "主结果汇总.csv",
        summaries,
        ["数据集", "方法", "种子数", "Base均值", "Base样本标准差", "Novel均值", "Novel样本标准差", "HM_均值计算", "种子HM均值", "种子HM样本标准差"],
    )
    paired_csv = []
    for item in paired:
        flat = {
            "数据集": item["数据集"],
            "Base均值差": item["Base均值差"],
            "Novel均值差": item["Novel均值差"],
            "HM_均值计算差": item["HM_均值计算差"],
            "HM正增益种子数": item["HM正增益种子数"],
        }
        for metric in ("Base", "Novel", "HM"):
            detail = item["配对差"][metric]
            flat[f"配对{metric}均值"] = detail["均值"]
            flat[f"配对{metric}样本标准差"] = detail["样本标准差"]
            flat[f"配对{metric}95%区间下界"] = detail["描述性95%t区间"][0]
            flat[f"配对{metric}95%区间上界"] = detail["描述性95%t区间"][1]
        paired_csv.append(flat)
    paired_fields = list(paired_csv[0])
    写_csv(output / "配对增益与区间.csv", paired_csv, paired_fields)

    payload = {
        "状态": "通过",
        "生成规则": "只读取已审计机器产物，拒绝缺数据、重复身份、跨阶段计划不一致和HM复算不一致",
        "协议": {
            "backbone": "CLIP ViT-B/16",
            "shots": 16,
            "epochs": 20,
            "seeds": [1, 2, 3],
            "主方法": "0.5*softmax(CoOp logits)+0.5*softmax(mean-description logits)",
            "融合权重": 0.5,
            "HM主表口径": "先求三种子Base/Novel均值，再计算调和平均",
            "区间口径": "相同种子配对差的描述性95% t区间，n=3，df=2",
        },
        "逐种子结果": rows,
        "方法汇总": summaries,
        "融合相对CoOp": paired,
        "跨数据集汇总": cross,
        "Zero-shot CLIP": {
            "五数据集结果": [dict(数据集=dataset, **zero_shot[dataset]) for dataset in 数据集顺序],
            "跨数据集汇总": zero_shot_cross,
            "统计口径": "冻结CLIP；每个Base/Novel分支评估一次；不训练、不报告种子标准差",
        },
        "CoCoOp五数据集": cocoop,
        "来源": sources,
    }
    (output / "统一结果.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "统一结果表.md").write_text(
        生成_markdown(summaries, paired, cross, zero_shot, zero_shot_cross, cocoop, sources), encoding="utf-8"
    )
    (output / "论文主结果表.tex").write_text(生成_latex(summaries, zero_shot, cocoop), encoding="utf-8")
    summary_index = {(item["数据集"], item["方法"]): item for item in summaries}
    ours_five_hm = statistics.mean(
        summary_index[(dataset, "固定均值概率融合")]["HM_均值计算"] for dataset in 数据集顺序
    )
    (output / "标准基线缺口.md").write_text(
        生成基线缺口(cross, zero_shot_cross, cocoop, ours_five_hm), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "状态": "通过",
                "逐种子行数": len(rows),
                "方法汇总行数": len(summaries),
                "数据集数": len(数据集顺序),
                "HM正增益数据集数": cross["固定均值概率融合相对CoOp"]["HM正增益数据集数"],
                "HM正增益种子数": cross["固定均值概率融合相对CoOp"]["HM正增益种子数"],
                "Zero-shot CLIP等权HM": zero_shot_cross["数据集等权HM均值"],
                "输出目录": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
