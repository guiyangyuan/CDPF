#!/usr/bin/env python3
"""独立复核 ImageNet 三种子结果，并计算十一数据集增补统计。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics


目录 = Path(__file__).resolve().parent
评估目录 = 目录 / "融合评估"
种子 = (1, 2, 3)
分支 = {
    "student": "CoOp",
    "description_mean": "冻结均值描述",
    "simple_fusion": "固定logit均值融合",
    "mean_probability_fusion": "固定均值概率融合",
    "classname": "类别名称",
    "classname_probability_fusion": "CoOp+类别名称",
}
权重分支 = {
    0.0: "description_probability_alpha_0_00",
    0.25: "description_probability_alpha_0_25",
    0.5: "description_probability_alpha_0_50",
    0.75: "description_probability_alpha_0_75",
    1.0: "description_probability_alpha_1_00",
}


def 调和平均(base: float, novel: float) -> float:
    return 2 * base * novel / (base + novel)


def 均值标准差(values: list[float]) -> dict[str, float]:
    return {"均值": statistics.mean(values), "样本标准差": statistics.stdev(values)}


def 读取指标(seed: int, split: str) -> dict:
    path = 评估目录 / f"seed{seed}" / split / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed", path
    assert payload["dataset"] == "ImageNet", path
    assert payload["seed"] == seed, path
    assert payload["subsample"] == split, path
    assert payload["input_hashes_before"] == payload["input_hashes_after"], path
    assert payload["source"]["model_sha256"] == payload["source"]["expected_model_sha256"], path
    for name, value in payload["metrics"].items():
        assert value["total"] == 25000, (path, name)
        expected = 100 * value["correct"] / value["total"]
        assert math.isclose(value["accuracy"], expected, abs_tol=1e-12), (path, name)
    return payload


def main() -> None:
    状态 = json.loads((目录 / "执行状态.json").read_text(encoding="utf-8"))
    数据核验 = json.loads((目录 / "数据核验.json").read_text(encoding="utf-8"))
    汇总 = json.loads((目录 / "结果汇总.json").read_text(encoding="utf-8"))
    assert 状态["状态"] == "完成" and 状态["已完成命令"] == 16
    assert 数据核验["状态"] == "通过"
    assert 数据核验["图像"]["训练图像数"] == 1281167
    assert 数据核验["图像"]["验证图像数"] == 50000
    assert 数据核验["类别与描述"]["synset数"] == 1000

    原始: dict[int, dict[str, dict[str, float]]] = {}
    评估器哈希 = set()
    for seed in 种子:
        base = 读取指标(seed, "base")
        novel = 读取指标(seed, "new")
        评估器哈希.update((base["source"]["evaluator_sha256"], novel["source"]["evaluator_sha256"]))
        原始[seed] = {}
        for branch in set(分支) | set(权重分支.values()):
            b = base["metrics"][branch]["accuracy"]
            n = novel["metrics"][branch]["accuracy"]
            原始[seed][branch] = {"Base": b, "Novel": n, "HM": 调和平均(b, n)}

    assert 评估器哈希 == {状态["评估器统一SHA256"]}
    for item in 汇总["逐种子"]:
        seed = item["种子"]
        for branch in ("student", "description_mean", "simple_fusion", "mean_probability_fusion"):
            for metric in ("Base", "Novel", "HM"):
                assert math.isclose(
                    item["方法"][branch][metric], 原始[seed][branch][metric], abs_tol=1e-12
                )

    三种子统计 = {}
    for branch, display in 分支.items():
        三种子统计[display] = {
            metric: 均值标准差([原始[seed][branch][metric] for seed in 种子])
            for metric in ("Base", "Novel", "HM")
        }

    配对增益 = {
        metric: 均值标准差([
            原始[seed]["mean_probability_fusion"][metric] - 原始[seed]["student"][metric]
            for seed in 种子
        ])
        for metric in ("Base", "Novel", "HM")
    }
    assert all(
        原始[seed]["mean_probability_fusion"]["HM"] > 原始[seed]["student"]["HM"]
        for seed in 种子
    )

    ImageNet权重 = []
    for alpha, branch in 权重分支.items():
        rows = [原始[seed][branch] for seed in 种子]
        ImageNet权重.append({
            "alpha": alpha,
            "Base": statistics.mean(row["Base"] for row in rows),
            "Novel": statistics.mean(row["Novel"] for row in rows),
            "HM": statistics.mean(row["HM"] for row in rows),
        })
    assert max(ImageNet权重, key=lambda row: row["HM"])["alpha"] == 0.5

    十数据集目录 = 目录.parent / "投稿关键对照十数据集-20260918" / "汇总"
    with (十数据集目录 / "关键对照数据集汇总.csv").open(encoding="utf-8-sig") as handle:
        十数据集行 = list(csv.DictReader(handle))
    十数据集权重 = list(csv.DictReader(
        (十数据集目录 / "统一权重稳健性.csv").open(encoding="utf-8-sig")
    ))

    十一数据集方法 = {}
    对应分支 = {
        "CoOp": "student",
        "类别名称": "classname",
        "CoOp+类别名称": "classname_probability_fusion",
        "类别描述": "description_mean",
        "TextConsistency": "mean_probability_fusion",
    }
    for method, branch in 对应分支.items():
        rows = [row for row in 十数据集行 if row["方法"] == method]
        assert len(rows) == 10, method
        image = 三种子统计[分支[branch]]
        十一数据集方法[method] = {
            "Base宏平均": (sum(float(row["Base均值"]) for row in rows) + image["Base"]["均值"]) / 11,
            "Novel宏平均": (sum(float(row["Novel均值"]) for row in rows) + image["Novel"]["均值"]) / 11,
            "HM宏平均": (sum(float(row["HM_均值计算"]) for row in rows) + image["HM"]["均值"]) / 11,
        }

    十一数据集权重 = []
    image_by_alpha = {row["alpha"]: row for row in ImageNet权重}
    for row in 十数据集权重:
        alpha = float(row["alpha"])
        image = image_by_alpha[alpha]
        十一数据集权重.append({
            "alpha": alpha,
            "Base宏平均": (10 * float(row["宏平均Base"]) + image["Base"]) / 11,
            "Novel宏平均": (10 * float(row["宏平均Novel"]) + image["Novel"]) / 11,
            "HM宏平均": (10 * float(row["宏平均HM"]) + image["HM"]) / 11,
        })
    assert max(十一数据集权重, key=lambda row: row["HM宏平均"])["alpha"] == 0.5

    结果 = {
        "状态": "通过",
        "数据与命令": {
            "训练图像": 1281167,
            "验证图像": 50000,
            "类别": 1000,
            "种子": list(种子),
            "完成命令": 状态["已完成命令"],
            "评估器SHA256": next(iter(评估器哈希)),
        },
        "ImageNet三种子统计": 三种子统计,
        "ImageNet融合相对CoOp配对增益": 配对增益,
        "ImageNet权重稳健性": ImageNet权重,
        "十一数据集方法宏平均": 十一数据集方法,
        "十一数据集权重稳健性": 十一数据集权重,
        "结论": {
            "ImageNet三种子HM均为正增益": True,
            "ImageNet最佳全局权重": 0.5,
            "十一数据集最佳全局权重": 0.5,
        },
    }
    (目录 / "独立复核.json").write_text(
        json.dumps(结果, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (目录 / "三种子统计.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("方法", "Base均值", "Base样本标准差", "Novel均值", "Novel样本标准差",
                         "种子HM均值", "种子HM样本标准差"))
        for method, metrics in 三种子统计.items():
            writer.writerow((method, metrics["Base"]["均值"], metrics["Base"]["样本标准差"],
                             metrics["Novel"]["均值"], metrics["Novel"]["样本标准差"],
                             metrics["HM"]["均值"], metrics["HM"]["样本标准差"]))
    with (目录 / "十一数据集方法宏平均.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("方法", "Base宏平均", "Novel宏平均", "HM宏平均"))
        writer.writeheader()
        for method, metrics in 十一数据集方法.items():
            writer.writerow({"方法": method, **metrics})
    with (目录 / "权重稳健性.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("范围", "alpha", "Base", "Novel", "HM"))
        for row in ImageNet权重:
            writer.writerow(("ImageNet", row["alpha"], row["Base"], row["Novel"], row["HM"]))
        for row in 十一数据集权重:
            writer.writerow(("十一数据集", row["alpha"], row["Base宏平均"],
                             row["Novel宏平均"], row["HM宏平均"]))


if __name__ == "__main__":
    main()
