#!/usr/bin/env python3
"""不调用主执行脚本，独立核对已保存权重、输入哈希、逐折指标和汇总。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


根目录 = Path(__file__).resolve().parents[3]
结果路径 = 根目录 / "reports/experiments/类别语义可靠性必要性检验-20260915/结果.json"
复算路径 = 结果路径.with_name("独立复算.json")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def folds_for(dataset, seed, split, labels):
    folds = np.empty(len(labels), dtype=np.int8)
    for c in range(int(labels.max()) + 1):
        indices = np.flatnonzero(labels == c).tolist()
        indices.sort(key=lambda row: hashlib.sha256(
            f"{dataset}|{seed}|{split}|{c}|{row}".encode()).digest())
        for rank, row in enumerate(indices):
            folds[row] = rank % 2
    return folds


def metrics(S, M, y, alpha):
    z = S + (M - S) * alpha
    shifted = z - z.max(1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(1)) + z.max(1)
    nll = float((logsumexp - z[np.arange(len(y)), y]).mean())
    return {"准确率": float((z.argmax(1) == y).mean()), "NLL": nll}


def close(a, b, tol=1e-10):
    return abs(float(a) - float(b)) <= tol


def harmonic(a, b):
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def main():
    result = json.loads(结果路径.read_text())
    errors = []
    recalculated = {}
    split_lookup = {}
    for row in result["逐划分"]:
        path = 根目录 / row["输入路径"]
        if sha(path) != row["输入SHA256"]:
            errors.append(f"输入哈希不一致: {row['输入路径']}")
        with np.load(path, allow_pickle=False) as p:
            y, S, M = p["labels"].astype(int), p["S"].astype(float), p["M"].astype(float)
        folds = folds_for(row["数据集"], row["种子"], row["划分"], y)
        key = (row["数据集"], row["种子"], row["划分"])
        split_lookup[key] = {}
        for saved in row["两折"]:
            fold = saved["评估折"]
            test = folds == fold
            if int(test.sum()) != saved["评估样本数"] or int((~test).sum()) != saved["训练样本数"]:
                errors.append(f"折大小不一致: {key} fold{fold}")
            values = np.asarray(saved["类别权重"]["alpha值"], dtype=float)[None, :]
            checks = {
                "固定0.5": metrics(S[test], M[test], y[test], 0.5),
                "训练折标量": metrics(S[test], M[test], y[test], float(saved["训练折标量"]["alpha"])),
                "类别权重": metrics(S[test], M[test], y[test], values),
            }
            for method, calculated in checks.items():
                for metric, value in calculated.items():
                    if not close(value, saved[method][metric]):
                        errors.append(f"指标不一致: {key} fold{fold} {method} {metric}")
            split_lookup[key][fold] = checks

    dataset_rows = []
    for dataset in sorted({k[0] for k in split_lookup}):
        seeds = sorted({k[1] for k in split_lookup if k[0] == dataset})
        run_metrics = []
        for seed in seeds:
            fold_metrics = []
            for fold in (0, 1):
                item = {}
                for method in ("固定0.5", "训练折标量", "类别权重"):
                    base = split_lookup[(dataset, seed, "Base")][fold][method]["准确率"]
                    novel = split_lookup[(dataset, seed, "Novel")][fold][method]["准确率"]
                    item[method] = {"Base": base, "Novel": novel, "HM": harmonic(base, novel)}
                fold_metrics.append(item)
            run_metrics.append({method: {metric: float(np.mean([f[method][metric] for f in fold_metrics]))
                                         for metric in ("Base", "Novel", "HM")}
                                for method in ("固定0.5", "训练折标量", "类别权重")})
        methods = {method: {metric: float(np.mean([r[method][metric] for r in run_metrics]))
                            for metric in ("Base", "Novel", "HM")}
                   for method in ("固定0.5", "训练折标量", "类别权重")}
        dataset_rows.append({
            "数据集": dataset,
            "指标": methods,
            "类别权重相对固定HM增益pp": 100 * (methods["类别权重"]["HM"] - methods["固定0.5"]["HM"]),
            "类别权重相对标量HM增益pp": 100 * (methods["类别权重"]["HM"] - methods["训练折标量"]["HM"]),
        })

    saved_by_dataset = {x["数据集"]: x for x in result["汇总"]["逐数据集"]}
    for row in dataset_rows:
        saved = saved_by_dataset[row["数据集"]]
        for key in ("类别权重相对固定HM增益pp", "类别权重相对标量HM增益pp"):
            if not close(row[key], saved[key]):
                errors.append(f"数据集汇总不一致: {row['数据集']} {key}")
    mean_fixed = float(np.mean([x["类别权重相对固定HM增益pp"] for x in dataset_rows]))
    mean_scalar = float(np.mean([x["类别权重相对标量HM增益pp"] for x in dataset_rows]))
    if not close(mean_fixed, result["汇总"]["等权平均类别权重相对固定HM增益pp"]):
        errors.append("等权固定增益不一致")
    if not close(mean_scalar, result["汇总"]["等权平均类别权重相对标量HM增益pp"]):
        errors.append("等权标量增益不一致")
    protocol_path = 根目录 / "reports/preparation/类别语义可靠性-20260915/必要性检验协议.json"
    if sha(protocol_path) != result["协议SHA256"]:
        errors.append("协议哈希不一致")
    receipt = {
        "状态": "通过" if not errors else "失败",
        "错误": errors,
        "核对输入数": len(result["逐划分"]),
        "核对评估折数": 2 * len(result["逐划分"]),
        "独立复算等权相对固定HM增益pp": mean_fixed,
        "独立复算等权相对标量HM增益pp": mean_scalar,
        "结果文件SHA256": sha(结果路径),
        "说明": "复算器不导入主执行脚本；从保存的alpha和原始S/M重新计算逐折准确率、NLL、HM及数据集等权汇总。",
    }
    复算路径.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
