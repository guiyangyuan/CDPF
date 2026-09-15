#!/usr/bin/env python3
"""独立重建三特征岭回归和完整指标，不导入迁移探针主脚本。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


根目录 = Path(__file__).resolve().parents[3]
结果路径 = 根目录 / "reports/experiments/类别语义可靠性迁移探针-20260915/结果.json"
必要性路径 = 根目录 / "reports/experiments/类别语义可靠性必要性检验-20260915/结果.json"
特征路径 = 根目录 / "reports/experiments/类别语义可靠性迁移探针-20260915/服务器文本导出/类别文本几何.json"
输出路径 = 结果路径.with_name("独立复算.json")
特征名 = ["类内一致性", "类别名对齐", "跨类间隔"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, tol=1e-10):
    return abs(float(a) - float(b)) <= tol


def harmonic(a, b):
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def fit(X, y):
    mean, std = X.mean(0), X.std(0)
    std = np.where(std < 1e-12, 1.0, std)
    design = np.column_stack([(X - mean) / std, np.ones(len(X))])
    penalty = np.diag([1.0, 1.0, 1.0, 0.0])
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return weights, mean, std


def predict(X, model):
    weights, mean, std = model
    design = np.column_stack([(X - mean) / std, np.ones(len(X))])
    return np.clip(design @ weights, 0.0, 1.0)


def actual_accuracy(path, alpha):
    with np.load(path, allow_pickle=False) as p:
        y, S, M = p["labels"].astype(int), p["S"].astype(float), p["M"].astype(float)
    return float(((S + (M - S) * alpha).argmax(1) == y).mean())


def correlation(a, b):
    return 0.0 if np.std(a) < 1e-12 or np.std(b) < 1e-12 else float(np.corrcoef(a, b)[0, 1])


def main():
    saved = json.loads(结果路径.read_text())
    necessary = json.loads(必要性路径.read_text())
    features = json.loads(特征路径.read_text())
    feature_lookup = {(x["数据集"], x["划分"]): x for x in features["数据"]}
    target_lookup = {(x["数据集"], x["种子"], x["划分"]): x for x in necessary["逐划分"]}
    errors = []
    recalculated = []
    for row in saved["逐运行"]:
        dataset, seed = row["数据集"], row["种子"]
        split_data = {}
        for split in ("Base", "Novel"):
            frow = feature_lookup[(dataset, split)]
            trow = target_lookup[(dataset, seed, split)]
            X = np.asarray([[item[name] for name in 特征名] for item in frow["特征"]], dtype=float)
            target = np.mean([np.asarray(fold["类别权重"]["alpha值"], dtype=float)
                              for fold in trow["两折"]], axis=0)
            split_data[split] = {"X": X, "target": target, "path": 根目录 / trow["输入路径"]}
        model = fit(split_data["Base"]["X"], split_data["Base"]["target"])
        pred = {split: predict(split_data[split]["X"], model) for split in ("Base", "Novel")}
        scalar = float(split_data["Base"]["target"].mean())
        metrics = {}
        for method, values in {"固定0.5": {"Base": .5, "Novel": .5},
                               "Base均值标量": {"Base": scalar, "Novel": scalar},
                               "文本映射": pred}.items():
            base = actual_accuracy(split_data["Base"]["path"], values["Base"])
            novel = actual_accuracy(split_data["Novel"]["path"], values["Novel"])
            metrics[method] = {"Base": base, "Novel": novel, "HM": harmonic(base, novel)}
            for name, value in metrics[method].items():
                if not close(value, row["指标"][method][name]):
                    errors.append(f"指标不一致:{dataset}:seed{seed}:{method}:{name}")
        extras = {
            "Base均值标量": scalar,
            "文本映射Base预测均值": float(pred["Base"].mean()),
            "文本映射Novel预测均值": float(pred["Novel"].mean()),
            "Novel诊断上界目标均值": float(split_data["Novel"]["target"].mean()),
            "Novel预测与诊断上界目标Pearson": correlation(pred["Novel"], split_data["Novel"]["target"]),
            "Novel预测与诊断上界目标MAE": float(np.abs(pred["Novel"] - split_data["Novel"]["target"]).mean()),
        }
        for name, value in extras.items():
            if not close(value, row[name]):
                errors.append(f"诊断不一致:{dataset}:seed{seed}:{name}")
        recalculated.append({"数据集": dataset, "种子": seed, "指标": metrics})
    dataset_results = []
    for dataset in sorted({x["数据集"] for x in recalculated}):
        rows = [x for x in recalculated if x["数据集"] == dataset]
        summary = {method: {metric: float(np.mean([r["指标"][method][metric] for r in rows]))
                            for metric in ("Base", "Novel", "HM")}
                   for method in ("固定0.5", "Base均值标量", "文本映射")}
        dataset_results.append({
            "数据集": dataset,
            "相对固定": 100 * (summary["文本映射"]["HM"] - summary["固定0.5"]["HM"]),
            "相对标量": 100 * (summary["文本映射"]["HM"] - summary["Base均值标量"]["HM"]),
        })
    mean_fixed = float(np.mean([x["相对固定"] for x in dataset_results]))
    mean_scalar = float(np.mean([x["相对标量"] for x in dataset_results]))
    if not close(mean_fixed, saved["等权平均文本映射相对固定HM增益pp"]):
        errors.append("等权固定增益不一致")
    if not close(mean_scalar, saved["等权平均文本映射相对Base均值标量HM增益pp"]):
        errors.append("等权标量增益不一致")
    for path, expected in ((必要性路径, saved["必要性结果SHA256"]),
                           (特征路径, saved["文本特征SHA256"])):
        if sha(path) != expected:
            errors.append(f"输入哈希不一致:{path.name}")
    receipt = {
        "状态": "通过" if not errors else "失败",
        "错误": errors,
        "核对运行数": len(recalculated),
        "独立复算等权相对固定HM增益pp": mean_fixed,
        "独立复算等权相对Base均值标量HM增益pp": mean_scalar,
        "结果SHA256": sha(结果路径),
        "说明": "不导入主脚本；独立重建Base标准化、lambda=1岭回归、裁剪、Base/Novel预测、全部准确率与HM。",
    }
    输出路径.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
