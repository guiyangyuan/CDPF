#!/usr/bin/env python3
"""用Base类别目标拟合三特征岭回归，并在Novel类别零标签迁移。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


根目录 = Path(__file__).resolve().parents[3]
必要性路径 = 根目录 / "reports/experiments/类别语义可靠性必要性检验-20260915/结果.json"
协议路径 = Path(__file__).with_name("迁移探针协议.json")
输出目录 = 根目录 / "reports/experiments/类别语义可靠性迁移探针-20260915"
特征名 = ["类内一致性", "类别名对齐", "跨类间隔"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit_ridge(X, y, regularization=1.0):
    mean, std = X.mean(0), X.std(0)
    std = np.where(std < 1e-12, 1.0, std)
    Z = (X - mean) / std
    design = np.column_stack([Z, np.ones(len(Z))])
    penalty = np.diag([regularization] * X.shape[1] + [0.0])
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return weights, mean, std


def predict(X, model):
    weights, mean, std = model
    design = np.column_stack([(X - mean) / std, np.ones(len(X))])
    return np.clip(design @ weights, 0.0, 1.0)


def metrics(path: Path, alpha):
    with np.load(path, allow_pickle=False) as p:
        y, S, M = p["labels"].astype(int), p["S"].astype(float), p["M"].astype(float)
    z = S + (M - S) * alpha
    return {"准确率": float((z.argmax(1) == y).mean()), "样本数": len(y)}


def hm(a, b):
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(协议路径.read_text())
    necessary = json.loads(必要性路径.read_text())
    features_payload = json.loads(args.features.read_text())
    feature_lookup = {(x["数据集"], x["划分"]): x for x in features_payload["数据"]}
    necessary_lookup = {(x["数据集"], x["种子"], x["划分"]): x for x in necessary["逐划分"]}
    results = []
    for run in necessary["汇总"]["逐运行"]:
        dataset, seed = run["数据集"], run["种子"]
        split_data = {}
        for split in ("Base", "Novel"):
            feature_row = feature_lookup[(dataset, split)]
            target_row = necessary_lookup[(dataset, seed, split)]
            X = np.asarray([[row[name] for name in 特征名] for row in feature_row["特征"]], dtype=float)
            target = np.mean([np.asarray(f["类别权重"]["alpha值"], dtype=float)
                              for f in target_row["两折"]], axis=0)
            if len(X) != len(target):
                raise ValueError(f"类别数不一致: {dataset} seed{seed} {split}")
            split_data[split] = {"X": X, "target": target, "path": 根目录 / target_row["输入路径"]}
        model = fit_ridge(split_data["Base"]["X"], split_data["Base"]["target"])
        predicted = {split: predict(split_data[split]["X"], model) for split in ("Base", "Novel")}
        scalar = float(split_data["Base"]["target"].mean())
        methods = {}
        for method, alpha_by_split in {
            "固定0.5": {"Base": 0.5, "Novel": 0.5},
            "Base均值标量": {"Base": scalar, "Novel": scalar},
            "文本映射": predicted,
        }.items():
            base = metrics(split_data["Base"]["path"], alpha_by_split["Base"])["准确率"]
            novel = metrics(split_data["Novel"]["path"], alpha_by_split["Novel"])["准确率"]
            methods[method] = {"Base": base, "Novel": novel, "HM": hm(base, novel)}
        results.append({
            "数据集": dataset,
            "种子": seed,
            "指标": methods,
            "Base均值标量": scalar,
            "文本映射Base预测均值": float(predicted["Base"].mean()),
            "文本映射Novel预测均值": float(predicted["Novel"].mean()),
            "Novel预测与诊断上界目标Pearson": corr(predicted["Novel"], split_data["Novel"]["target"]),
            "Novel预测与诊断上界目标MAE": float(np.abs(predicted["Novel"] - split_data["Novel"]["target"]).mean()),
            "岭回归系数": {name: float(model[0][i]) for i, name in enumerate(特征名)},
            "截距": float(model[0][-1]),
        })
    datasets = []
    for dataset in sorted({x["数据集"] for x in results}):
        rows = [x for x in results if x["数据集"] == dataset]
        summary = {method: {metric: float(np.mean([r["指标"][method][metric] for r in rows]))
                            for metric in ("Base", "Novel", "HM")}
                   for method in ("固定0.5", "Base均值标量", "文本映射")}
        datasets.append({
            "数据集": dataset,
            "种子数": len(rows),
            "指标": summary,
            "文本映射相对固定HM增益pp": 100 * (summary["文本映射"]["HM"] - summary["固定0.5"]["HM"]),
            "文本映射相对Base均值标量HM增益pp": 100 * (summary["文本映射"]["HM"] - summary["Base均值标量"]["HM"]),
        })
    mean_fixed = float(np.mean([x["文本映射相对固定HM增益pp"] for x in datasets]))
    mean_scalar = float(np.mean([x["文本映射相对Base均值标量HM增益pp"] for x in datasets]))
    nonworse = sum(x["文本映射相对Base均值标量HM增益pp"] >= -1e-12 for x in datasets)
    checks = {
        "等权平均相对固定HM至少加0.20pp": mean_fixed >= 0.20,
        "等权平均相对Base均值标量HM至少加0.10pp": mean_scalar >= 0.10,
        "至少2个数据集不低于Base均值标量": nonworse >= 2,
    }
    payload = {
        "协议": protocol,
        "协议SHA256": sha(协议路径),
        "文本特征文件": str(args.features),
        "文本特征SHA256": sha(args.features),
        "必要性结果SHA256": sha(必要性路径),
        "逐运行": results,
        "逐数据集": datasets,
        "等权平均文本映射相对固定HM增益pp": mean_fixed,
        "等权平均文本映射相对Base均值标量HM增益pp": mean_scalar,
        "文本映射不低于Base均值标量的数据集数": nonworse,
        "继续条件": checks,
        "通过": all(checks.values()),
    }
    输出目录.mkdir(parents=True, exist_ok=True)
    output = 输出目录 / "结果.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    lines = [
        "# 类别语义可靠性文本迁移探针",
        "",
        f"**结论：{'通过' if payload['通过'] else '未通过'}。** 本结果属于开发探针，Base目标来自已使用过的Base测试输出，不进入论文主表。",
        "",
        "| 数据集 | 固定0.5 HM | Base均值标量HM | 文本映射HM | 映射-固定 Δpp | 映射-标量 Δpp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in datasets:
        m = row["指标"]
        lines.append(f"| {row['数据集']} | {100*m['固定0.5']['HM']:.3f} | {100*m['Base均值标量']['HM']:.3f} | "
                     f"{100*m['文本映射']['HM']:.3f} | {row['文本映射相对固定HM增益pp']:+.3f} | "
                     f"{row['文本映射相对Base均值标量HM增益pp']:+.3f} |")
    lines += ["", f"三数据集等权：相对固定 {mean_fixed:+.4f}pp，相对Base均值标量 {mean_scalar:+.4f}pp；"
              f"{nonworse}/3 数据集不低于标量。", "", "## 冻结判定", ""]
    lines += [f"- {'通过' if value else '未通过'}：{name}" for name, value in checks.items()]
    lines += ["", "Novel图片和标签没有参与文本映射拟合；保存的Novel诊断上界目标只用于计算相关性和MAE。"
              "若未通过，不改变三项文本特征或岭回归设置。", ""]
    (输出目录 / "结果报告.md").write_text("\n".join(lines))
    print(json.dumps({"通过": payload["通过"], "相对固定HM增益pp": mean_fixed,
                      "相对标量HM增益pp": mean_scalar, "结果SHA256": sha(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
