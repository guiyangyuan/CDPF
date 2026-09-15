#!/usr/bin/env python3
"""对既有S/M逐样本logits做类别特异融合权重的两折必要性上界检验。"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


根目录 = Path(__file__).resolve().parents[3]
输出目录 = 根目录 / "reports/experiments/类别语义可靠性必要性检验-20260915"
协议路径 = Path(__file__).with_name("必要性检验协议.json")

输入规格 = [
    ("DTD", 1, "Base", "reports/experiments/置信度门控DTD筛选-20260914/原始结果/DTD/均值描述/种子1/基础测试集/预测.npz"),
    ("DTD", 1, "Novel", "reports/experiments/置信度门控DTD筛选-20260914/原始结果/DTD/均值描述/种子1/新类别测试集/预测.npz"),
    ("UCF101", 1, "Base", "reports/experiments/语义一致性门控UCF101独立验证-20260915/UCF101/种子1/基础测试集/预测.npz"),
    ("UCF101", 1, "Novel", "reports/experiments/语义一致性门控UCF101独立验证-20260915/UCF101/种子1/新类别测试集/预测.npz"),
    ("UCF101", 2, "Base", "reports/experiments/语义一致性门控UCF101独立验证-20260915/UCF101/种子2/基础测试集/预测.npz"),
    ("UCF101", 2, "Novel", "reports/experiments/语义一致性门控UCF101独立验证-20260915/UCF101/种子2/新类别测试集/预测.npz"),
    ("UCF101", 3, "Base", "reports/experiments/语义一致性门控UCF101独立验证-20260915/UCF101/种子3/基础测试集/预测.npz"),
    ("UCF101", 3, "Novel", "reports/experiments/语义一致性门控UCF101独立验证-20260915/UCF101/种子3/新类别测试集/预测.npz"),
    ("OxfordPets", 1, "Base", "reports/experiments/语义一致性门控OxfordPets独立验证-20260915/基础测试集门控预测.npz"),
    ("OxfordPets", 1, "Novel", "reports/experiments/语义一致性门控OxfordPets独立验证-20260915/新类别测试集门控预测.npz"),
]


def 文件哈希(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def 稳定两折(dataset: str, seed: int, split: str, labels: np.ndarray) -> np.ndarray:
    folds = np.empty(len(labels), dtype=np.int8)
    for c in range(int(labels.max()) + 1):
        idx = np.flatnonzero(labels == c)
        assert len(idx) >= 2
        ordered = sorted(idx.tolist(), key=lambda row: hashlib.sha256(
            f"{dataset}|{seed}|{split}|{c}|{row}".encode()).digest())
        for rank, row in enumerate(ordered):
            folds[row] = rank % 2
    return folds


def softmax(z: np.ndarray) -> np.ndarray:
    work = z.astype(np.float64, copy=False)
    work = work - work.max(axis=1, keepdims=True)
    exp = np.exp(work)
    return exp / exp.sum(axis=1, keepdims=True)


def 指标(S: np.ndarray, M: np.ndarray, y: np.ndarray, alpha: np.ndarray | float) -> dict:
    z = S + (M - S) * alpha
    p = softmax(z)
    nll = -np.log(p[np.arange(len(y)), y].clip(1e-300)).mean()
    return {"准确率": float((z.argmax(1) == y).mean()), "NLL": float(nll)}


def 训练标量(S: np.ndarray, M: np.ndarray, y: np.ndarray) -> float:
    best = None
    for value in np.linspace(0.0, 1.0, 101):
        nll = 指标(S, M, y, float(value))["NLL"]
        candidate = (nll, abs(float(value) - 0.5), float(value))
        if best is None or candidate < best:
            best = candidate
    return best[2]


def 训练类别权重(S: np.ndarray, M: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    S = S.astype(np.float64, copy=False)
    M = M.astype(np.float64, copy=False)
    d = M - S
    classes = S.shape[1]
    alpha = np.full(classes, 0.5, dtype=np.float64)
    m = np.zeros_like(alpha)
    v = np.zeros_like(alpha)
    onehot = np.zeros_like(S)
    onehot[np.arange(len(y)), y] = 1.0
    initial = 指标(S, M, y, alpha)["NLL"]
    for step in range(1, 401):
        p = softmax(S + d * alpha[None, :])
        grad = ((p - onehot) * d).mean(axis=0)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * np.square(grad)
        mhat = m / (1.0 - 0.9 ** step)
        vhat = v / (1.0 - 0.999 ** step)
        alpha = np.clip(alpha - 0.02 * mhat / (np.sqrt(vhat) + 1e-8), 0.0, 1.0)
    final = 指标(S, M, y, alpha)["NLL"]
    if not np.isfinite(alpha).all() or final > initial + 1e-10:
        raise RuntimeError(f"类别权重优化异常: initial={initial}, final={final}")
    return alpha, {"训练初始NLL": initial, "训练最终NLL": final}


def hm(base: float, novel: float) -> float:
    return 0.0 if base + novel == 0 else 2.0 * base * novel / (base + novel)


def 加载一组(dataset: str, seed: int, split: str, relative: str) -> dict:
    path = 根目录 / relative
    with np.load(path, allow_pickle=False) as payload:
        labels = payload["labels"].astype(np.int64)
        S = payload["S"].astype(np.float64)
        M = payload["M"].astype(np.float64)
        if "F" in payload.files:
            error = float(np.max(np.abs(payload["F"].astype(np.float64) - 0.5 * (S + M))))
            if error > 1e-5:
                raise ValueError(f"{relative} 的F公式不一致: {error}")
    if S.shape != M.shape or S.shape[0] != len(labels) or S.shape[1] != int(labels.max()) + 1:
        raise ValueError(f"{relative} 形状或标签范围不合法")
    if not np.isfinite(S).all() or not np.isfinite(M).all():
        raise ValueError(f"{relative} 含非有限数")
    folds = 稳定两折(dataset, seed, split, labels)
    fold_results = []
    for eval_fold in (0, 1):
        train = folds != eval_fold
        test = folds == eval_fold
        scalar = 训练标量(S[train], M[train], labels[train])
        classwise, trace = 训练类别权重(S[train], M[train], labels[train])
        fold_results.append({
            "评估折": eval_fold,
            "训练样本数": int(train.sum()),
            "评估样本数": int(test.sum()),
            "固定0.5": 指标(S[test], M[test], labels[test], 0.5),
            "训练折标量": {**指标(S[test], M[test], labels[test], scalar), "alpha": scalar},
            "类别权重": {
                **指标(S[test], M[test], labels[test], classwise[None, :]),
                "alpha值": classwise.tolist(),
                "alpha均值": float(classwise.mean()),
                "alpha标准差": float(classwise.std()),
                "alpha最小值": float(classwise.min()),
                "alpha最大值": float(classwise.max()),
                **trace,
            },
        })
    return {
        "数据集": dataset,
        "种子": seed,
        "划分": split,
        "类别数": int(S.shape[1]),
        "样本数": int(len(labels)),
        "输入路径": relative,
        "输入SHA256": 文件哈希(path),
        "两折": fold_results,
    }


def 聚合(rows: list[dict]) -> dict:
    by_run = {}
    for row in rows:
        by_run.setdefault((row["数据集"], row["种子"]), {})[row["划分"]] = row
    run_results = []
    for (dataset, seed), splits in sorted(by_run.items()):
        if set(splits) != {"Base", "Novel"}:
            raise ValueError(f"缺少Base或Novel: {dataset} seed{seed}")
        fold_rows = []
        for fold in (0, 1):
            record = {"评估折": fold}
            for method in ("固定0.5", "训练折标量", "类别权重"):
                b = splits["Base"]["两折"][fold][method]["准确率"]
                n = splits["Novel"]["两折"][fold][method]["准确率"]
                record[method] = {"Base": b, "Novel": n, "HM": hm(b, n)}
            fold_rows.append(record)
        methods = {}
        for method in ("固定0.5", "训练折标量", "类别权重"):
            methods[method] = {metric: float(np.mean([x[method][metric] for x in fold_rows]))
                               for metric in ("Base", "Novel", "HM")}
        run_results.append({"数据集": dataset, "种子": seed, "两折配对": fold_rows, "两折均值": methods})

    dataset_results = []
    for dataset in sorted({x["数据集"] for x in run_results}):
        selected = [x for x in run_results if x["数据集"] == dataset]
        methods = {}
        for method in ("固定0.5", "训练折标量", "类别权重"):
            methods[method] = {metric: float(np.mean([x["两折均值"][method][metric] for x in selected]))
                               for metric in ("Base", "Novel", "HM")}
        delta_fixed = 100.0 * (methods["类别权重"]["HM"] - methods["固定0.5"]["HM"])
        delta_scalar = 100.0 * (methods["类别权重"]["HM"] - methods["训练折标量"]["HM"])
        dataset_results.append({
            "数据集": dataset,
            "种子数": len(selected),
            "指标": methods,
            "类别权重相对固定HM增益pp": delta_fixed,
            "类别权重相对标量HM增益pp": delta_scalar,
        })
    mean_fixed = float(np.mean([x["类别权重相对固定HM增益pp"] for x in dataset_results]))
    mean_scalar = float(np.mean([x["类别权重相对标量HM增益pp"] for x in dataset_results]))
    nonworse = sum(x["类别权重相对标量HM增益pp"] >= -1e-12 for x in dataset_results)
    checks = {
        "等权平均相对固定HM至少加0.30pp": mean_fixed >= 0.30,
        "等权平均相对标量HM至少加0.15pp": mean_scalar >= 0.15,
        "至少2个数据集不低于标量": nonworse >= 2,
    }
    return {
        "逐运行": run_results,
        "逐数据集": dataset_results,
        "等权平均类别权重相对固定HM增益pp": mean_fixed,
        "等权平均类别权重相对标量HM增益pp": mean_scalar,
        "类别权重不低于标量的数据集数": nonworse,
        "继续条件": checks,
        "通过": all(checks.values()),
    }


def 报告文本(result: dict) -> str:
    lines = [
        "# 类别语义可靠性加权：必要性上界检验",
        "",
        f"**冻结结论：{'通过，允许进入文本特征迁移阶段' if result['汇总']['通过'] else '未通过，停止该方向且不开GPU'}。**",
        "",
        "本检验只回答一个问题：同一数据集内，不同类别采用不同的学生/描述融合权重，是否比固定0.5和训练折选出的单一标量更好。每个权重只在一半图片上拟合、在另一半图片上评估，并双向交换。Novel标签参与了这个诊断，因此这是上界，不是可部署方法，也不进入论文主结果表。",
        "",
        "| 数据集 | 固定0.5 HM | 标量HM | 类别权重HM | 类别-固定 Δpp | 类别-标量 Δpp |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["汇总"]["逐数据集"]:
        m = row["指标"]
        lines.append(
            f"| {row['数据集']} | {100*m['固定0.5']['HM']:.3f} | {100*m['训练折标量']['HM']:.3f} | "
            f"{100*m['类别权重']['HM']:.3f} | {row['类别权重相对固定HM增益pp']:+.3f} | "
            f"{row['类别权重相对标量HM增益pp']:+.3f} |")
    alpha_summary = []
    for dataset in sorted({x["数据集"] for x in result["逐划分"]}):
        record = {"数据集": dataset}
        for split in ("Base", "Novel"):
            values = [fold["类别权重"]["alpha均值"]
                      for row in result["逐划分"] if row["数据集"] == dataset and row["划分"] == split
                      for fold in row["两折"]]
            record[split] = float(np.mean(values))
        alpha_summary.append(record)
    lines += [
        "",
        f"三个数据集等权平均：类别权重相对固定0.5为 {result['汇总']['等权平均类别权重相对固定HM增益pp']:+.4f}pp，"
        f"相对训练折标量为 {result['汇总']['等权平均类别权重相对标量HM增益pp']:+.4f}pp；"
        f"{result['汇总']['类别权重不低于标量的数据集数']}/3 数据集不低于标量。",
        "",
        "| 数据集 | Base类别alpha均值 | Novel类别alpha均值 |",
        "|---|---:|---:|",
        *[f"| {row['数据集']} | {row['Base']:.3f} | {row['Novel']:.3f} |" for row in alpha_summary],
        "",
        "DTD和UCF101存在明显的Base/Novel权重分布差异，直接把Base平均权重搬到Novel并不可行。"
        "后续只有文本特征能够跨类别预测这种差异，才允许继续。",
        "",
        "## 冻结判定",
        "",
    ]
    for name, passed in result["汇总"]["继续条件"].items():
        lines.append(f"- {'通过' if passed else '未通过'}：{name}")
    lines += [
        "",
        "若三项都通过，下一步只允许从冻结文本嵌入提取类内一致性、类别名对齐和跨类间隔，用Base类学习低容量可靠性映射，再在Novel类零标签迁移。若未通过，不改变划分、优化器或阈值追分。",
        "",
        "## 完整性",
        "",
        f"- 输入文件：{len(result['输入文件'])} 个，均记录SHA256。",
        "- 每个类别至少两张图片；S/M/F公式、标签范围、数值有限性和优化前后训练NLL均自动检查。",
        "- 机器可读的逐折、逐种子、逐数据集结果见 `结果.json`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    protocol = json.loads(协议路径.read_text())
    rows = [加载一组(*spec) for spec in 输入规格]
    result = {
        "协议": protocol,
        "协议SHA256": 文件哈希(协议路径),
        "输入文件": [{k: row[k] for k in ("数据集", "种子", "划分", "输入路径", "输入SHA256", "样本数", "类别数")}
                     for row in rows],
        "逐划分": rows,
        "汇总": 聚合(rows),
    }
    输出目录.mkdir(parents=True, exist_ok=True)
    result_path = 输出目录 / "结果.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    (输出目录 / "结果报告.md").write_text(报告文本(result))
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    print(json.dumps({
        "通过": result["汇总"]["通过"],
        "相对固定HM增益pp": result["汇总"]["等权平均类别权重相对固定HM增益pp"],
        "相对标量HM增益pp": result["汇总"]["等权平均类别权重相对标量HM增益pp"],
        "结果SHA256": digest,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
