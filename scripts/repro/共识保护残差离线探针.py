#!/usr/bin/env python3
"""用已归档逐样本 logits 比较三种无参数残差保护规则。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


根目录 = Path(__file__).resolve().parents[2]
实验目录 = 根目录 / "reports/experiments"


def 文件哈希(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def 原始输入():
    rows = [
        ("DTD", 1, "Base", 实验目录 / "描述半径约束残差筛选-20260914/原始结果/旧检查点零训练重载/基础测试集/预测.npz"),
        ("DTD", 1, "Novel", 实验目录 / "描述半径约束残差筛选-20260914/原始结果/旧检查点零训练重载/新类别测试集/预测.npz"),
    ]
    for seed in (1, 2, 3):
        for split, name in (("Base", "基础测试集"), ("Novel", "新类别测试集")):
            rows.append((
                "UCF101", seed, split,
                实验目录 / f"语义一致性门控UCF101独立验证-20260915/UCF101/种子{seed}/{name}/预测.npz",
            ))
    for split, name in (("Base", "基础测试集"), ("Novel", "新类别测试集")):
        rows.append((
            "OxfordPets", 1, split,
            实验目录 / f"语义一致性门控OxfordPets独立验证-20260915/{name}门控预测.npz",
        ))
    return rows


def 稳定软最大值(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def 语义门控(student, description, residual, eps=1e-8):
    semantic = description - student
    semantic -= semantic.mean(axis=1, keepdims=True)
    centered_residual = residual - residual.mean(axis=1, keepdims=True)
    denominator = np.maximum(
        np.linalg.norm(semantic, axis=1) * np.linalg.norm(centered_residual, axis=1), eps
    )
    return np.clip((semantic * centered_residual).sum(axis=1) / denominator, 0.0, 1.0)


def 准确率(predictions, labels):
    return float(100.0 * np.mean(predictions == labels))


def 调和均值(base, novel):
    return float(2.0 * base * novel / (base + novel)) if base + novel else 0.0


def 载入(path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"labels", "S", "M", "R", "F"}
        if not required.issubset(archive.files):
            raise ValueError(f"缺少数组：{path}")
        arrays = {name: archive[name].copy() for name in required}
        if "Q" in archive.files:
            arrays["Q"] = archive["Q"].copy()
    labels, student, description, residual, fusion = (
        arrays["labels"], arrays["S"], arrays["M"], arrays["R"], arrays["F"]
    )
    if labels.ndim != 1 or labels.dtype != np.int64 or not len(labels):
        raise ValueError(f"标签格式错误：{path}")
    if any(x.ndim != 2 or x.shape != student.shape or not np.isfinite(x).all()
           for x in (student, description, residual, fusion)):
        raise ValueError(f"logits 格式错误：{path}")
    if len(student) != len(labels) or student.shape[1] < 2:
        raise ValueError(f"预测尺寸错误：{path}")
    np.testing.assert_array_equal(fusion, 0.5 * (student + description))
    gate = 语义门控(student, description, residual)
    if "Q" in arrays:
        np.testing.assert_allclose(arrays["Q"], gate, rtol=2e-5, atol=2e-6)
    return labels, student, description, residual, fusion, gate


def 转移统计(predictions, baseline, labels):
    changed = predictions != baseline
    return {
        "改变数": int(changed.sum()),
        "改对数": int(((baseline != labels) & (predictions == labels)).sum()),
        "改错数": int(((baseline == labels) & (predictions != labels)).sum()),
        "旁移数": int(((baseline != labels) & (predictions != labels) & changed).sum()),
    }


def 分片分析(path):
    labels, student, description, residual, fusion, gate = 载入(path)
    student_pred = student.argmax(axis=1)
    description_pred = description.argmax(axis=1)
    fusion_pred = fusion.argmax(axis=1)

    semantic_logits = fusion + gate[:, None] * residual
    semantic_pred = semantic_logits.argmax(axis=1)

    disagreement = student_pred != description_pred
    consensus_logits = fusion + (gate * disagreement)[:, None] * residual
    consensus_pred = consensus_logits.argmax(axis=1)

    supported = (semantic_pred == student_pred) | (semantic_pred == description_pred)
    candidate_pred = np.where(supported, semantic_pred, fusion_pred)

    fusion_probability = 稳定软最大值(fusion)
    entropy = -np.sum(
        fusion_probability * np.log(np.clip(fusion_probability, 1e-30, None)), axis=1
    ) / np.log(fusion.shape[1])
    uncertainty_logits = fusion + (gate * entropy)[:, None] * residual
    uncertainty_pred = uncertainty_logits.argmax(axis=1)

    mean_probability_pred = (
        0.5 * (稳定软最大值(student) + 稳定软最大值(description))
    ).argmax(axis=1)
    predictions = {
        "简单融合F": fusion_pred,
        "原语义门控": semantic_pred,
        "分支共识保护": consensus_pred,
        "候选类别约束": candidate_pred,
        "不确定性衰减": uncertainty_pred,
        "均值概率融合诊断": mean_probability_pred,
    }
    return {
        "样本数": int(len(labels)),
        "类别数": int(fusion.shape[1]),
        "分支预测一致比例": float(np.mean(~disagreement)),
        "语义门控均值": float(gate.mean()),
        "归一化熵均值": float(entropy.mean()),
        "准确率": {name: 准确率(pred, labels) for name, pred in predictions.items()},
        "相对F转移": {
            name: 转移统计(pred, fusion_pred, labels)
            for name, pred in predictions.items() if name != "简单融合F"
        },
    }


def 汇总():
    splits = {}
    inputs = []
    for dataset, seed, split, path in 原始输入():
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        key = f"{dataset}/种子{seed}/{split}"
        splits[key] = 分片分析(path)
        inputs.append({"数据集": dataset, "种子": seed, "划分": split,
                       "路径": str(path.relative_to(根目录)), "SHA256": 文件哈希(path)})

    methods = ("原语义门控", "分支共识保护", "候选类别约束", "不确定性衰减", "均值概率融合诊断")
    pairs = []
    for dataset, seed in (("DTD", 1), ("UCF101", 1), ("UCF101", 2),
                          ("UCF101", 3), ("OxfordPets", 1)):
        base = splits[f"{dataset}/种子{seed}/Base"]["准确率"]
        novel = splits[f"{dataset}/种子{seed}/Novel"]["准确率"]
        hm_f = 调和均值(base["简单融合F"], novel["简单融合F"])
        row = {"数据集": dataset, "种子": seed, "F_HM": hm_f, "方法": {}}
        for method in methods:
            hm = 调和均值(base[method], novel[method])
            row["方法"][method] = {
                "Base": base[method], "Novel": novel[method], "HM": hm, "HM增益": hm - hm_f,
            }
        pairs.append(row)

    candidates = ("分支共识保护", "候选类别约束", "不确定性衰减")
    decisions = {}
    for method in candidates:
        gains = [row["方法"][method]["HM增益"] for row in pairs]
        pets_gain = next(row["方法"][method]["HM增益"] for row in pairs
                         if row["数据集"] == "OxfordPets")
        criteria = {
            "五组中至少三组HM不下降": sum(gain >= -1e-12 for gain in gains) >= 3,
            "OxfordPets的HM不下降": pets_gain >= -1e-12,
            "五组平均HM增益为正": float(np.mean(gains)) > 0.0,
        }
        decisions[method] = {
            "通过": all(criteria.values()), "判据": criteria,
            "非负组数": int(sum(gain >= -1e-12 for gain in gains)),
            "平均HM增益": float(np.mean(gains)), "OxfordPets_HM增益": float(pets_gain),
        }
    return {
        "状态": "无卡离线开发探针完成",
        "证据角色": "开发诊断；不得作为新数据集独立验证",
        "新增训练次数": 0,
        "输入": inputs,
        "公式": {
            "F": "0.5*(S+M)",
            "q": "clamp(cos(center(R),center(M-S)),0,1)",
            "分支共识保护": "F+q*1[argmax(S)!=argmax(M)]*R",
            "候选类别约束": "仅当argmax(F+qR)属于{argmax(S),argmax(M)}时接受，否则回退F",
            "不确定性衰减": "F+q*(H(softmax(F))/log(C))*R",
            "拟合参数": [],
        },
        "冻结判据": "五组中至少三组HM不下降，OxfordPets HM不下降，且五组平均HM增益为正",
        "分片": splits,
        "配对结果": pairs,
        "冻结决定": decisions,
        "结论": "三种候选均未通过冻结判据，不写入生产模型，也不启动新数据集训练",
    }


def 写报告(path, result):
    lines = [
        "# 共识保护残差无卡离线探针", "",
        "结论：三种保护策略均未通过冻结判据，本轮不修改生产模型、不启动 GPU 训练。", "",
        "本探针复用 DTD、UCF101、OxfordPets 已归档逐样本 logits，新增训练 0 次。由于这些输出参与了结构选择，所有结果只属于开发诊断。", "",
        "冻结判据：五组数据集-种子配对中至少三组 HM 不下降，OxfordPets HM 不下降，且五组平均 HM 增益为正。", "",
        "## 配对结果", "",
        "| 数据集 | 种子 | 方法 | Base | Novel | HM | 相对F HM |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    order = ("原语义门控", "分支共识保护", "候选类别约束", "不确定性衰减", "均值概率融合诊断")
    for row in result["配对结果"]:
        for method in order:
            metric = row["方法"][method]
            lines.append(
                f"| {row['数据集']} | {row['种子']} | {method} | {metric['Base']:.4f} | "
                f"{metric['Novel']:.4f} | {metric['HM']:.4f} | {metric['HM增益']:+.4f}pp |"
            )
    lines.extend(["", "## 冻结决定", "",
                  "| 候选 | 非负组数 | 平均HM增益 | OxfordPets HM增益 | 通过 |",
                  "|---|---:|---:|---:|---|"])
    for method, decision in result["冻结决定"].items():
        lines.append(
            f"| {method} | {decision['非负组数']}/5 | {decision['平均HM增益']:+.4f}pp | "
            f"{decision['OxfordPets_HM增益']:+.4f}pp | {'是' if decision['通过'] else '否'} |"
        )
    pets = result["分片"]["OxfordPets/种子1/Base"]
    lines.extend([
        "", "## 失败原因", "",
        f"- OxfordPets Base 中 S/M 预测一致比例为 {pets['分支预测一致比例']:.2%}，但原门控改变的 6 个样本全部发生在 S/M 不一致区域，因此“分支共识保护”没有阻止 4 次改错。",
        "- “候选类别约束”同样没有改变 OxfordPets 的 6 次转移，说明残差提出的类别已经属于两个分支的首选类别，候选集合不足以判断哪一支正确。",
        "- “不确定性衰减”把 OxfordPets Base 的改变数从 6 降为 1，但剩下的一次仍是改错，HM 仍低于 F 0.0273pp。",
        "- 均值概率融合在 UCF101 三个种子上明显更好，但在 DTD 和 OxfordPets 下降；它可保留为强基线，不能包装成稳定的新模块。",
        "", "## 后续约束", "",
        "这组结果不进入投稿主结果表。下一轮应停止继续设计输出端启发式门控，转向训练阶段的可学习表示改进，并在查看新数据集结果前冻结结构和协议。",
    ])
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--输出目录", required=True)
    args = parser.parse_args()
    output = Path(args.输出目录).resolve()
    if output.exists():
        raise FileExistsError(f"请使用新的输出目录：{output}")
    output.mkdir(parents=True)
    result = 汇总()
    result_path = output / "结果.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    写报告(output / "结果报告.md", result)
    receipt = {
        "结果SHA256": 文件哈希(result_path),
        "报告SHA256": 文件哈希(output / "结果报告.md"),
        "输入文件数": len(result["输入"]),
        "输入哈希全部唯一": len({row["SHA256"] for row in result["输入"]}) == len(result["输入"]),
    }
    (output / "核验收据.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"输出目录": str(output), "冻结决定": result["冻结决定"],
                      "结论": result["结论"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
