#!/usr/bin/env python3
"""冻结的伪新类校准融合留一数据集离线探针。"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn


仓库 = Path(__file__).resolve().parents[3]
准备目录 = Path(__file__).resolve().parent
输出目录 = 仓库 / "reports/experiments/伪新类校准融合离线探针-20260915"
固定种子 = 20260915


输入规格 = [
    {
        "数据集": "DTD",
        "种子": 1,
        "Base": "reports/experiments/置信度门控DTD筛选-20260914/原始结果/DTD/均值描述/种子1/基础测试集/预测.npz",
        "Novel": "reports/experiments/置信度门控DTD筛选-20260914/原始结果/DTD/均值描述/种子1/新类别测试集/预测.npz",
    },
    *[
        {
            "数据集": "UCF101",
            "种子": seed,
            "Base": f"reports/experiments/语义一致性门控UCF101独立验证-20260915/UCF101/种子{seed}/基础测试集/预测.npz",
            "Novel": f"reports/experiments/语义一致性门控UCF101独立验证-20260915/UCF101/种子{seed}/新类别测试集/预测.npz",
        }
        for seed in (1, 2, 3)
    ],
    {
        "数据集": "OxfordPets",
        "种子": 1,
        "Base": "reports/experiments/语义一致性门控OxfordPets独立验证-20260915/基础测试集门控预测.npz",
        "Novel": "reports/experiments/语义一致性门控OxfordPets独立验证-20260915/新类别测试集门控预测.npz",
    },
]


def 哈希(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def 稳定softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x -= x.max(axis=1, keepdims=True)
    x = np.exp(x)
    return x / x.sum(axis=1, keepdims=True)


def 熵(prob: np.ndarray) -> np.ndarray:
    return -(prob * np.log(np.clip(prob, 1e-12, 1.0))).sum(axis=1)


def 间隔(prob: np.ndarray) -> np.ndarray:
    top2 = np.partition(prob, -2, axis=1)[:, -2:]
    return top2.max(axis=1) - top2.min(axis=1)


def 构造特征(ps: np.ndarray, pm: np.ndarray) -> np.ndarray:
    类数 = ps.shape[1]
    混合 = 0.5 * (ps + pm)
    js = 0.5 * (
        (ps * (np.log(np.clip(ps, 1e-12, 1.0)) - np.log(np.clip(混合, 1e-12, 1.0)))).sum(1)
        + (pm * (np.log(np.clip(pm, 1e-12, 1.0)) - np.log(np.clip(混合, 1e-12, 1.0)))).sum(1)
    ) / math.log(2.0)
    return np.stack(
        [
            熵(ps) / math.log(类数),
            熵(pm) / math.log(类数),
            间隔(ps),
            间隔(pm),
            js,
            (ps.argmax(1) == pm.argmax(1)).astype(np.float64),
            ps.max(1) - pm.max(1),
        ],
        axis=1,
    )


def 读取一组(path: Path, 数据集: str, 种子: int, 划分: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        if not {"labels", "S", "M"}.issubset(z.files):
            raise ValueError(f"输入缺少labels/S/M: {path}")
        labels = z["labels"].astype(np.int64)
        s = z["S"].astype(np.float64)
        m = z["M"].astype(np.float64)
    if s.shape != m.shape or s.ndim != 2 or len(labels) != len(s):
        raise ValueError(f"形状不一致: {path}")
    if not np.isfinite(s).all() or not np.isfinite(m).all():
        raise ValueError(f"存在非有限logits: {path}")
    if labels.min() < 0 or labels.max() >= s.shape[1]:
        raise ValueError(f"标签越界: {path}")
    ps, pm = 稳定softmax(s), 稳定softmax(m)
    idx = np.arange(len(labels))
    return {
        "数据集": 数据集,
        "种子": 种子,
        "划分": 划分,
        "路径": str(path.relative_to(仓库)),
        "标签": labels,
        "学生概率": ps,
        "描述概率": pm,
        "特征": 构造特征(ps, pm),
        "学生真类概率": ps[idx, labels],
        "描述真类概率": pm[idx, labels],
        "学生平方和": np.square(ps).sum(1),
        "描述平方和": np.square(pm).sum(1),
        "概率点积": (ps * pm).sum(1),
    }


def 读取输入() -> tuple[list[dict], list[dict]]:
    组 = []
    清单 = []
    for spec in 输入规格:
        for split in ("Base", "Novel"):
            path = 仓库 / spec[split]
            if not path.is_file():
                raise FileNotFoundError(path)
            item = 读取一组(path, spec["数据集"], spec["种子"], split)
            组.append(item)
            清单.append(
                {
                    "路径": spec[split],
                    "数据集": spec["数据集"],
                    "种子": spec["种子"],
                    "划分": split,
                    "样本数": len(item["标签"]),
                    "类别数": item["学生概率"].shape[1],
                    "字节数": path.stat().st_size,
                    "SHA256": 哈希(path),
                }
            )
    return 组, 清单


class 小型门控(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.输入层 = nn.Linear(7, 16)
        self.输出层 = nn.Linear(16, 1)
        nn.init.xavier_uniform_(self.输入层.weight)
        nn.init.zeros_(self.输入层.bias)
        nn.init.zeros_(self.输出层.weight)
        nn.init.zeros_(self.输出层.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.输出层(torch.tanh(self.输入层(x))).squeeze(1))


def 训练权重(训练组: list[dict]) -> np.ndarray:
    数据集种子数 = defaultdict(set)
    for item in 训练组:
        数据集种子数[item["数据集"]].add(item["种子"])
    数据集数 = len(数据集种子数)
    weights = []
    for item in 训练组:
        n = len(item["标签"])
        w = 1.0 / 数据集数 / len(数据集种子数[item["数据集"]]) / 2.0 / n
        weights.append(np.full(n, w, dtype=np.float64))
    result = np.concatenate(weights)
    result /= result.sum()
    return result


def 训练门控(训练组: list[dict]) -> tuple[小型门控, np.ndarray, np.ndarray, dict]:
    torch.manual_seed(固定种子)
    torch.use_deterministic_algorithms(True)
    x_np = np.concatenate([item["特征"] for item in 训练组])
    mean = x_np.mean(0)
    std = x_np.std(0)
    std[std < 1e-8] = 1.0
    x = torch.tensor((x_np - mean) / std, dtype=torch.float64)
    psy = torch.tensor(np.concatenate([item["学生真类概率"] for item in 训练组]), dtype=torch.float64)
    pmy = torch.tensor(np.concatenate([item["描述真类概率"] for item in 训练组]), dtype=torch.float64)
    ssq = torch.tensor(np.concatenate([item["学生平方和"] for item in 训练组]), dtype=torch.float64)
    msq = torch.tensor(np.concatenate([item["描述平方和"] for item in 训练组]), dtype=torch.float64)
    dot = torch.tensor(np.concatenate([item["概率点积"] for item in 训练组]), dtype=torch.float64)
    weight = torch.tensor(训练权重(训练组), dtype=torch.float64)
    model = 小型门控().double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.001)
    轨迹 = []
    for step in range(600):
        g = model(x)
        py = torch.clamp(g * psy + (1.0 - g) * pmy, min=1e-12)
        nll = -torch.log(py)
        brier = (
            g.square() * ssq
            + (1.0 - g).square() * msq
            + 2.0 * g * (1.0 - g) * dot
            - 2.0 * py
            + 1.0
        )
        anchor = (g - 0.5).square()
        loss = torch.sum(weight * (nll + 0.25 * brier + 0.02 * anchor))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0, 99, 199, 399, 599):
            轨迹.append({"步": step + 1, "损失": float(loss.detach())})
    state = {name: value.detach().cpu().numpy().tolist() for name, value in model.state_dict().items()}
    return model, mean, std, {"损失轨迹": 轨迹, "状态": state}


def ece(prob: np.ndarray, labels: np.ndarray) -> float:
    pred = prob.argmax(1)
    conf = prob.max(1)
    edges = np.linspace(0.0, 1.0, 16)
    value = 0.0
    for i in range(15):
        if i == 14:
            mask = (conf >= edges[i]) & (conf <= edges[i + 1])
        else:
            mask = (conf >= edges[i]) & (conf < edges[i + 1])
        if mask.any():
            value += mask.mean() * abs((pred[mask] == labels[mask]).mean() - conf[mask].mean())
    return float(value)


def 指标(prob: np.ndarray, labels: np.ndarray) -> dict:
    idx = np.arange(len(labels))
    return {
        "准确率": float(100.0 * (prob.argmax(1) == labels).mean()),
        "正确数": int((prob.argmax(1) == labels).sum()),
        "样本数": int(len(labels)),
        "NLL": float(-np.log(np.clip(prob[idx, labels], 1e-12, 1.0)).mean()),
        "Brier": float((np.square(prob).sum(1) - 2.0 * prob[idx, labels] + 1.0).mean()),
        "ECE15": ece(prob, labels),
    }


def hm(base: float, novel: float) -> float:
    return float(2.0 * base * novel / (base + novel)) if base + novel else 0.0


def 门控预测(model: 小型门控, mean: np.ndarray, std: np.ndarray, item: dict) -> tuple[np.ndarray, np.ndarray]:
    x = torch.tensor((item["特征"] - mean) / std, dtype=torch.float64)
    with torch.no_grad():
        g = model(x).cpu().numpy()
    prob = g[:, None] * item["学生概率"] + (1.0 - g[:, None]) * item["描述概率"]
    return prob, g


def 主程序() -> None:
    start = time.time()
    输出目录.mkdir(parents=True, exist_ok=True)
    组, 清单 = 读取输入()
    数据集 = sorted({item["数据集"] for item in 组})
    if 数据集 != ["DTD", "OxfordPets", "UCF101"]:
        raise AssertionError(数据集)
    结果 = {"协议": "PNCF-LODO-v1", "固定种子": 固定种子, "数据集": {}, "模型": {}}
    for held_out in 数据集:
        train = [item for item in 组 if item["数据集"] != held_out]
        test = [item for item in 组 if item["数据集"] == held_out]
        model, mean, std, 训练记录 = 训练门控(train)
        结果["模型"][held_out] = {
            "训练数据集": sorted({item["数据集"] for item in train}),
            "标准化均值": mean.tolist(),
            "标准化标准差": std.tolist(),
            **训练记录,
        }
        每种子 = defaultdict(dict)
        split_deltas = []
        for item in test:
            base_prob = 0.5 * (item["学生概率"] + item["描述概率"])
            gate_prob, g = 门控预测(model, mean, std, item)
            baseline = 指标(base_prob, item["标签"])
            gated = 指标(gate_prob, item["标签"])
            student_right = item["学生概率"].argmax(1) == item["标签"]
            desc_right = item["描述概率"].argmax(1) == item["标签"]
            oracle = 100.0 * np.logical_or(student_right, desc_right).mean()
            record = {
                "固定概率融合": baseline,
                "门控融合": gated,
                "差值": {k: gated[k] - baseline[k] for k in ("准确率", "NLL", "Brier", "ECE15")},
                "门控权重": {
                    "均值": float(g.mean()),
                    "标准差": float(g.std()),
                    "最小值": float(g.min()),
                    "最大值": float(g.max()),
                },
                "分支Oracle准确率": float(oracle),
                "仅CoOp正确": int(np.logical_and(student_right, ~desc_right).sum()),
                "仅描述正确": int(np.logical_and(~student_right, desc_right).sum()),
            }
            每种子[item["种子"]][item["划分"]] = record
            split_deltas.append(record["差值"])
        seed_rows = []
        for seed, splits in sorted(每种子.items()):
            if set(splits) != {"Base", "Novel"}:
                raise AssertionError((held_out, seed, splits.keys()))
            b0 = splits["Base"]["固定概率融合"]["准确率"]
            n0 = splits["Novel"]["固定概率融合"]["准确率"]
            b1 = splits["Base"]["门控融合"]["准确率"]
            n1 = splits["Novel"]["门控融合"]["准确率"]
            seed_rows.append(
                {
                    "种子": seed,
                    "固定概率融合HM": hm(b0, n0),
                    "门控融合HM": hm(b1, n1),
                    "HM增益_pp": hm(b1, n1) - hm(b0, n0),
                    "Base增益_pp": b1 - b0,
                    "Novel增益_pp": n1 - n0,
                    "划分": splits,
                }
            )
        结果["数据集"][held_out] = {
            "种子结果": seed_rows,
            "平均HM增益_pp": float(np.mean([r["HM增益_pp"] for r in seed_rows])),
            "平均Base增益_pp": float(np.mean([r["Base增益_pp"] for r in seed_rows])),
            "平均Novel增益_pp": float(np.mean([r["Novel增益_pp"] for r in seed_rows])),
            "平均NLL差值": float(np.mean([r["NLL"] for r in split_deltas])),
            "平均Brier差值": float(np.mean([r["Brier"] for r in split_deltas])),
            "平均ECE15差值": float(np.mean([r["ECE15"] for r in split_deltas])),
        }
    ds_rows = list(结果["数据集"].values())
    summary = {
        "等权平均HM增益_pp": float(np.mean([r["平均HM增益_pp"] for r in ds_rows])),
        "HM不下降数据集数": int(sum(r["平均HM增益_pp"] >= 0.0 for r in ds_rows)),
        "OxfordPets_HM增益_pp": 结果["数据集"]["OxfordPets"]["平均HM增益_pp"],
        "等权平均NLL差值": float(np.mean([r["平均NLL差值"] for r in ds_rows])),
        "等权平均Brier差值": float(np.mean([r["平均Brier差值"] for r in ds_rows])),
        "等权平均ECE15差值": float(np.mean([r["平均ECE15差值"] for r in ds_rows])),
    }
    checks = {
        "平均HM至少加0.20pp": summary["等权平均HM增益_pp"] >= 0.2,
        "至少两个数据集HM不下降": summary["HM不下降数据集数"] >= 2,
        "OxfordPets_HM不下降": summary["OxfordPets_HM增益_pp"] >= 0.0,
        "平均NLL改善": summary["等权平均NLL差值"] < 0.0,
        "平均Brier改善": summary["等权平均Brier差值"] < 0.0,
    }
    summary["判据"] = checks
    summary["通过"] = all(checks.values())
    summary["决定"] = "进入正式GPU实现" if summary["通过"] else "停止固定候选，不实现GPU版本"
    结果["汇总"] = summary
    (输出目录 / "结果.json").write_text(json.dumps(结果, ensure_ascii=False, indent=2) + "\n")
    (输出目录 / "输入清单.json").write_text(
        json.dumps({"文件数": len(清单), "文件": 清单}, ensure_ascii=False, indent=2) + "\n"
    )
    elapsed = time.time() - start
    receipt = {
        "状态": "完成",
        "协议SHA256": 哈希(准备目录 / "实验协议.json"),
        "执行脚本SHA256": 哈希(Path(__file__)),
        "输入文件数": len(清单),
        "新增GPU训练次数": 0,
        "自动重试": 0,
        "运行秒数": elapsed,
        "Python": sys.version,
        "平台": platform.platform(),
        "NumPy": np.__version__,
        "PyTorch": torch.__version__,
    }
    (输出目录 / "运行收据.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# 伪新类校准融合离线探针结果",
        "",
        f"结论：**{'通过开发筛选，可以准备GPU版本' if summary['通过'] else '未通过开发筛选，停止该固定候选'}**。本轮新增GPU训练0次。",
        "",
        "该探针使用此前已经查看过的测试输出，只用于判断门控信号能否跨数据集迁移，不构成投稿独立结果。每个留出数据集的门控只在另外两个数据集上训练。",
        "",
        "| 留出数据集 | Base增益 | Novel增益 | HM增益 | NLL差值 | Brier差值 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("DTD", "UCF101", "OxfordPets"):
        row = 结果["数据集"][name]
        lines.append(
            f"| {name} | {row['平均Base增益_pp']:+.4f} | {row['平均Novel增益_pp']:+.4f} | "
            f"{row['平均HM增益_pp']:+.4f} | {row['平均NLL差值']:+.6f} | {row['平均Brier差值']:+.6f} |"
        )
    lines += [
        "",
        f"三个数据集等权平均HM增益为 **{summary['等权平均HM增益_pp']:+.4f}pp**，"
        f"HM不下降数据集数为 **{summary['HM不下降数据集数']}/3**；平均NLL差值 "
        f"{summary['等权平均NLL差值']:+.6f}，平均Brier差值 {summary['等权平均Brier差值']:+.6f}。",
        "",
        "门控在 DTD 上牺牲 Base、轻微改善 Novel；在 OxfordPets 上方向相反；在 UCF101 三个种子上都明显损伤 Novel。"
        "这说明熵、概率间隔和分支分歧可以改善部分平方误差与ECE，但不能稳定判断哪个分支会在新类别上分类正确。"
        "当前证据不支持把该门控实现成正式训练模块。",
        "",
        "## 冻结判据",
        "",
    ]
    for name, ok in checks.items():
        lines.append(f"- {'通过' if ok else '失败'}：{name}")
    lines += [
        "",
        f"固定决定：**{summary['决定']}**。不根据这批留出结果修改结构、损失、学习率或特征后再次计算。",
        "",
        "下一步不再叠加输出端门控。若继续模型创新，应转向利用 Base 支持图像学习可迁移的类别原型修正，"
        "并在任何新测试结果出现前先用训练/验证类别完成留类验证；另一条低风险路线是保留已经跨三数据集稳定的固定概率融合，"
        "补齐公平对照和更多数据集后按较低门槛期刊投稿。本轮没有启动这两条后续工作。",
        "",
        "完整逐种子、逐划分指标、门控权重、模型参数和损失轨迹见[结果.json](结果.json)。输入哈希见[输入清单.json](输入清单.json)。",
    ]
    (输出目录 / "结果报告.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    主程序()
