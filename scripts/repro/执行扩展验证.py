#!/usr/bin/env python3
"""运行冻结的扩展数据集验证；默认只打印计划，--execute 才启动 GPU 命令。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

数据集名称 = {
    "eurosat": "EuroSAT",
    "oxford_flowers": "OxfordFlowers",
    "caltech101": "Caltech101",
    "oxford_pets": "OxfordPets",
    "food101": "Food101",
    "stanford_cars": "StanfordCars",
    "sun397": "SUN397",
}
预期模型源码摘要 = "c1d57b85c0b929441df2c8e70bb12cf9901ba29c19b37716eb9bb9dc73abde5a"


def 构造计划(data_root: Path, output_root: Path, dataset: str, seeds: list[int]) -> dict:
    from scripts.repro import run_pilot

    data_root = data_root.resolve()
    output_root = output_root.resolve()
    基线根目录 = output_root / "基线训练"
    融合根目录 = output_root / "融合评估"
    基线命令 = run_pilot.build_commands(
        data_root, 基线根目录, seeds, 20, ["baseline"],
        device="cuda", python=sys.executable, dataset=dataset, kd_weight=0.0,
    )
    融合命令 = []
    for seed in seeds:
        for split in ("base", "new"):
            融合命令.append([
                sys.executable, str(ROOT / "scripts/repro/evaluate_frozen_fusion.py"),
                "--source-root", str(ROOT),
                "--baseline-dir", str(基线根目录 / "baseline" / f"seed{seed}"),
                "--dataset", dataset,
                "--seed", str(seed),
                "--split", split,
                "--data-root", str(data_root),
                "--output-dir", str(融合根目录 / f"seed{seed}" / split),
                "--expected-source-sha256", 预期模型源码摘要,
            ])
    汇总命令 = [
        sys.executable, str(ROOT / "scripts/repro/汇总扩展验证.py"),
        "--root", str(output_root),
    ]
    return {
        "协议": "冻结均值概率融合扩展验证-v1",
        "数据集": 数据集名称[dataset],
        "数据集键": dataset,
        "种子": seeds,
        "训练轮数": 20,
        "样本数": 16,
        "骨干": "ViT-B/16",
        "主比较": "0.5*softmax(CoOp)+0.5*softmax(冻结均值描述) 对 CoOp",
        "附加报告": ["冻结均值描述", "固定logit均值融合"],
        "门槛": {
            "融合HM严格高于CoOp": True,
            "融合Novel相对CoOp最低允许pp": -0.5,
            "正HM种子比例至少": "三分之二；单种子阶段必须为正",
        },
        "选择限制": "固定alpha=0.5，不搜索权重、温度、描述、轮数或阈值，不因结果修改公式",
        "证据限制": "三个数据集有历史调参或论文成绩，只作为未参与本轮融合规则选择的扩展验证，不称严格盲测",
        "预期模型源码SHA256": 预期模型源码摘要,
        "数据根目录": str(data_root),
        "输出根目录": str(output_root),
        "命令": 基线命令 + 融合命令 + [汇总命令],
        "命令数": len(基线命令) + len(融合命令) + 1,
    }


def 原子写入(path: Path, payload: dict) -> None:
    临时文件 = path.with_name(path.name + ".tmp")
    临时文件.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    临时文件.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(数据集名称), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed not in (1, 2, 3) for seed in args.seeds):
        parser.error("seeds只能是不重复的1、2、3")

    计划 = 构造计划(args.data_root, args.output_root, args.dataset, args.seeds)
    if not args.execute:
        print(json.dumps(计划, ensure_ascii=False, indent=2))
        return 0

    输出根目录 = args.output_root.resolve()
    if 输出根目录.exists():
        raise FileExistsError(f"输出目录必须是全新路径: {输出根目录}")
    if not args.data_root.resolve().is_dir():
        raise FileNotFoundError(f"数据根目录不存在: {args.data_root.resolve()}")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("冻结扩展验证需要GPU，未创建输出目录")
    from scripts.repro.preflight import check
    from trainers.textconsistency_runtime import source_digest
    实际源码摘要 = source_digest()
    if 实际源码摘要 != 预期模型源码摘要:
        raise ValueError(f"模型源码摘要不匹配: {实际源码摘要}")

    数据检查 = check(args.data_root, args.dataset)
    输出根目录.mkdir(parents=True)
    计划["实际模型源码SHA256"] = 实际源码摘要
    计划["开始时间"] = datetime.now(timezone.utc).isoformat()
    原子写入(输出根目录 / "实验协议.json", 计划)
    原子写入(输出根目录 / "数据检查.json", 数据检查)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "Dassl.pytorch"), env.get("PYTHONPATH", "")])
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    状态 = {"状态": "运行中", "已完成命令": 0, "总命令": len(计划["命令"]), "当前命令": None}
    原子写入(输出根目录 / "执行状态.json", 状态)
    for index, command in enumerate(计划["命令"], start=1):
        状态["当前命令"] = index
        原子写入(输出根目录 / "执行状态.json", 状态)
        try:
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        except Exception as exc:
            状态.update({"状态": "失败", "错误": f"{type(exc).__name__}: {exc}"})
            原子写入(输出根目录 / "执行状态.json", 状态)
            raise
        状态["已完成命令"] = index
    状态.update({"状态": "完成", "当前命令": None, "结束时间": datetime.now(timezone.utc).isoformat()})
    原子写入(输出根目录 / "执行状态.json", 状态)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
