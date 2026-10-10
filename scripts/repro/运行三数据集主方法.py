#!/usr/bin/env python3
"""生成或执行冻结主方法扩展队列，兼容新增三数据集和剩余两数据集。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
新增三数据集 = (
    ("Caltech101", "caltech101"),
    ("OxfordPets", "oxford_pets"),
    ("Food101", "food101"),
)
剩余两数据集 = (
    ("StanfordCars", "stanford_cars"),
    ("SUN397", "sun397"),
)
套件 = {"新增三数据集": 新增三数据集, "剩余两数据集": 剩余两数据集}
协议数据集 = {
    "投稿主方法新增三数据集冻结扩展-v1": 新增三数据集,
    "投稿主方法剩余两数据集冻结扩展-v1": 剩余两数据集,
}
数据目录 = {
    "caltech101": "caltech-101",
    "oxford_pets": "oxford_pets",
    "food101": "food-101",
    "stanford_cars": "stanford_cars",
    "sun397": "sun397",
}
种子 = (1, 2, 3)
预期模型源码摘要 = "b90fa7789983c3ea595a837933d23ee5b4a23e59009cd0c5c42addd99e30d5ef"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def 原子写json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def 源码文件(dataset_rows: tuple[tuple[str, str], ...]) -> list[Path]:
    files = [
        Path("scripts/repro/运行三数据集主方法.py"),
        Path("scripts/repro/执行扩展验证.py"),
        Path("scripts/repro/run_pilot.py"),
        Path("scripts/repro/evaluate_frozen_fusion.py"),
        Path("scripts/repro/汇总扩展验证.py"),
        Path("scripts/repro/preflight.py"),
        Path("configs/repro/扩展验证.yaml"),
    ]
    for _, key in dataset_rows:
        files.extend(
            [
                Path("configs/datasets") / f"{key}.yaml",
                Path("datasets") / f"{key}.py",
            ]
        )
    missing = [path.as_posix() for path in files if not (ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少执行文件：{missing}")
    return files


def 生成计划(
    data_root: Path,
    output_root: Path,
    python: Path,
    dataset_rows: tuple[tuple[str, str], ...] = 新增三数据集,
    protocol: str = "投稿主方法新增三数据集冻结扩展-v1",
) -> dict:
    sys.path.insert(0, str(ROOT))
    from scripts.repro import 执行扩展验证

    data_root = data_root.resolve()
    output_root = output_root.resolve()
    runs = []
    if 协议数据集.get(protocol) != dataset_rows:
        raise ValueError("协议与数据集套件不匹配")
    for name, key in dataset_rows:
        dataset_root = output_root / name
        child = 执行扩展验证.构造计划(data_root, dataset_root, key, list(种子))
        command = [
            str(python), "-B", str(ROOT / "scripts/repro/执行扩展验证.py"),
            "--data-root", str(data_root),
            "--output-root", str(dataset_root),
            "--dataset", key,
            "--seeds", *[str(seed) for seed in 种子],
            "--execute",
        ]
        runs.append(
            {
                "序号": len(runs) + 1,
                "数据集": name,
                "数据集键": key,
                "输出目录": str(dataset_root),
                "训练数": 3,
                "内部命令数": child["命令数"],
                "内部计划": child,
                "执行命令": command,
            }
        )
    return {
        "协议": protocol,
        "状态": "待执行",
        "数据集": [name for name, _ in dataset_rows],
        "固定协议": {
            "backbone": "CLIP ViT-B/16",
            "shots": 16,
            "context_tokens": 4,
            "epochs": 20,
            "seeds": list(种子),
            "训练方法": "CoOp，仅base类别交叉熵",
            "主方法": "0.5*softmax(CoOp logits)+0.5*softmax(冻结均值描述 logits)",
            "trainable_parameters": 2048,
            "选择限制": "套件内全部数据集与三种子全部执行，不按Novel结果调参、停跑、删种子或改公式",
        },
        "预期模型源码SHA256": 预期模型源码摘要,
        "源码目录": str(ROOT),
        "数据目录": str(data_root),
        "输出目录": str(output_root),
        "Python": str(python),
        "训练数": len(dataset_rows) * len(种子),
        "内部命令数": sum(run["内部命令数"] for run in runs),
        "源码文件SHA256": {path.as_posix(): sha256(ROOT / path) for path in 源码文件(dataset_rows)},
        "运行": runs,
    }


def 检查计划(plan: dict, require_runtime: bool) -> tuple[Path, Path]:
    dataset_rows = 协议数据集.get(plan.get("协议"))
    if dataset_rows is None:
        raise ValueError("未知主方法协议")
    expected_training = len(dataset_rows) * len(种子)
    expected_commands = len(dataset_rows) * 16
    if plan.get("训练数") != expected_training or plan.get("内部命令数") != expected_commands:
        raise ValueError("计划身份、训练数或命令数错误")
    if plan.get("数据集") != [name for name, _ in dataset_rows]:
        raise ValueError("数据集顺序被修改")
    actual = {path.as_posix(): sha256(ROOT / path) for path in 源码文件(dataset_rows)}
    if actual != plan.get("源码文件SHA256"):
        raise ValueError("计划生成后执行代码或配置发生变化")
    data_root = Path(plan["数据目录"])
    output_root = Path(plan["输出目录"])
    if require_runtime:
        if not data_root.is_dir():
            raise FileNotFoundError(f"数据目录不存在：{data_root}")
        expected = [数据目录[key] for _, key in dataset_rows]
        missing = [name for name in expected if not (data_root / name).is_dir()]
        if missing:
            raise FileNotFoundError(f"缺少数据集目录：{missing}")
    return data_root, output_root


def 执行(plan_path: Path) -> int:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _, output_root = 检查计划(plan, require_runtime=True)
    if output_root.exists():
        raise FileExistsError(f"输出目录必须不存在：{output_root}")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("没有可用CUDA，拒绝创建输出或执行训练")
    output_root.mkdir(parents=True)
    shutil.copy2(plan_path, output_root / "执行计划.json")
    status = {
        "状态": "运行中",
        "协议": plan["协议"],
        "GPU": torch.cuda.get_device_name(0),
        "开始时间Unix": time.time(),
        "总数据集": len(plan["数据集"]),
        "已完成数据集": 0,
        "运行": [],
    }
    原子写json(output_root / "执行状态.json", status)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "4",
            "PYTHONPATH": f"{ROOT}:{ROOT / 'Dassl.pytorch'}",
        }
    )
    try:
        for run in plan["运行"]:
            started = time.time()
            log_path = output_root / f"{run['序号']:02d}-{run['数据集']}-完整日志.txt"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    run["执行命令"], cwd=ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT, check=False,
                )
            record = {
                "数据集": run["数据集"],
                "退出码": completed.returncode,
                "耗时秒": time.time() - started,
                "日志": str(log_path),
            }
            status["运行"].append(record)
            if completed.returncode != 0:
                status.update({"状态": "失败", "失败数据集": run["数据集"]})
                原子写json(output_root / "执行状态.json", status)
                raise RuntimeError(f"{run['数据集']}执行失败，见{log_path}")
            status["已完成数据集"] += 1
            原子写json(output_root / "执行状态.json", status)
        status.update({"状态": "完成", "结束时间Unix": time.time()})
        status["总耗时秒"] = status["结束时间Unix"] - status["开始时间Unix"]
        原子写json(output_root / "执行状态.json", status)
    except Exception:
        if status.get("状态") == "运行中":
            status.update({"状态": "失败", "结束时间Unix": time.time()})
            原子写json(output_root / "执行状态.json", status)
        raise
    print(json.dumps({"状态": "完成", "数据集数": len(plan["数据集"]), "训练数": plan["训练数"], "GPU": status["GPU"]}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("计划")
    plan_parser.add_argument("--data-root", type=Path, required=True)
    plan_parser.add_argument("--output-root", type=Path, required=True)
    plan_parser.add_argument("--python", type=Path, default=Path(sys.executable))
    plan_parser.add_argument("--plan-path", type=Path, required=True)
    plan_parser.add_argument("--suite", choices=tuple(套件), default="新增三数据集")
    run_parser = subparsers.add_parser("执行")
    run_parser.add_argument("--plan-path", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "计划":
        if args.output_root.exists():
            raise FileExistsError(f"计划输出目录必须不存在：{args.output_root}")
        dataset_rows = 套件[args.suite]
        protocol = (
            "投稿主方法新增三数据集冻结扩展-v1"
            if args.suite == "新增三数据集"
            else "投稿主方法剩余两数据集冻结扩展-v1"
        )
        plan = 生成计划(args.data_root, args.output_root, args.python, dataset_rows, protocol)
        原子写json(args.plan_path, plan)
        print(json.dumps({"状态": "计划完成", "训练数": plan["训练数"], "内部命令数": plan["内部命令数"]}, ensure_ascii=False))
        return 0
    return 执行(args.plan_path)


if __name__ == "__main__":
    raise SystemExit(main())
