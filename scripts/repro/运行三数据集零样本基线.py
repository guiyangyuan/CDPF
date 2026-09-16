#!/usr/bin/env python3
"""生成或执行扩展数据集 Zero-shot CLIP 基线队列，并从日志独立汇总。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


新增三数据集 = [
    ("Caltech101", "caltech101.yaml"),
    ("OxfordPets", "oxford_pets.yaml"),
    ("Food101", "food101.yaml"),
]
剩余两数据集 = [
    ("StanfordCars", "stanford_cars.yaml"),
    ("SUN397", "sun397.yaml"),
]
套件 = {"新增三数据集": 新增三数据集, "剩余两数据集": 剩余两数据集}
协议数据集 = {
    "投稿Zero-shot-CLIP新增三数据集-v1": 新增三数据集,
    "投稿Zero-shot-CLIP剩余两数据集-v1": 剩余两数据集,
}
分支 = [("base", "基础测试"), ("new", "新类别测试")]
预期CLIP摘要 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
准确率模式 = re.compile(r"^\* accuracy:\s*([0-9.]+)%$", re.MULTILINE)
正确数模式 = re.compile(r"^\* correct:\s*([0-9,]+)$", re.MULTILINE)
总数模式 = re.compile(r"^\* total:\s*([0-9,]+)$", re.MULTILINE)


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


def 调和平均(base: float, novel: float) -> float:
    return 0.0 if base + novel == 0 else 2.0 * base * novel / (base + novel)


def 源码清单(source_root: Path, dataset_rows: list[tuple[str, str]]) -> list[Path]:
    files = [
        Path("train.py"),
        Path("trainers/zsclip.py"),
        Path("trainers/coop.py"),
        Path("configs/repro/零样本基线.yaml"),
    ]
    files.extend(Path("configs/datasets") / filename for _, filename in dataset_rows)
    missing = [str(path) for path in files if not (source_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少执行文件：{missing}")
    return files


def 生成计划(
    source_root: Path,
    data_root: Path,
    output_root: Path,
    python: Path,
    dataset_rows: list[tuple[str, str]] = 新增三数据集,
    protocol: str = "投稿Zero-shot-CLIP新增三数据集-v1",
) -> dict:
    source_root = source_root.resolve()
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    common_config = source_root / "configs/repro/零样本基线.yaml"
    commands = []
    if 协议数据集.get(protocol) != dataset_rows:
        raise ValueError("协议与数据集套件不匹配")
    for dataset, filename in dataset_rows:
        for split, split_name in 分支:
            result_dir = output_root / dataset / split_name
            command = [
                str(python),
                "-B",
                str(source_root / "train.py"),
                "--root",
                str(data_root),
                "--seed",
                "1",
                "--trainer",
                "ZeroshotCLIP",
                "--dataset-config-file",
                str(source_root / "configs/datasets" / filename),
                "--config-file",
                str(common_config),
                "--output-dir",
                str(result_dir),
                "--eval-only",
                "DATASET.NUM_SHOTS",
                "16",
                "DATASET.SUBSAMPLE_CLASSES",
                split,
            ]
            commands.append(
                {
                    "序号": len(commands) + 1,
                    "数据集": dataset,
                    "分支": split,
                    "分支名称": split_name,
                    "输出目录": str(result_dir),
                    "命令": command,
                }
            )
    return {
        "协议": protocol,
        "状态": "待执行",
        "源码目录": str(source_root),
        "数据目录": str(data_root),
        "输出目录": str(output_root),
        "Python": str(python),
        "方法": "ZeroshotCLIP，单一数据集模板，冻结CLIP ViT-B/16",
        "数据集": [name for name, _ in dataset_rows],
        "分支": [split for split, _ in 分支],
        "命令数": len(commands),
        "源码文件SHA256": {path.as_posix(): sha256(source_root / path) for path in 源码清单(source_root, dataset_rows)},
        "预期CLIP_SHA256": 预期CLIP摘要,
        "命令": commands,
    }


def 检查计划身份(plan: dict) -> tuple[Path, Path]:
    source_root = Path(plan["源码目录"])
    output_root = Path(plan["输出目录"])
    dataset_rows = 协议数据集.get(plan.get("协议"))
    if dataset_rows is None or plan.get("命令数") != len(dataset_rows) * len(分支):
        raise ValueError("执行计划身份或命令数错误")
    if plan.get("数据集") != [name for name, _ in dataset_rows]:
        raise ValueError("执行计划数据集顺序错误")
    actual = {path.as_posix(): sha256(source_root / path) for path in 源码清单(source_root, dataset_rows)}
    if actual != plan["源码文件SHA256"]:
        raise ValueError("源码或配置在计划生成后发生变化")
    clip_path = source_root / "clip/ViT-B-16.pt"
    if not clip_path.is_file():
        raise FileNotFoundError(f"缺少CLIP权重：{clip_path}")
    clip_sha = sha256(clip_path)
    if clip_sha != 预期CLIP摘要:
        raise ValueError(f"CLIP权重摘要错误：{clip_sha}")
    return source_root, output_root


def 解析日志(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺少评估日志：{path}")
    text = path.read_text(encoding="utf-8")
    accuracy = 准确率模式.findall(text)
    correct = 正确数模式.findall(text)
    total = 总数模式.findall(text)
    if len(accuracy) != 1 or len(correct) != 1 or len(total) != 1:
        raise ValueError(f"评估指标数量异常：{path}")
    parsed = {
        "准确率": float(accuracy[0]),
        "正确数": int(correct[0].replace(",", "")),
        "总数": int(total[0].replace(",", "")),
        "日志SHA256": sha256(path),
    }
    recomputed = 100.0 * parsed["正确数"] / parsed["总数"]
    if abs(parsed["准确率"] - recomputed) > 0.00051:
        raise ValueError(f"日志准确率不能由正确数复算：{path}")
    parsed["复算准确率"] = recomputed
    return parsed


def 生成结果(plan: dict, status: dict, output_root: Path) -> dict:
    dataset_rows = 协议数据集[plan["协议"]]
    expected_commands = len(dataset_rows) * len(分支)
    if status.get("状态") != "完成" or status.get("已完成命令") != expected_commands:
        raise ValueError("执行状态未完成")
    rows = []
    by_dataset: dict[str, dict[str, dict]] = {}
    for item in plan["命令"]:
        output = Path(item["输出目录"])
        metrics = 解析日志(output / "log.txt")
        if list(output.rglob("*.pth.tar*")):
            raise ValueError(f"零样本评估不应生成训练检查点：{output}")
        row = {"数据集": item["数据集"], "分支": item["分支"], **metrics}
        rows.append(row)
        by_dataset.setdefault(item["数据集"], {})[item["分支"]] = metrics
    if len(rows) != expected_commands or any(set(value) != {"base", "new"} for value in by_dataset.values()):
        raise ValueError("扩展数据集Base/Novel结果不完整")
    summaries = []
    for dataset, _ in dataset_rows:
        base = by_dataset[dataset]["base"]["复算准确率"]
        novel = by_dataset[dataset]["new"]["复算准确率"]
        summaries.append({"数据集": dataset, "Base": base, "Novel": novel, "HM": 调和平均(base, novel)})
    return {
        "状态": "通过",
        "协议": plan["协议"],
        "方法": plan["方法"],
        "结果": summaries,
        "逐分支复算": rows,
        "检查": {
            "执行命令完成": expected_commands,
            "结果分支": expected_commands,
            "训练检查点数量": 0,
            "源码计划前后相同": True,
            "CLIP_SHA256": 预期CLIP摘要,
        },
    }


def 写结果文件(result: dict, output_root: Path) -> None:
    原子写json(output_root / "结果.json", result)
    with (output_root / "结果.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["数据集", "Base", "Novel", "HM"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(result["结果"])
    lines = [
        "# 扩展数据集 Zero-shot CLIP 基线结果",
        "",
        "方法为冻结 OpenAI CLIP ViT-B/16 与各数据集单一标准模板；每个 Base/Novel 分支只评估一次，不训练、不选择种子、不生成检查点。",
        "",
        "| 数据集 | Base | Novel | HM |",
        "|---|---:|---:|---:|",
    ]
    for item in result["结果"]:
        lines.append(f"| {item['数据集']} | {item['Base']:.2f} | {item['Novel']:.2f} | {item['HM']:.2f} |")
    lines.extend(
        [
            "",
            "这些数字是本地同数据划分实测的标准零样本参考，不是论文外部数字。它不使用16-shot训练样本；`NUM_SHOTS=16`只保持数据加载配置与主表一致。",
            "",
            "机器可读证据见[结果JSON](结果.json)、[结果CSV](结果.csv)、[执行状态](执行状态.json)和[冻结计划](执行计划.json)。",
            "",
        ]
    )
    (output_root / "结果报告.md").write_text("\n".join(lines), encoding="utf-8")


def 写文件清单(output_root: Path) -> None:
    target = output_root / "文件清单.txt"
    entries = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file() and item != target):
        entries.append(f"{sha256(path)}  {path.relative_to(output_root).as_posix()}")
    target.write_text("\n".join(entries) + "\n", encoding="utf-8")


def 执行(plan_path: Path) -> int:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source_root, output_root = 检查计划身份(plan)
    expected_commands = plan["命令数"]
    if output_root.exists():
        raise FileExistsError(f"输出目录必须不存在：{output_root}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("没有可用CUDA，拒绝创建输出或执行评估")
    gpu = torch.cuda.get_device_name(0)
    output_root.mkdir(parents=True)
    (output_root / "运行日志").mkdir()
    shutil.copy2(plan_path, output_root / "执行计划.json")
    status = {
        "状态": "运行中",
        "开始时间Unix": time.time(),
        "GPU": gpu,
        "总命令": expected_commands,
        "已完成命令": 0,
        "命令": [],
    }
    原子写json(output_root / "执行状态.json", status)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "4",
            "PYTHONPATH": f"{source_root}:{source_root / 'Dassl.pytorch'}",
        }
    )
    try:
        for item in plan["命令"]:
            started = time.time()
            log_path = output_root / "运行日志" / f"{item['序号']:02d}-{item['数据集']}-{item['分支名称']}.txt"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    item["命令"], cwd=source_root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            record = {
                "序号": item["序号"],
                "数据集": item["数据集"],
                "分支": item["分支"],
                "退出码": completed.returncode,
                "耗时秒": time.time() - started,
                "运行日志": str(log_path),
            }
            status["命令"].append(record)
            if completed.returncode != 0:
                status["状态"] = "失败"
                status["失败命令"] = record
                原子写json(output_root / "执行状态.json", status)
                raise RuntimeError(f"命令失败，见{log_path}")
            status["已完成命令"] += 1
            原子写json(output_root / "执行状态.json", status)
        status["状态"] = "完成"
        status["结束时间Unix"] = time.time()
        status["总耗时秒"] = status["结束时间Unix"] - status["开始时间Unix"]
        原子写json(output_root / "执行状态.json", status)
        result = 生成结果(plan, status, output_root)
        写结果文件(result, output_root)
        写文件清单(output_root)
    except Exception:
        if status.get("状态") == "运行中":
            status["状态"] = "失败"
            status["结束时间Unix"] = time.time()
            原子写json(output_root / "执行状态.json", status)
        raise
    print(json.dumps({"状态": "通过", "GPU": gpu, "输出目录": str(output_root)}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("计划")
    plan_parser.add_argument("--source-root", type=Path, required=True)
    plan_parser.add_argument("--data-root", type=Path, required=True)
    plan_parser.add_argument("--output-root", type=Path, required=True)
    plan_parser.add_argument("--python", type=Path, default=Path(sys.executable))
    plan_parser.add_argument("--plan-path", type=Path, required=True)
    plan_parser.add_argument("--suite", choices=tuple(套件), default="新增三数据集")
    execute_parser = subparsers.add_parser("执行")
    execute_parser.add_argument("--plan-path", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "计划":
        if args.output_root.exists():
            raise FileExistsError(f"计划输出目录必须不存在：{args.output_root}")
        dataset_rows = 套件[args.suite]
        protocol = (
            "投稿Zero-shot-CLIP新增三数据集-v1"
            if args.suite == "新增三数据集"
            else "投稿Zero-shot-CLIP剩余两数据集-v1"
        )
        payload = 生成计划(args.source_root, args.data_root, args.output_root, args.python, dataset_rows, protocol)
        原子写json(args.plan_path, payload)
        print(json.dumps({"状态": "计划完成", "命令数": payload["命令数"], "计划": str(args.plan_path)}, ensure_ascii=False))
        return 0
    return 执行(args.plan_path)


if __name__ == "__main__":
    raise SystemExit(main())
