#!/usr/bin/env python3
"""生成、执行和汇总五数据集 CoCoOp 标准基线队列。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


数据集 = [
    ("UCF101", "ucf101.yaml", "ucf101", "核心三数据集"),
    ("DTD", "dtd.yaml", "dtd", "核心三数据集"),
    ("FGVC-Aircraft", "fgvc_aircraft.yaml", "fgvc_aircraft", "核心三数据集"),
    ("EuroSAT", "eurosat.yaml", "eurosat", "扩展两数据集"),
    ("OxfordFlowers", "oxford_flowers.yaml", "oxford_flowers", "扩展两数据集"),
]
种子 = (1, 2, 3)
协议名称 = "投稿CoCoOp五数据集标准基线-v1"
上游提交 = "ff61507c790454bce7c5052c3ac39e60772f1f89"
上游训练器摘要 = "90c7b866d59bce0fd825fa099ab03606d72ef9965bbf5d9cee20729728d1bc84"
上游配置摘要 = "3294d36c623187c137a2bf43ba6eefd7ccbf50d14b7183fae95731dcb2aa44c1"
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


def 源码清单(source_root: Path) -> list[Path]:
    files = [
        Path("train.py"),
        Path("trainers/cocoop.py"),
        Path("clip/model.py"),
        Path("clip/clip.py"),
        Path("configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml"),
        Path("Dassl.pytorch/dassl/engine/trainer.py"),
        Path("Dassl.pytorch/dassl/utils/torchtools.py"),
        Path("scripts/repro/运行联合上下文提示基线.py"),
    ]
    for _, filename, module_name, _ in 数据集:
        files.append(Path("configs/datasets") / filename)
        files.append(Path("datasets") / f"{module_name}.py")
    missing = [path.as_posix() for path in files if not (source_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少执行源码：{missing}")
    return files


def 固定选项() -> list[str]:
    return [
        "DATASET.NUM_SHOTS", "16",
        "TRAINER.COCOOP.N_CTX", "4",
        "TRAINER.COCOOP.CTX_INIT", "a photo of a",
        "TRAINER.COCOOP.PREC", "fp16",
        "DATALOADER.TRAIN_X.BATCH_SIZE", "1",
        "DATALOADER.TEST.BATCH_SIZE", "100",
        "DATALOADER.NUM_WORKERS", "8",
        "OPTIM.MAX_EPOCH", "10",
        "TEST.FINAL_MODEL", "last_step",
        "TRAIN.CHECKPOINT_FREQ", "0",
    ]


def 生成计划(source_root: Path, data_root: Path, output_root: Path, python: Path) -> dict:
    source_root = source_root.resolve()
    data_root = data_root.resolve()
    output_root = output_root.resolve()
    trainer_config = source_root / "configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml"
    runs = []
    for dataset, dataset_config, _, stage in 数据集:
        for seed in 种子:
            run_root = output_root / stage / dataset / f"种子{seed}"
            training = run_root / "训练"
            common = [
                str(python), "-B", str(source_root / "train.py"),
                "--root", str(data_root),
                "--seed", str(seed),
                "--trainer", "CoCoOp",
                "--dataset-config-file", str(source_root / "configs/datasets" / dataset_config),
                "--config-file", str(trainer_config),
            ]
            train_command = common + [
                "--output-dir", str(training),
                *固定选项(),
                "DATASET.SUBSAMPLE_CLASSES", "base",
            ]
            eval_commands = {}
            for split, split_name in (("base", "基础测试"), ("new", "新类别测试")):
                eval_commands[split] = common + [
                    "--output-dir", str(run_root / split_name),
                    "--model-dir", str(training),
                    "--load-epoch", "10",
                    "--eval-only",
                    *固定选项(),
                    "DATASET.SUBSAMPLE_CLASSES", split,
                ]
            runs.append(
                {
                    "序号": len(runs) + 1,
                    "阶段": stage,
                    "数据集": dataset,
                    "种子": seed,
                    "运行目录": str(run_root),
                    "训练目录": str(training),
                    "检查点": str(training / "prompt_learner/model.pth.tar-10"),
                    "命令": {"训练": train_command, "base": eval_commands["base"], "new": eval_commands["new"]},
                }
            )
    return {
        "协议": 协议名称,
        "状态": "待执行",
        "方法": "CoCoOp（Conditional Context Optimization）",
        "上游": {
            "仓库": "https://github.com/KaiyangZhou/CoOp",
            "提交": 上游提交,
            "训练器SHA256": 上游训练器摘要,
            "配置SHA256": 上游配置摘要,
        },
        "固定协议": {
            "backbone": "CLIP ViT-B/16",
            "shots": 16,
            "context_tokens": 4,
            "context_initialization": "a photo of a",
            "precision": "fp16",
            "train_batch_size": 1,
            "test_batch_size": 100,
            "epochs": 10,
            "seeds": list(种子),
            "checkpoint": "指定第10轮；Base与Novel独立重载同一SHA256检查点",
            "selection": "不使用Novel结果选择epoch、数据集、种子或超参数",
        },
        "源码目录": str(source_root),
        "数据目录": str(data_root),
        "输出目录": str(output_root),
        "Python": str(python),
        "预期CLIP_SHA256": 预期CLIP摘要,
        "源码文件SHA256": {path.as_posix(): sha256(source_root / path) for path in 源码清单(source_root)},
        "阶段": {
            "核心三数据集": ["UCF101", "DTD", "FGVC-Aircraft"],
            "扩展两数据集": ["EuroSAT", "OxfordFlowers"],
        },
        "运行数": len(runs),
        "命令数": len(runs) * 3,
        "运行": runs,
    }


def 检查计划身份(plan: dict, require_runtime: bool) -> tuple[Path, Path, Path]:
    if plan.get("协议") != 协议名称 or plan.get("运行数") != 15 or plan.get("命令数") != 45:
        raise ValueError("执行计划身份、运行数或命令数错误")
    if plan.get("固定协议") != {
        "backbone": "CLIP ViT-B/16",
        "shots": 16,
        "context_tokens": 4,
        "context_initialization": "a photo of a",
        "precision": "fp16",
        "train_batch_size": 1,
        "test_batch_size": 100,
        "epochs": 10,
        "seeds": [1, 2, 3],
        "checkpoint": "指定第10轮；Base与Novel独立重载同一SHA256检查点",
        "selection": "不使用Novel结果选择epoch、数据集、种子或超参数",
    }:
        raise ValueError("固定协议被修改")
    source_root = Path(plan["源码目录"])
    data_root = Path(plan["数据目录"])
    output_root = Path(plan["输出目录"])
    actual = {path.as_posix(): sha256(source_root / path) for path in 源码清单(source_root)}
    if actual != plan["源码文件SHA256"]:
        raise ValueError("源码或配置在计划生成后发生变化")
    if require_runtime:
        if not data_root.is_dir():
            raise FileNotFoundError(f"数据目录不存在：{data_root}")
        missing = [dataset_dir for _, _, dataset_dir, _ in 数据集 if not (data_root / dataset_dir).is_dir()]
        if missing:
            raise FileNotFoundError(f"缺少数据集目录：{missing}")
        clip_path = source_root / "clip/ViT-B-16.pt"
        if not clip_path.is_file() or sha256(clip_path) != 预期CLIP摘要:
            raise ValueError("CLIP ViT-B/16权重缺失或SHA256不一致")
    return source_root, data_root, output_root


def 解析日志(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺少评估日志：{path}")
    text = path.read_text(encoding="utf-8")
    accuracy = 准确率模式.findall(text)
    correct = 正确数模式.findall(text)
    total = 总数模式.findall(text)
    if len(accuracy) != 1 or len(correct) != 1 or len(total) != 1:
        raise ValueError(f"评估指标数量异常：{path}")
    result = {
        "准确率": float(accuracy[0]),
        "正确数": int(correct[0].replace(",", "")),
        "总数": int(total[0].replace(",", "")),
        "日志SHA256": sha256(path),
    }
    result["复算准确率"] = 100.0 * result["正确数"] / result["总数"]
    if abs(result["准确率"] - result["复算准确率"]) > 0.00051:
        raise ValueError(f"准确率不能由correct/total复算：{path}")
    return result


def 执行命令(command: list[str], source_root: Path, log_path: Path, env: dict[str, str]) -> tuple[int, float]:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=source_root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    return completed.returncode, time.time() - started


def 阶段运行(plan: dict, stage: str) -> list[dict]:
    return [item for item in plan["运行"] if item["阶段"] == stage]


def 汇总逐种子(rows: list[dict]) -> list[dict]:
    summaries = []
    for dataset in [name for name, _, _, _ in 数据集 if any(row["数据集"] == name for row in rows)]:
        group = sorted((row for row in rows if row["数据集"] == dataset), key=lambda row: row["种子"])
        if [row["种子"] for row in group] != [1, 2, 3]:
            raise ValueError(f"三种子结果不完整：{dataset}")
        base = [row["Base"] for row in group]
        novel = [row["Novel"] for row in group]
        hms = [row["HM"] for row in group]
        base_mean = statistics.mean(base)
        novel_mean = statistics.mean(novel)
        summaries.append(
            {
                "数据集": dataset,
                "种子数": 3,
                "Base均值": base_mean,
                "Base样本标准差": statistics.stdev(base),
                "Novel均值": novel_mean,
                "Novel样本标准差": statistics.stdev(novel),
                "HM_均值计算": 调和平均(base_mean, novel_mean),
                "种子HM均值": statistics.mean(hms),
                "种子HM样本标准差": statistics.stdev(hms),
            }
        )
    return summaries


def 写阶段结果(stage_root: Path, stage: str, rows: list[dict], status: dict) -> None:
    summaries = 汇总逐种子(rows)
    result = {
        "状态": "通过",
        "协议": 协议名称,
        "阶段": stage,
        "逐种子": rows,
        "数据集汇总": summaries,
        "执行": {"GPU": status["GPU"], "命令数": status["已完成命令"], "总耗时秒": status["总耗时秒"]},
    }
    原子写json(stage_root / "阶段结果.json", result)
    with (stage_root / "逐种子结果.csv").open("w", encoding="utf-8", newline="") as file:
        fields = ["数据集", "种子", "Base", "Novel", "HM", "检查点SHA256"]
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# CoCoOp {stage}结果",
        "",
        "| 数据集 | Base均值 ± SD | Novel均值 ± SD | HM（均值） | 种子HM均值 ± SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['数据集']} | {item['Base均值']:.2f} ± {item['Base样本标准差']:.2f} | "
            f"{item['Novel均值']:.2f} ± {item['Novel样本标准差']:.2f} | {item['HM_均值计算']:.2f} | "
            f"{item['种子HM均值']:.2f} ± {item['种子HM样本标准差']:.2f} |"
        )
    lines.extend(["", "Base与Novel均从同一第10轮检查点独立重载；没有依据Novel结果选择epoch或删除种子。", ""])
    (stage_root / "阶段结果报告.md").write_text("\n".join(lines), encoding="utf-8")


def 写文件清单(root: Path) -> None:
    target = root / "文件清单.txt"
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item != target):
        entries.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    target.write_text("\n".join(entries) + "\n", encoding="utf-8")


def 执行阶段(plan_path: Path, stage: str) -> int:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source_root, _, output_root = 检查计划身份(plan, require_runtime=True)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("没有可用CUDA，拒绝创建输出或执行训练")
    runs = 阶段运行(plan, stage)
    expected_runs = 9 if stage == "核心三数据集" else 6
    if len(runs) != expected_runs:
        raise ValueError(f"阶段运行数错误：{len(runs)}")
    stage_root = output_root / stage
    if stage_root.exists():
        raise FileExistsError(f"阶段输出必须不存在，拒绝覆盖：{stage_root}")
    stage_root.mkdir(parents=True)
    shutil.copy2(plan_path, stage_root / "执行计划.json")
    status = {
        "状态": "运行中",
        "协议": 协议名称,
        "阶段": stage,
        "GPU": torch.cuda.get_device_name(0),
        "开始时间Unix": time.time(),
        "总命令": len(runs) * 3,
        "已完成命令": 0,
        "命令": [],
    }
    原子写json(stage_root / "执行状态.json", status)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "4",
            "PYTHONPATH": f"{source_root}:{source_root / 'Dassl.pytorch'}",
        }
    )
    rows = []
    try:
        for run in runs:
            run_root = Path(run["运行目录"])
            if run_root.exists():
                raise FileExistsError(f"单次运行目录已存在，拒绝覆盖：{run_root}")
            run_root.mkdir(parents=True)
            checkpoint_sha = None
            metrics = {}
            for command_name in ("训练", "base", "new"):
                log_name = {"训练": "训练", "base": "基础测试", "new": "新类别测试"}[command_name]
                log_path = stage_root / "运行日志" / f"{run['序号']:02d}-{run['数据集']}-种子{run['种子']}-{log_name}.txt"
                returncode, duration = 执行命令(run["命令"][command_name], source_root, log_path, env)
                record = {
                    "运行序号": run["序号"],
                    "数据集": run["数据集"],
                    "种子": run["种子"],
                    "步骤": command_name,
                    "退出码": returncode,
                    "耗时秒": duration,
                    "运行日志": str(log_path),
                }
                status["命令"].append(record)
                if returncode != 0:
                    status["状态"] = "失败"
                    status["失败命令"] = record
                    原子写json(stage_root / "执行状态.json", status)
                    raise RuntimeError(f"命令失败，见：{log_path}")
                if command_name == "训练":
                    checkpoint = Path(run["检查点"])
                    if not checkpoint.is_file():
                        raise FileNotFoundError(f"缺少指定第10轮检查点：{checkpoint}")
                    checkpoint_sha = sha256(checkpoint)
                else:
                    eval_dir = run_root / ("基础测试" if command_name == "base" else "新类别测试")
                    parsed = 解析日志(eval_dir / "log.txt")
                    if list(eval_dir.rglob("*.pth.tar*")):
                        raise ValueError(f"评估目录不应包含训练检查点：{eval_dir}")
                    parsed["加载检查点SHA256"] = checkpoint_sha
                    metrics[command_name] = parsed
                status["已完成命令"] += 1
                原子写json(stage_root / "执行状态.json", status)
            base = metrics["base"]["复算准确率"]
            novel = metrics["new"]["复算准确率"]
            rows.append(
                {
                    "数据集": run["数据集"],
                    "种子": run["种子"],
                    "Base": base,
                    "Novel": novel,
                    "HM": 调和平均(base, novel),
                    "检查点SHA256": checkpoint_sha,
                    "基础测试日志SHA256": metrics["base"]["日志SHA256"],
                    "新类别测试日志SHA256": metrics["new"]["日志SHA256"],
                }
            )
        if {path.as_posix(): sha256(source_root / path) for path in 源码清单(source_root)} != plan["源码文件SHA256"]:
            raise ValueError("执行期间源码或配置发生变化")
        status["状态"] = "完成"
        status["结束时间Unix"] = time.time()
        status["总耗时秒"] = status["结束时间Unix"] - status["开始时间Unix"]
        原子写json(stage_root / "执行状态.json", status)
        写阶段结果(stage_root, stage, rows, status)
        写文件清单(stage_root)
    except Exception:
        if status.get("状态") == "运行中":
            status["状态"] = "失败"
            status["结束时间Unix"] = time.time()
            原子写json(stage_root / "执行状态.json", status)
        raise
    print(json.dumps({"状态": "通过", "阶段": stage, "运行数": len(runs), "GPU": status["GPU"]}, ensure_ascii=False))
    return 0


def 最终汇总(plan_path: Path) -> int:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _, _, output_root = 检查计划身份(plan, require_runtime=False)
    rows = []
    stage_evidence = []
    for stage in ("核心三数据集", "扩展两数据集"):
        result_path = output_root / stage / "阶段结果.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("状态") != "通过" or result.get("阶段") != stage:
            raise ValueError(f"阶段结果没有通过：{stage}")
        rows.extend(result["逐种子"])
        stage_evidence.append({"阶段": stage, "结果SHA256": sha256(result_path)})
    identities = {(row["数据集"], row["种子"]) for row in rows}
    expected = {(dataset, seed) for dataset, _, _, _ in 数据集 for seed in 种子}
    if len(rows) != 15 or identities != expected:
        raise ValueError("五数据集三种子结果不完整或存在重复")
    summaries = 汇总逐种子(rows)
    cross = {
        "数据集等权Base均值": statistics.mean(row["Base均值"] for row in summaries),
        "数据集等权Novel均值": statistics.mean(row["Novel均值"] for row in summaries),
        "数据集等权HM均值": statistics.mean(row["HM_均值计算"] for row in summaries),
    }
    payload = {
        "状态": "通过",
        "协议": 协议名称,
        "逐种子": rows,
        "数据集汇总": summaries,
        "跨数据集汇总": cross,
        "阶段证据": stage_evidence,
    }
    原子写json(output_root / "五数据集最终结果.json", payload)
    with (output_root / "五数据集最终结果.csv").open("w", encoding="utf-8", newline="") as file:
        fields = ["数据集", "种子数", "Base均值", "Base样本标准差", "Novel均值", "Novel样本标准差", "HM_均值计算", "种子HM均值", "种子HM样本标准差"]
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps({"状态": "通过", "数据集数": 5, "逐种子行数": 15, "输出目录": str(output_root)}, ensure_ascii=False))
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
    run_parser = subparsers.add_parser("执行阶段")
    run_parser.add_argument("--plan-path", type=Path, required=True)
    run_parser.add_argument("--stage", choices=["核心三数据集", "扩展两数据集"], required=True)
    summary_parser = subparsers.add_parser("汇总")
    summary_parser.add_argument("--plan-path", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "计划":
        if args.output_root.exists():
            raise FileExistsError(f"计划输出目录必须不存在：{args.output_root}")
        payload = 生成计划(args.source_root, args.data_root, args.output_root, args.python)
        原子写json(args.plan_path, payload)
        print(json.dumps({"状态": "计划完成", "运行数": payload["运行数"], "命令数": payload["命令数"], "计划": str(args.plan_path)}, ensure_ascii=False))
        return 0
    if args.command == "执行阶段":
        return 执行阶段(args.plan_path, args.stage)
    return 最终汇总(args.plan_path)


if __name__ == "__main__":
    raise SystemExit(main())
