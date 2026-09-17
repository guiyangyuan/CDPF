#!/usr/bin/env python3
"""运行 DTD 与 FGVC-Aircraft 的 12 次冻结关键对照推理；不训练。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


数据集 = {
    "dtd": ("DescribableTextures", "9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba"),
    "fgvc_aircraft": ("FGVCAircraft", "9bd4a7f248f09c47be17c1bb64921bd9a29d9f3b46e03e8caac9b934b2a95bba"),
}


def 写入JSON(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def 生成计划(args: argparse.Namespace) -> dict:
    evaluator = args.release_root / "scripts/repro/evaluate_frozen_fusion.py"
    commands = []
    for key, (display, digest) in 数据集.items():
        for seed in (1, 2, 3):
            baseline = args.baseline_root / "formal" / key / "baseline" / f"seed{seed}"
            for split in ("base", "new"):
                output = args.output_root / display / f"seed{seed}" / split
                command = [
                    str(args.python), "-B", str(evaluator),
                    "--source-root", str(args.source_root),
                    "--baseline-dir", str(baseline),
                    "--dataset", key,
                    "--seed", str(seed),
                    "--split", split,
                    "--data-root", str(args.data_root),
                    "--output-dir", str(output),
                    "--expected-source-sha256", digest,
                ]
                if args.validate_only:
                    command.append("--validate-only")
                commands.append({
                    "序号": len(commands) + 1,
                    "数据集": display,
                    "数据集键": key,
                    "种子": seed,
                    "分支": split,
                    "历史检查点": str(baseline),
                    "输出目录": str(output),
                    "命令": command,
                })
    return {
        "协议": "投稿关键对照剩余两数据集-v1",
        "状态": "待执行",
        "模式": "仅元数据核验" if args.validate_only else "GPU冻结推理",
        "训练次数": 0,
        "推理命令数": len(commands),
        "数据集": list(数据集),
        "种子": [1, 2, 3],
        "分支": ["base", "new"],
        "固定主方法alpha": 0.5,
        "统一alpha网格": [0.0, 0.25, 0.5, 0.75, 1.0],
        "命令": commands,
    }


def 执行(args: argparse.Namespace, plan: dict) -> dict:
    if args.output_root.exists():
        raise FileExistsError(f"请使用新的输出目录：{args.output_root}")
    args.output_root.mkdir(parents=True)
    写入JSON(args.output_root / "执行计划.json", plan)
    status = {"状态": "运行中", "训练次数": 0, "已完成": 0, "总数": len(plan["命令"]), "运行": []}
    for row in plan["命令"]:
        started = time.time()
        log = args.output_root / "运行日志" / f"{row['序号']:02d}-{row['数据集']}-seed{row['种子']}-{row['分支']}.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(row["命令"], stdout=handle, stderr=subprocess.STDOUT, text=True)
        record = {
            "序号": row["序号"], "数据集": row["数据集"], "种子": row["种子"], "分支": row["分支"],
            "退出码": completed.returncode, "耗时秒": time.time() - started, "日志": str(log),
        }
        status["运行"].append(record)
        if completed.returncode != 0:
            status["状态"] = "失败"
            status["失败命令"] = row
            写入JSON(args.output_root / "执行状态.json", status)
            raise RuntimeError(f"第 {row['序号']} 条命令失败，见 {log}")
        status["已完成"] += 1
        写入JSON(args.output_root / "执行状态.json", status)

    status["状态"] = "完成"
    if not args.validate_only:
        summary = args.output_root / "汇总"
        command = [
            str(args.python), "-B", str(args.release_root / "scripts/repro/汇总类别名称与权重稳健性.py"),
            "--input-root", str(args.output_root), "--output-dir", str(summary),
            "--datasets", "DTD", "FGVCAircraft",
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        status["汇总命令"] = command
        status["汇总退出码"] = completed.returncode
        status["汇总标准输出"] = completed.stdout[-4000:]
        status["汇总标准错误"] = completed.stderr[-4000:]
        if completed.returncode != 0:
            status["状态"] = "汇总失败"
            写入JSON(args.output_root / "执行状态.json", status)
            raise RuntimeError("12次推理完成，但两数据集汇总失败")
    写入JSON(args.output_root / "执行状态.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/textconsistency/data"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("/root/autodl-tmp/textconsistency/.venv/bin/python"))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    plan = 生成计划(args)
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(执行(args, plan), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
