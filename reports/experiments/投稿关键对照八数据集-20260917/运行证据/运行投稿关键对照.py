#!/usr/bin/env python3
"""生成或执行十数据集类别名称与固定权重对照；全程不训练。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.repro.evaluate_frozen_fusion import DATASETS


数据集顺序 = [
    "ucf101", "dtd", "fgvc_aircraft", "eurosat", "oxford_flowers",
    "caltech101", "oxford_pets", "food101", "stanford_cars", "sun397",
]


def 历史输入(server_root: Path, dataset: str, seed: int) -> tuple[Path, Path]:
    releases = server_root / "releases"
    runs = server_root / "runs"
    if dataset == "ucf101":
        return (
            releases / "cpu-prep-20260910",
            server_root / f"reused-inputs/attribution-20260911/ucf101/baseline/seed{seed}",
        )
    if dataset in ("dtd", "fgvc_aircraft"):
        return (
            releases / "structure-20260910-v1",
            runs / f"crossdataset-study-20260910-v1/formal/{dataset}/baseline/seed{seed}",
        )
    if dataset == "eurosat":
        stage = "seed1" if seed == 1 else "seed23"
        return (
            releases / "扩展验证-20260915-8f54900",
            runs / f"扩展验证-EuroSAT-{stage}-20260915-8f54900/基线训练/baseline/seed{seed}",
        )
    if dataset == "oxford_flowers":
        stage = "seed1" if seed == 1 else "seed23"
        return (
            releases / "扩展验证-20260915-8f54900",
            runs / f"牛津花卉扩展验证-{stage}-20260915-8f54900-正式/基线训练/baseline/seed{seed}",
        )
    if dataset in ("caltech101", "oxford_pets", "food101"):
        display = {"caltech101": "Caltech101", "oxford_pets": "OxfordPets", "food101": "Food101"}[dataset]
        return (
            releases / "三数据集主方法-20260916-9bd4a7f",
            runs / f"投稿三数据集主方法-20260916-v2/{display}/基线训练/baseline/seed{seed}",
        )
    if dataset in ("stanford_cars", "sun397"):
        display = {"stanford_cars": "StanfordCars", "sun397": "SUN397"}[dataset]
        return (
            releases / "三数据集主方法-20260916-9bd4a7f",
            runs / f"投稿剩余两数据集主方法-20260917/{display}/基线训练/baseline/seed{seed}",
        )
    raise ValueError(f"未知数据集：{dataset}")


def 生成计划(
    release_root: Path,
    server_root: Path,
    data_root: Path,
    output_root: Path,
    python: Path,
    *,
    validate_only: bool = False,
) -> dict:
    evaluator = release_root / "scripts/repro/evaluate_frozen_fusion.py"
    commands = []
    for dataset in 数据集顺序:
        for seed in (1, 2, 3):
            source_root, baseline_dir = 历史输入(server_root, dataset, seed)
            for split in ("base", "new"):
                output_dir = output_root / DATASETS[dataset][0] / f"seed{seed}" / split
                command = [
                    str(python), "-B", str(evaluator),
                    "--source-root", str(source_root),
                    "--baseline-dir", str(baseline_dir),
                    "--dataset", dataset,
                    "--seed", str(seed),
                    "--split", split,
                    "--data-root", str(data_root),
                    "--output-dir", str(output_dir),
                    "--expected-source-sha256", DATASETS[dataset][1],
                ]
                if validate_only:
                    command.append("--validate-only")
                commands.append({
                    "序号": len(commands) + 1, "数据集": DATASETS[dataset][0], "数据集键": dataset,
                    "种子": seed, "分支": split, "历史源码": str(source_root),
                    "历史检查点": str(baseline_dir), "输出目录": str(output_dir), "命令": command,
                })
    return {
        "协议": "投稿类别名称与权重稳健性-v1",
        "状态": "待执行", "模式": "仅元数据核验" if validate_only else "GPU冻结推理",
        "训练次数": 0, "推理命令数": len(commands), "数据集": 数据集顺序,
        "种子": [1, 2, 3], "分支": ["base", "new"],
        "对照": ["CoOp", "类别名称", "CoOp+类别名称", "类别描述", "CoOp+类别描述"],
        "全局alpha网格": [0.0, 0.25, 0.5, 0.75, 1.0],
        "选择限制": "所有数据集一次性执行同一alpha网格；不按数据集选最优值，主方法仍固定0.5。",
        "命令": commands,
    }


def 原子写json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def 执行(plan: dict, output_root: Path, release_root: Path, python: Path) -> dict:
    if output_root.exists():
        raise FileExistsError(f"请使用新输出目录：{output_root}")
    output_root.mkdir(parents=True)
    status = {"状态": "运行中", "已完成": 0, "总数": len(plan["命令"]), "运行": []}
    原子写json(output_root / "执行计划.json", plan)
    for row in plan["命令"]:
        started = time.time()
        log_path = output_root / "运行日志" / f"{row['序号']:02d}-{row['数据集']}-seed{row['种子']}-{row['分支']}.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(row["命令"], stdout=log, stderr=subprocess.STDOUT, text=True)
        receipt = {**{key: row[key] for key in ("序号", "数据集", "种子", "分支")},
                   "退出码": result.returncode, "耗时秒": time.time() - started, "日志": str(log_path)}
        status["运行"].append(receipt)
        if result.returncode != 0:
            status["状态"] = "失败"; status["失败命令"] = row
            原子写json(output_root / "执行状态.json", status)
            raise RuntimeError(f"第{row['序号']}条命令失败，见 {log_path}")
        status["已完成"] += 1
        原子写json(output_root / "执行状态.json", status)
    status["状态"] = "完成"
    if plan["模式"] == "GPU冻结推理":
        summary_dir = output_root / "汇总"
        command = [str(python), "-B", str(release_root / "scripts/repro/汇总类别名称与权重稳健性.py"),
                   "--input-root", str(output_root), "--output-dir", str(summary_dir)]
        completed = subprocess.run(command, capture_output=True, text=True)
        status["汇总命令"] = command; status["汇总退出码"] = completed.returncode
        status["汇总标准输出"] = completed.stdout[-4000:]
        status["汇总标准错误"] = completed.stderr[-4000:]
        if completed.returncode != 0:
            status["状态"] = "汇总失败"
            原子写json(output_root / "执行状态.json", status)
            raise RuntimeError("全部推理完成但汇总失败")
    原子写json(output_root / "执行状态.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--server-root", type=Path, default=Path("/root/autodl-tmp/textconsistency"))
    parser.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/textconsistency/data"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path("/root/autodl-tmp/textconsistency/.venv/bin/python"))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    plan = 生成计划(args.release_root, args.server_root, args.data_root, args.output_root,
                args.python, validate_only=args.validate_only)
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2)); return 0
    print(json.dumps(执行(plan, args.output_root, args.release_root, args.python), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
