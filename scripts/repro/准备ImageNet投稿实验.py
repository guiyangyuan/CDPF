#!/usr/bin/env python3
"""准备、核验并执行论文使用的 ImageNet Base-to-Novel 主实验。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Mapping


ROOT = Path(__file__).resolve().parents[2]
官方类别文件SHA256 = "89e0b642c6697a3a43a06a2bb65f9fefb3ca295ccc5f4ba9ce385f1a3a58c629"
描述文件SHA256 = "31da80d50a83ef6935238eafc5b44df4b839d23f3cead8591fcfa2cb2d808b91"
预期模型源码摘要 = "e4d4343f31b7cdbcccd3b81ddc99536080c62e573783b27c9366a3c64a7892db"
预期重复名称 = {
    "missile": ["n03773504", "n04008634"],
    "sunglasses": ["n04355933", "n04356056"],
}
图像后缀 = {".jpeg", ".jpg", ".png"}
种子 = (1, 2, 3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def 原子写json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def 标准化类别名称(name: str) -> str:
    return " ".join(name.replace("_", " ").casefold().split())


def 读取类别名称(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 ImageNet 类别名称文件：{path}")
    rows: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            raise ValueError(f"类别名称文件第 {number} 行为空")
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or re.fullmatch(r"n\d{8}", parts[0]) is None or not parts[1].strip():
            raise ValueError(f"类别名称文件第 {number} 行格式错误")
        synset, classname = parts[0], parts[1].strip()
        if synset in rows:
            raise ValueError(f"类别名称文件重复 synset：{synset}")
        rows[synset] = classname
    return rows


def 核验描述覆盖(classnames: Mapping[str, str], templates: Mapping[str, object],
               n_views: int = 20) -> dict:
    if n_views < 1:
        raise ValueError("n_views 必须为正数")
    lookup: dict[str, object] = {}
    for key, value in templates.items():
        normalized = 标准化类别名称(key)
        if normalized in lookup and lookup[normalized] != value:
            raise ValueError(f"描述文件存在规范化后冲突的类别：{key}")
        lookup[normalized] = value

    by_name: dict[str, list[str]] = defaultdict(list)
    missing: list[str] = []
    short: list[str] = []
    effective_counts: dict[str, int] = {}
    for synset, classname in classnames.items():
        normalized = 标准化类别名称(classname)
        by_name[normalized].append(synset)
        candidates = lookup.get(normalized)
        if not isinstance(candidates, list) or any(not isinstance(item, str) for item in candidates):
            missing.append(synset)
            continue
        unique = list(dict.fromkeys(" ".join(item.split()) for item in candidates if item.strip()))
        effective_counts[synset] = min(len(unique), n_views)
        if len(unique) < n_views:
            short.append(synset)

    return {
        "synset数": len(classnames),
        "唯一显示名称数": len(by_name),
        "描述键数": len(templates),
        "每类使用描述上限": n_views,
        "缺少描述的synset": missing,
        "少于20条描述的synset": short,
        "重复显示名称": {name: synsets for name, synsets in by_name.items() if len(synsets) > 1},
        "有效描述数范围": (
            [min(effective_counts.values()), max(effective_counts.values())]
            if effective_counts else []
        ),
    }


def _类别目录(root: Path) -> set[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"缺少图像目录：{root}")
    return {path.name for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")}


def _图像数量(folder: Path) -> int:
    return sum(
        1 for path in folder.iterdir()
        if path.is_file() and not path.name.startswith(".") and path.suffix.casefold() in 图像后缀
    )


def 核验图像目录(image_root: Path, classnames: Mapping[str, str], *,
             expected_train_images: int = 1_281_167,
             expected_val_images: int = 50_000,
             minimum_train_per_class: int = 16,
             expected_val_per_class: int = 50) -> dict:
    expected = set(classnames)
    train_root, val_root = image_root / "train", image_root / "val"
    train_dirs, val_dirs = _类别目录(train_root), _类别目录(val_root)
    if train_dirs != expected or val_dirs != expected:
        detail = {
            "训练缺失": sorted(expected - train_dirs),
            "训练多余": sorted(train_dirs - expected),
            "验证缺失": sorted(expected - val_dirs),
            "验证多余": sorted(val_dirs - expected),
        }
        raise ValueError(f"ImageNet 类别目录与 classnames.txt 不一致：{detail}")

    train_counts = {synset: _图像数量(train_root / synset) for synset in sorted(expected)}
    val_counts = {synset: _图像数量(val_root / synset) for synset in sorted(expected)}
    too_small = {key: value for key, value in train_counts.items() if value < minimum_train_per_class}
    wrong_val = {key: value for key, value in val_counts.items() if value != expected_val_per_class}
    if too_small:
        raise ValueError(f"训练类别不足 {minimum_train_per_class} 张：{too_small}")
    if wrong_val:
        raise ValueError(f"验证集不是每类 {expected_val_per_class} 张：{wrong_val}")
    train_total, val_total = sum(train_counts.values()), sum(val_counts.values())
    if train_total != expected_train_images or val_total != expected_val_images:
        raise ValueError(
            f"ImageNet 图像总数错误：train={train_total}/{expected_train_images}, "
            f"val={val_total}/{expected_val_images}"
        )
    return {
        "类别目录数": len(expected),
        "训练图像数": train_total,
        "验证图像数": val_total,
        "训练每类最少": min(train_counts.values()),
        "训练每类最多": max(train_counts.values()),
        "验证每类": expected_val_per_class,
        "类别目录完全匹配": True,
    }


def 模型源码摘要(source_root: Path) -> str:
    digest = hashlib.sha256()
    files = [source_root / "train.py"]
    for folder in ("trainers", "datasets", "clip", "Dassl.pytorch/dassl"):
        files.extend(sorted((source_root / folder).rglob("*.py")))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"模型源码不完整：{missing[0]}")
    for path in files:
        digest.update(str(path.relative_to(source_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def 核验数据集(source_root: Path, data_root: Path) -> dict:
    dataset_root = data_root.resolve() / "imagenet"
    classnames_path = dataset_root / "classnames.txt"
    if sha256(classnames_path) != 官方类别文件SHA256:
        raise ValueError("classnames.txt 不是 CoOp 官方固定文件")
    classnames = 读取类别名称(classnames_path)
    if len(classnames) != 1000 or list(classnames) != sorted(classnames):
        raise ValueError("ImageNet 必须包含按 synset 排序的 1000 个类别")
    template_path = source_root / "templates/imagenet_prompts_full.json"
    if sha256(template_path) != 描述文件SHA256:
        raise ValueError("ImageNet 描述文件哈希发生变化")
    descriptions = 核验描述覆盖(classnames, json.loads(template_path.read_text(encoding="utf-8")))
    if descriptions["缺少描述的synset"] or descriptions["少于20条描述的synset"]:
        raise ValueError("ImageNet 描述覆盖不足，拒绝启动实验")
    if descriptions["重复显示名称"] != 预期重复名称:
        raise ValueError("ImageNet 重复显示名称与冻结协议不一致")
    images = 核验图像目录(dataset_root / "images", classnames)
    ordered = list(classnames)
    base, novel = ordered[:500], ordered[500:]
    return {
        "状态": "通过",
        "数据目录": str(dataset_root),
        "类别名称文件SHA256": sha256(classnames_path),
        "描述文件SHA256": sha256(template_path),
        "类别与描述": descriptions,
        "图像": images,
        "类别划分": {
            "规则": "按synset排序前500类为Base，后500类为Novel",
            "Base数": len(base),
            "Novel数": len(novel),
            "Base类别SHA256": hashlib.sha256("\n".join(base).encode()).hexdigest(),
            "Novel类别SHA256": hashlib.sha256("\n".join(novel).encode()).hexdigest(),
        },
        "缓存": {
            "preprocessed.pkl存在": (dataset_root / "preprocessed.pkl").is_file(),
            "fewshot目录存在": (dataset_root / "split_fewshot").is_dir(),
        },
    }


def _计划源码文件(source_root: Path) -> list[Path]:
    relative = [
        "scripts/repro/准备ImageNet投稿实验.py",
        "scripts/repro/run_pilot.py",
        "scripts/repro/evaluate_frozen_fusion.py",
        "scripts/repro/汇总扩展验证.py",
        "configs/datasets/imagenet.yaml",
        "configs/repro/扩展验证.yaml",
        "datasets/imagenet.py",
        "trainers/textconsistency_common.py",
        "trainers/textconsistency_metadata.py",
        "templates/imagenet_prompts_full.json",
        "assets/imagenet/classnames.txt",
    ]
    paths = [source_root / path for path in relative]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少 ImageNet 实验文件：{missing[0]}")
    return paths


def 生成执行计划(source_root: Path, data_root: Path, output_root: Path, python: Path) -> dict:
    source_root, data_root, output_root = source_root.resolve(), data_root.resolve(), output_root.resolve()
    sys.path.insert(0, str(source_root))
    from scripts.repro import run_pilot

    actual_digest = 模型源码摘要(source_root)
    if actual_digest != 预期模型源码摘要:
        raise ValueError(f"模型源码摘要未冻结：{actual_digest}")
    baseline_root = output_root / "基线训练"
    fusion_root = output_root / "融合评估"
    baseline_commands = run_pilot.build_commands(
        data_root,
        baseline_root,
        list(种子),
        20,
        ["baseline"],
        device="cuda",
        python=str(python),
        dataset="imagenet",
        kd_weight=0.0,
    )
    fusion_commands = []
    for seed in 种子:
        for split in ("base", "new"):
            fusion_commands.append([
                str(python), "-B", str(source_root / "scripts/repro/evaluate_frozen_fusion.py"),
                "--source-root", str(source_root),
                "--baseline-dir", str(baseline_root / "baseline" / f"seed{seed}"),
                "--dataset", "imagenet",
                "--seed", str(seed),
                "--split", split,
                "--data-root", str(data_root),
                "--output-dir", str(fusion_root / f"seed{seed}" / split),
                "--expected-source-sha256", actual_digest,
            ])
    summary_command = [
        str(python), "-B", str(source_root / "scripts/repro/汇总扩展验证.py"),
        "--root", str(output_root),
    ]
    commands = baseline_commands + fusion_commands + [summary_command]
    source_files = _计划源码文件(source_root)
    return {
        "协议": "ImageNet投稿主实验-v1",
        "状态": "待执行",
        "数据集": "ImageNet",
        "数据集键": "imagenet",
        "数据目录": str(data_root),
        "输出目录": str(output_root),
        "源码目录": str(source_root),
        "Python": str(python),
        "固定协议": {
            "backbone": "CLIP ViT-B/16",
            "shots": 16,
            "context_tokens": 4,
            "epochs": 20,
            "seeds": list(种子),
            "alpha": 0.5,
            "训练方法": "CoOp，仅Base类别交叉熵",
            "主方法": "0.5*softmax(CoOp logits)+0.5*softmax(冻结均值描述 logits)",
            "类别划分": "按synset排序前500类为Base，后500类为Novel",
            "选择限制": "三种子完整执行；不搜索权重、温度、描述、轮数，不按结果停跑或删种子",
        },
        "模型源码SHA256": actual_digest,
        "官方类别名称SHA256": 官方类别文件SHA256,
        "描述文件SHA256": 描述文件SHA256,
        "训练数": 3,
        "基线训练与复评命令数": len(baseline_commands),
        "冻结融合评估数": len(fusion_commands),
        "命令数": len(commands),
        "源码文件SHA256": {
            str(path.relative_to(source_root)): sha256(path) for path in source_files
        },
        "命令": commands,
    }


def 检查执行计划(plan: Mapping[str, object]) -> tuple[Path, Path, Path]:
    if plan.get("协议") != "ImageNet投稿主实验-v1":
        raise ValueError("执行计划协议错误")
    if plan.get("训练数") != 3 or plan.get("冻结融合评估数") != 6 or plan.get("命令数") != 16:
        raise ValueError("执行计划命令身份错误")
    source_root = Path(str(plan["源码目录"])).resolve()
    data_root = Path(str(plan["数据目录"])).resolve()
    output_root = Path(str(plan["输出目录"])).resolve()
    if 模型源码摘要(source_root) != plan.get("模型源码SHA256"):
        raise ValueError("计划生成后模型源码发生变化")
    current = {
        str(path.relative_to(source_root)): sha256(path) for path in _计划源码文件(source_root)
    }
    if current != plan.get("源码文件SHA256"):
        raise ValueError("计划生成后执行文件或资源发生变化")
    return source_root, data_root, output_root


def 执行(plan_path: Path) -> int:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source_root, data_root, output_root = 检查执行计划(plan)
    if output_root.exists():
        raise FileExistsError(f"输出目录必须不存在：{output_root}")
    audit = 核验数据集(source_root, data_root)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("没有可用 CUDA，拒绝创建输出或启动训练")

    output_root.mkdir(parents=True)
    shutil.copy2(plan_path, output_root / "执行计划.json")
    原子写json(output_root / "数据核验.json", audit)
    原子写json(output_root / "实验协议.json", {
        "协议": plan["协议"],
        "数据集": "ImageNet",
        "数据集键": "imagenet",
        "种子": list(种子),
        "训练轮数": 20,
        "样本数": 16,
        "骨干": "ViT-B/16",
        "主比较": plan["固定协议"]["主方法"],
    })
    status = {
        "状态": "运行中",
        "GPU": torch.cuda.get_device_name(0),
        "开始时间Unix": time.time(),
        "已完成命令": 0,
        "总命令": len(plan["命令"]),
    }
    原子写json(output_root / "执行状态.json", status)
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "4",
        "PYTHONPATH": f"{source_root}:{source_root / 'Dassl.pytorch'}",
    })
    try:
        for index, command in enumerate(plan["命令"], start=1):
            status["当前命令"] = index
            原子写json(output_root / "执行状态.json", status)
            log_path = output_root / f"命令{index:02d}-完整日志.txt"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=source_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0:
                status.update({"状态": "失败", "退出码": completed.returncode, "失败日志": str(log_path)})
                原子写json(output_root / "执行状态.json", status)
                raise RuntimeError(f"ImageNet 第 {index} 条命令失败，见 {log_path}")
            status["已完成命令"] = index
        status.update({"状态": "完成", "当前命令": None, "结束时间Unix": time.time()})
        status["总耗时秒"] = status["结束时间Unix"] - status["开始时间Unix"]
        原子写json(output_root / "执行状态.json", status)
    except Exception:
        if status.get("状态") == "运行中":
            status.update({"状态": "失败", "结束时间Unix": time.time()})
            原子写json(output_root / "执行状态.json", status)
        raise
    print(json.dumps({"状态": "完成", "输出目录": str(output_root)}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    install = subparsers.add_parser("安装类别名称")
    install.add_argument("--data-root", type=Path, required=True)

    verify = subparsers.add_parser("核验")
    verify.add_argument("--data-root", type=Path, required=True)
    verify.add_argument("--output", type=Path)

    planning = subparsers.add_parser("计划")
    planning.add_argument("--data-root", type=Path, required=True)
    planning.add_argument("--output-root", type=Path, required=True)
    planning.add_argument("--plan-path", type=Path, required=True)
    planning.add_argument("--python", type=Path, default=Path(sys.executable))

    running = subparsers.add_parser("执行")
    running.add_argument("--plan-path", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "安装类别名称":
        source = ROOT / "assets/imagenet/classnames.txt"
        target = args.data_root.resolve() / "imagenet/classnames.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) != 官方类别文件SHA256:
            raise FileExistsError(f"目标已有不同文件，拒绝覆盖：{target}")
        shutil.copy2(source, target)
        print(json.dumps({"状态": "完成", "文件": str(target), "SHA256": sha256(target)}, ensure_ascii=False))
        return 0
    if args.command == "核验":
        result = 核验数据集(ROOT, args.data_root)
        if args.output:
            原子写json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "计划":
        if args.output_root.exists():
            raise FileExistsError(f"实验输出目录必须是新路径：{args.output_root}")
        plan = 生成执行计划(ROOT, args.data_root, args.output_root, args.python)
        原子写json(args.plan_path, plan)
        print(json.dumps({"状态": "计划完成", "命令数": plan["命令数"]}, ensure_ascii=False))
        return 0
    return 执行(args.plan_path)


if __name__ == "__main__":
    raise SystemExit(main())
