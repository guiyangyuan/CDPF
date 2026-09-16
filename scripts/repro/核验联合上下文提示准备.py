#!/usr/bin/env python3
"""核验 CoCoOp 上游身份、本地适配边界和五数据集冻结计划。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


上游训练器摘要 = "90c7b866d59bce0fd825fa099ab03606d72ef9965bbf5d9cee20729728d1bc84"
上游配置摘要 = "3294d36c623187c137a2bf43ba6eefd7ccbf50d14b7183fae95731dcb2aa44c1"
数据集 = ["UCF101", "DTD", "FGVC-Aircraft", "EuroSAT", "OxfordFlowers"]
种子 = [1, 2, 3]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def 去除加载适配(tree: ast.Module) -> ast.Module:
    return ast.Module(
        body=[node for node in tree.body if not (isinstance(node, ast.FunctionDef) and node.name == "load_clip_to_cpu")],
        type_ignores=[],
    )


def 选项值(command: list[str], key: str) -> str:
    matches = [index for index, value in enumerate(command) if value == key]
    if len(matches) != 1 or matches[0] + 1 >= len(command):
        raise ValueError(f"命令中的固定选项数量异常：{key}")
    return command[matches[0] + 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--official-trainer", type=Path, required=True)
    parser.add_argument("--official-config", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    checks = []

    if sha256(args.official_trainer) != 上游训练器摘要:
        raise ValueError("上游CoCoOp训练器SHA256错误")
    if sha256(args.official_config) != 上游配置摘要:
        raise ValueError("上游CoCoOp配置SHA256错误")
    checks.append("上游固定提交的训练器与配置SHA256一致")

    official_tree = ast.parse(args.official_trainer.read_text(encoding="utf-8"))
    local_trainer = root / "trainers/cocoop.py"
    local_tree = ast.parse(local_trainer.read_text(encoding="utf-8"))
    if ast.dump(去除加载适配(official_tree), include_attributes=False) != ast.dump(去除加载适配(local_tree), include_attributes=False):
        raise ValueError("除CLIP加载适配外，本地CoCoOp训练器AST与上游不一致")
    checks.append("除固定本地CLIP加载适配外，CoCoOp训练器AST与上游一致")

    local_config = root / "configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml"
    if args.official_config.read_text(encoding="utf-8").strip() != local_config.read_text(encoding="utf-8").strip():
        raise ValueError("本地CoCoOp配置内容与上游不一致")
    checks.append("CoCoOp配置内容与上游一致，只有文件末尾换行差异")

    if plan.get("协议") != "投稿CoCoOp五数据集标准基线-v1":
        raise ValueError("计划协议错误")
    if plan.get("运行数") != 15 or plan.get("命令数") != 45:
        raise ValueError("计划运行数或命令数错误")
    identities = {(item["数据集"], item["种子"]) for item in plan["运行"]}
    expected = {(dataset, seed) for dataset in 数据集 for seed in 种子}
    if identities != expected or len(plan["运行"]) != len(expected):
        raise ValueError("计划缺少数据集/种子或存在重复")
    checks.append("五数据集×三种子共15次训练身份完整且无重复")

    expected_source = plan["源码文件SHA256"]
    actual_source = {name: sha256(root / name) for name in expected_source}
    if actual_source != expected_source:
        raise ValueError("计划中的源码SHA256与当前文件不一致")
    checks.append(f"计划中的{len(actual_source)}个源码与配置SHA256一致")

    stage_counts = {"核心三数据集": 0, "扩展两数据集": 0}
    fixed = {
        "DATASET.NUM_SHOTS": "16",
        "TRAINER.COCOOP.N_CTX": "4",
        "TRAINER.COCOOP.CTX_INIT": "a photo of a",
        "TRAINER.COCOOP.PREC": "fp16",
        "DATALOADER.TRAIN_X.BATCH_SIZE": "1",
        "DATALOADER.TEST.BATCH_SIZE": "100",
        "OPTIM.MAX_EPOCH": "10",
        "TEST.FINAL_MODEL": "last_step",
    }
    for item in plan["运行"]:
        stage_counts[item["阶段"]] += 1
        if not item["检查点"].endswith("prompt_learner/model.pth.tar-10"):
            raise ValueError("检查点不是指定第10轮")
        train = item["命令"]["训练"]
        if "--eval-only" in train or 选项值(train, "DATASET.SUBSAMPLE_CLASSES") != "base":
            raise ValueError("训练命令身份错误")
        for key, value in fixed.items():
            if 选项值(train, key) != value:
                raise ValueError(f"训练固定选项错误：{key}")
        for split in ("base", "new"):
            command = item["命令"][split]
            if "--eval-only" not in command or 选项值(command, "--load-epoch") != "10":
                raise ValueError("评估命令没有指定第10轮")
            if 选项值(command, "--model-dir") != item["训练目录"]:
                raise ValueError("Base/Novel没有加载该运行的同一训练目录")
            if 选项值(command, "DATASET.SUBSAMPLE_CLASSES") != split:
                raise ValueError("评估分支错误")
            for key, value in fixed.items():
                if 选项值(command, key) != value:
                    raise ValueError(f"评估固定选项错误：{key}")
    if stage_counts != {"核心三数据集": 9, "扩展两数据集": 6}:
        raise ValueError(f"阶段运行数错误：{stage_counts}")
    checks.append("45条命令均固定官方10轮协议，Base/Novel加载同一训练目录与第10轮")

    output_root = Path(plan["输出目录"])
    if output_root.exists():
        raise FileExistsError(f"静态计划的输出目录不应存在：{output_root}")
    checks.append("计划输出目录不存在，执行器会拒绝覆盖已有阶段目录")

    payload = {
        "状态": "通过",
        "基准类型": "CoCoOp同协议本地复现准备；不是准确率结果",
        "上游提交": plan["上游"]["提交"],
        "检查项数量": len(checks),
        "检查": checks,
        "计划统计": {"训练数": 15, "评估数": 30, "总命令数": 45, "核心阶段训练数": 9, "扩展阶段训练数": 6},
        "未完成": ["真实数据加载预检", "CUDA行为测试", "正式训练", "Base/Novel准确率", "归档与关机"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"状态": "通过", "检查项数量": len(checks), "输出": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
