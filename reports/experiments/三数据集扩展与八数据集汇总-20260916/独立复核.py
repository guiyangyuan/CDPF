#!/usr/bin/env python3
"""独立复核三数据集归档，并与已审计五数据集结果合并。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import tarfile
import tempfile
from pathlib import Path


数据集顺序 = [
    "UCF101",
    "DTD",
    "FGVC-Aircraft",
    "EuroSAT",
    "OxfordFlowers",
    "Caltech101",
    "OxfordPets",
    "Food101",
]
新增数据集 = ["Caltech101", "OxfordPets", "Food101"]
方法顺序 = ["Zero-shot CLIP", "CoOp", "固定均值概率融合", "CoCoOp"]
融合字段 = "mean_probability_fusion"
容差 = 1e-8


def 读取_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def 写入_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hm(base: float, novel: float) -> float:
    return 2 * base * novel / (base + novel)


def 接近(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=容差):
        raise AssertionError(f"{label}: {actual} != {expected}")


def 安全解压(archive: Path, target: Path) -> None:
    target_root = target.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            destination = (target / member.name).resolve()
            if target_root != destination and target_root not in destination.parents:
                raise ValueError(f"归档路径越界: {member.name}")
        tf.extractall(target)


def 复核状态(status: dict, expected_count: int, count_key: str) -> None:
    assert status["状态"] == "完成"
    assert status[count_key] == expected_count
    assert all(item["退出码"] == 0 for item in status["命令"] if "退出码" in item)


def 解析准确率日志(path: Path) -> tuple[int, int, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    totals = re.findall(r"\* total:\s*([0-9,]+)", text)
    corrects = re.findall(r"\* correct:\s*([0-9,]+)", text)
    accuracies = re.findall(r"\* accuracy:\s*([0-9.]+)%", text)
    if not totals or not corrects or not accuracies:
        raise AssertionError(f"未在日志中找到准确率: {path}")
    total = int(totals[-1].replace(",", ""))
    correct = int(corrects[-1].replace(",", ""))
    reported = float(accuracies[-1])
    computed = correct / total * 100
    if abs(computed - reported) > 0.00051:
        raise AssertionError(f"日志显示与计数不符: {path}")
    return correct, total, computed


def 复算预测(path: Path) -> tuple[int, dict[str, float]]:
    counts: dict[str, int] = {}
    total = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            total += 1
            label = row["label"]
            for method, prediction in row["predictions"].items():
                counts[method] = counts.get(method, 0) + int(prediction["label"] == label)
    assert total > 0
    return total, {method: correct / total * 100 for method, correct in counts.items()}


def 复核零样本(root: Path) -> tuple[dict, list[dict]]:
    run = root / "runs/投稿三数据集零样本基线-20260916"
    status = 读取_json(run / "执行状态.json")
    复核状态(status, 6, "已完成命令")
    result = 读取_json(run / "结果.json")
    assert result["状态"] == "通过"
    raw_rows = {(x["数据集"], x["分支"]): x for x in result["逐分支复算"]}
    rows = []
    for item in result["结果"]:
        dataset = item["数据集"]
        for branch, key, filename in (
            ("base", "Base", f"{dataset}/基础测试/log.txt"),
            ("new", "Novel", f"{dataset}/新类别测试/log.txt"),
        ):
            raw = raw_rows[(dataset, branch)]
            log = run / filename
            assert sha256(log) == raw["日志SHA256"]
            correct, total, accuracy = 解析准确率日志(log)
            assert (correct, total) == (raw["正确数"], raw["总数"])
            接近(accuracy, item[key], f"Zero-shot {dataset} {branch}")
        接近(hm(item["Base"], item["Novel"]), item["HM"], f"Zero-shot {dataset} HM")
        rows.append({"数据集": dataset, "方法": "Zero-shot CLIP", **{k: item[k] for k in ("Base", "Novel", "HM")}})
    return status, rows


def 复核主方法(root: Path) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    run = root / "runs/投稿三数据集主方法-20260916-v2"
    status = 读取_json(run / "执行状态.json")
    assert status["状态"] == "完成"
    assert status["已完成数据集"] == status["总数据集"] == 3
    assert all(item["退出码"] == 0 for item in status["运行"])
    dataset_rows: list[dict] = []
    seed_rows: list[dict] = []
    checkpoint_rows: list[dict] = []
    paired_rows: list[dict] = []
    method_map = {"student": "CoOp", 融合字段: "固定均值概率融合"}
    for dataset in 新增数据集:
        summary = 读取_json(run / dataset / "结果汇总.json")
        assert summary["通过"] is True
        per_seed = {int(x["种子"]): x for x in summary["逐种子"]}
        observed: dict[str, dict[str, list[float]]] = {
            method: {"Base": [], "Novel": []}
            for method in ("student", "description_mean", "simple_fusion", 融合字段)
        }
        for seed in (1, 2, 3):
            checkpoint = run / dataset / f"基线训练/baseline/seed{seed}/textconsistency/model.pth.tar-20"
            assert checkpoint.is_file()
            checkpoint_rows.append({"数据集": dataset, "种子": seed, "SHA256": sha256(checkpoint)})
            branch_values: dict[str, dict[str, float]] = {}
            for branch, key in (("base", "Base"), ("new", "Novel")):
                prediction_file = run / dataset / f"融合评估/seed{seed}/{branch}/predictions.jsonl"
                total, values = 复算预测(prediction_file)
                branch_values[key] = values
                for method in observed:
                    expected = per_seed[seed]["方法"][method][key]
                    接近(values[method], expected, f"主方法 {dataset} seed{seed} {branch} {method}")
                    observed[method][key].append(values[method])
                seed_rows.append({
                    "数据集": dataset,
                    "种子": seed,
                    "分支": branch,
                    "样本数": total,
                    "CoOp": values["student"],
                    "固定均值概率融合": values[融合字段],
                })
            for internal, display in method_map.items():
                base = branch_values["Base"][internal]
                novel = branch_values["Novel"][internal]
                expected_hm = per_seed[seed]["方法"][internal]["HM"]
                接近(hm(base, novel), expected_hm, f"主方法 {dataset} seed{seed} {display} HM")
            coop_hm = hm(branch_values["Base"]["student"], branch_values["Novel"]["student"])
            fusion_hm = hm(branch_values["Base"][融合字段], branch_values["Novel"][融合字段])
            assert fusion_hm > coop_hm
            paired_rows.append({
                "数据集": dataset,
                "种子": seed,
                "CoOp HM": coop_hm,
                "固定均值概率融合HM": fusion_hm,
                "HM增益pp": fusion_hm - coop_hm,
            })
        for internal in observed:
            base = statistics.mean(observed[internal]["Base"])
            novel = statistics.mean(observed[internal]["Novel"])
            expected = summary["均值"][internal]
            接近(base, expected["Base均值"], f"主方法 {dataset} {internal} Base均值")
            接近(novel, expected["Novel均值"], f"主方法 {dataset} {internal} Novel均值")
            接近(hm(base, novel), expected["均值的HM"], f"主方法 {dataset} {internal} HM")
        for internal, display in method_map.items():
            mean = summary["均值"][internal]
            dataset_rows.append({
                "数据集": dataset,
                "方法": display,
                "Base": mean["Base均值"],
                "Novel": mean["Novel均值"],
                "HM": mean["均值的HM"],
            })
    assert len(checkpoint_rows) == 9
    assert len(paired_rows) == 9
    return status, dataset_rows, seed_rows, checkpoint_rows, paired_rows


def 复核_cocoop(root: Path) -> tuple[dict, list[dict], list[dict]]:
    run = root / "runs/投稿三数据集联合上下文基线-20260916"
    stage = run / "新增三数据集"
    status = 读取_json(stage / "执行状态.json")
    复核状态(status, 27, "已完成命令")
    result = 读取_json(run / "三数据集最终结果.json")
    assert result["状态"] == "通过"
    seeds = []
    for item in result["逐种子"]:
        dataset, seed = item["数据集"], int(item["种子"])
        checkpoint = stage / dataset / f"种子{seed}/训练/prompt_learner/model.pth.tar-10"
        base_log = stage / dataset / f"种子{seed}/基础测试/log.txt"
        novel_log = stage / dataset / f"种子{seed}/新类别测试/log.txt"
        assert sha256(checkpoint) == item["检查点SHA256"]
        assert sha256(base_log) == item["基础测试日志SHA256"]
        assert sha256(novel_log) == item["新类别测试日志SHA256"]
        _, _, base = 解析准确率日志(base_log)
        _, _, novel = 解析准确率日志(novel_log)
        接近(base, item["Base"], f"CoCoOp {dataset} seed{seed} Base")
        接近(novel, item["Novel"], f"CoCoOp {dataset} seed{seed} Novel")
        接近(hm(base, novel), item["HM"], f"CoCoOp {dataset} seed{seed} HM")
        seeds.append(item)
    rows = []
    for item in result["数据集汇总"]:
        dataset = item["数据集"]
        own = [x for x in seeds if x["数据集"] == dataset]
        assert len(own) == 3
        base = statistics.mean(x["Base"] for x in own)
        novel = statistics.mean(x["Novel"] for x in own)
        接近(base, item["Base均值"], f"CoCoOp {dataset} Base均值")
        接近(novel, item["Novel均值"], f"CoCoOp {dataset} Novel均值")
        接近(hm(base, novel), item["HM_均值计算"], f"CoCoOp {dataset} HM")
        rows.append({"数据集": dataset, "方法": "CoCoOp", "Base": base, "Novel": novel, "HM": hm(base, novel)})
    return status, rows, seeds


def 合并八数据集(repo: Path, new_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    old = 读取_json(repo / "reports/experiments/投稿实验统一结果-20260915/统一结果.json")
    rows: list[dict] = []
    for item in old["Zero-shot CLIP"]["五数据集结果"]:
        rows.append({"数据集": item["数据集"], "方法": "Zero-shot CLIP", **{k: item[k] for k in ("Base", "Novel", "HM")}})
    for item in old["方法汇总"]:
        if item["方法"] in ("CoOp", "固定均值概率融合"):
            rows.append({
                "数据集": item["数据集"],
                "方法": item["方法"],
                "Base": item["Base均值"],
                "Novel": item["Novel均值"],
                "HM": item["HM_均值计算"],
            })
    for item in old["CoCoOp五数据集"]["数据集汇总"]:
        rows.append({
            "数据集": item["数据集"],
            "方法": "CoCoOp",
            "Base": item["Base均值"],
            "Novel": item["Novel均值"],
            "HM": item["HM_均值计算"],
        })
    rows.extend(new_rows)
    identities = {(x["数据集"], x["方法"]) for x in rows}
    assert len(rows) == len(identities) == 32
    assert identities == {(d, m) for d in 数据集顺序 for m in 方法顺序}
    summaries = []
    for scope, selected in (("原五数据集", 数据集顺序[:5]), ("新增三数据集", 新增数据集), ("全部八数据集", 数据集顺序)):
        for method in 方法顺序:
            own = [x for x in rows if x["数据集"] in selected and x["方法"] == method]
            assert len(own) == len(selected)
            summaries.append({
                "范围": scope,
                "方法": method,
                "数据集数": len(own),
                "Base等权均值": statistics.mean(x["Base"] for x in own),
                "Novel等权均值": statistics.mean(x["Novel"] for x in own),
                "HM等权均值": statistics.mean(x["HM"] for x in own),
            })
    return rows, summaries


def 写入_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def main() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=here / "完整实验归档.tar.gz")
    parser.add_argument("--output", type=Path, default=here / "独立复核结果.json")
    args = parser.parse_args()
    archive = args.archive.resolve()
    assert archive.is_file()
    with tempfile.TemporaryDirectory(prefix="三数据集扩展复核-") as tmp:
        root = Path(tmp)
        安全解压(archive, root)
        zero_status, zero_rows = 复核零样本(root)
        main_status, main_rows, seed_rows, main_checkpoints, paired_rows = 复核主方法(root)
        cocoop_status, cocoop_rows, cocoop_seeds = 复核_cocoop(root)
        new_rows = zero_rows + main_rows + cocoop_rows
        all_rows, summaries = 合并八数据集(repo, new_rows)
        fusion = {(x["数据集"]): x for x in all_rows if x["方法"] == "固定均值概率融合"}
        cocoop = {(x["数据集"]): x for x in all_rows if x["方法"] == "CoCoOp"}
        comparison = [
            {
                "数据集": dataset,
                "主方法HM": fusion[dataset]["HM"],
                "CoCoOp HM": cocoop[dataset]["HM"],
                "主方法相对CoCoOp HM增益pp": fusion[dataset]["HM"] - cocoop[dataset]["HM"],
            }
            for dataset in 数据集顺序
        ]
        assert all(x["主方法相对CoCoOp HM增益pp"] > 0 for x in comparison)
        result = {
            "状态": "通过",
            "结论": "三套计划全部完成；新增三数据集逐样本或逐日志复算通过；主方法在八个数据集的HM均高于本地CoCoOp。",
            "归档": {
                "文件": archive.name,
                "字节数": archive.stat().st_size,
                "SHA256": sha256(archive),
            },
            "执行完整性": {
                "Zero-shot": {"状态": zero_status["状态"], "完成命令": zero_status["已完成命令"], "总命令": zero_status["总命令"]},
                "主方法": {"状态": main_status["状态"], "完成数据集": main_status["已完成数据集"], "总数据集": main_status["总数据集"]},
                "CoCoOp": {"状态": cocoop_status["状态"], "完成命令": cocoop_status["已完成命令"], "总命令": cocoop_status["总命令"]},
            },
            "主方法检查点": main_checkpoints,
            "主方法新增三数据集逐种子比较": paired_rows,
            "CoCoOp逐种子": cocoop_seeds,
            "新增三数据集结果": new_rows,
            "八数据集逐数据集结果": all_rows,
            "范围汇总": summaries,
            "主方法相对CoCoOp": comparison,
            "复核规则": [
                "Zero-shot从原始日志correct/total复算",
                "主方法从predictions.jsonl逐样本复算四种预测",
                "CoCoOp核对九个检查点和十八份测试日志SHA256，并从correct/total复算",
                "所有HM按2*Base*Novel/(Base+Novel)复算",
                "八数据集合并仅使用本次归档和仓库中已审计五数据集结果",
            ],
        }
        写入_json(args.output, result)
        写入_csv(here / "新增三数据集结果.csv", sorted(new_rows, key=lambda x: (新增数据集.index(x["数据集"]), 方法顺序.index(x["方法"]))), ["数据集", "方法", "Base", "Novel", "HM"])
        写入_csv(here / "八数据集结果.csv", sorted(all_rows, key=lambda x: (数据集顺序.index(x["数据集"]), 方法顺序.index(x["方法"]))), ["数据集", "方法", "Base", "Novel", "HM"])
        写入_csv(here / "八数据集汇总.csv", summaries, ["范围", "方法", "数据集数", "Base等权均值", "Novel等权均值", "HM等权均值"])
        写入_csv(here / "主方法新增三数据集逐种子复算.csv", seed_rows, ["数据集", "种子", "分支", "样本数", "CoOp", "固定均值概率融合"])
    print(json.dumps({"状态": "通过", "输出": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
