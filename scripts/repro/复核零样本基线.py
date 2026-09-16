#!/usr/bin/env python3
"""独立复核下载后的五数据集 Zero-shot CLIP 证据包。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tarfile
from pathlib import Path, PurePosixPath


数据集顺序 = ["UCF101", "DTD", "FGVC-Aircraft", "EuroSAT", "OxfordFlowers"]
分支 = {"base": "基础测试", "new": "新类别测试"}
准确率模式 = re.compile(r"^\* accuracy:\s*([0-9.]+)%$", re.MULTILINE)
正确数模式 = re.compile(r"^\* correct:\s*([0-9,]+)$", re.MULTILINE)
总数模式 = re.compile(r"^\* total:\s*([0-9,]+)$", re.MULTILINE)


def 摘要(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def 读取_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def 调和平均(base: float, novel: float) -> float:
    return 0.0 if base + novel == 0 else 2.0 * base * novel / (base + novel)


def 解析日志(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    accuracy = 准确率模式.findall(text)
    correct = 正确数模式.findall(text)
    total = 总数模式.findall(text)
    if len(accuracy) != 1 or len(correct) != 1 or len(total) != 1:
        raise ValueError(f"日志指标数量异常：{path}")
    correct_value = int(correct[0].replace(",", ""))
    total_value = int(total[0].replace(",", ""))
    recomputed = 100.0 * correct_value / total_value
    if not math.isclose(float(accuracy[0]), recomputed, rel_tol=0.0, abs_tol=0.00051):
        raise ValueError(f"日志准确率不能由 correct/total 复算：{path}")
    return {
        "正确数": correct_value,
        "总数": total_value,
        "日志打印准确率": float(accuracy[0]),
        "复算准确率": recomputed,
        "SHA256": 摘要(path),
    }


def 核验归档(path: Path, expected_sha: str | None) -> dict:
    actual_sha = 摘要(path)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"归档SHA256不一致：{actual_sha}")
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        unsafe = []
        for member in members:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts:
                unsafe.append(member.name)
        if unsafe:
            raise ValueError(f"归档含不安全路径：{unsafe}")
    return {"SHA256": actual_sha, "字节数": path.stat().st_size, "成员数": len(members)}


def 核验文件清单(evidence_dir: Path) -> dict:
    manifest_path = evidence_dir / "文件清单.txt"
    entries = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        target = evidence_dir / relative
        if not target.is_file():
            raise FileNotFoundError(f"清单文件缺失：{relative}")
        actual = 摘要(target)
        if actual != expected:
            raise ValueError(f"清单SHA256不一致：{relative}")
        entries.append(relative)
    all_files = {
        path.relative_to(evidence_dir).as_posix()
        for path in evidence_dir.rglob("*")
        if path.is_file() and path.name != "文件清单.txt"
    }
    local_metadata = {"独立复核.json", "独立复核报告.md", "备份收据.json", "关机确认.json"}
    unlisted = all_files - set(entries)
    if set(entries) - all_files or unlisted - local_metadata:
        raise ValueError("文件清单与证据目录的原始文件集合不一致")
    return {
        "清单记录数": len(entries),
        "已核验原始文件数": len(entries),
        "允许的本地补充文件": sorted(unlisted),
    }


def 核验结果(evidence_dir: Path) -> tuple[list[dict], dict]:
    plan = 读取_json(evidence_dir / "执行计划.json")
    status = 读取_json(evidence_dir / "执行状态.json")
    result = 读取_json(evidence_dir / "结果.json")
    if plan.get("协议") != "投稿Zero-shot-CLIP五数据集-v1" or plan.get("命令数") != 10:
        raise ValueError("执行计划身份或命令数错误")
    if status.get("状态") != "完成" or status.get("已完成命令") != 10:
        raise ValueError("执行状态未完成")
    if any(item.get("退出码") != 0 for item in status.get("命令", [])):
        raise ValueError("存在失败命令")
    if result.get("状态") != "通过" or result.get("检查", {}).get("训练检查点数量") != 0:
        raise ValueError("结果状态或零检查点检查失败")
    if list(evidence_dir.rglob("*.pth.tar*")):
        raise ValueError("证据目录中出现了训练检查点")

    recorded_branches = {
        (item["数据集"], item["分支"]): item for item in result["逐分支复算"]
    }
    recorded_summaries = {item["数据集"]: item for item in result["结果"]}
    rows = []
    for dataset in 数据集顺序:
        values = {}
        for branch, dirname in 分支.items():
            log_path = evidence_dir / dataset / dirname / "log.txt"
            parsed = 解析日志(log_path)
            recorded = recorded_branches[(dataset, branch)]
            for key in ("正确数", "总数", "日志SHA256"):
                parsed_key = "SHA256" if key == "日志SHA256" else key
                if parsed[parsed_key] != recorded[key]:
                    raise ValueError(f"结果JSON与日志不一致：{dataset}/{branch}/{key}")
            if not math.isclose(parsed["复算准确率"], recorded["复算准确率"], abs_tol=1e-12):
                raise ValueError(f"复算准确率不一致：{dataset}/{branch}")
            values[branch] = parsed["复算准确率"]
            rows.append({"数据集": dataset, "分支": branch, **parsed})
        summary = recorded_summaries[dataset]
        expected_hm = 调和平均(values["base"], values["new"])
        if not all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in (
                (summary["Base"], values["base"]),
                (summary["Novel"], values["new"]),
                (summary["HM"], expected_hm),
            )
        ):
            raise ValueError(f"汇总结果不一致：{dataset}")

    return rows, {
        "GPU": status["GPU"],
        "总耗时秒": status["总耗时秒"],
        "命令数": len(status["命令"]),
        "分支结果数": len(rows),
        "数据集数": len(recorded_summaries),
        "训练检查点数": 0,
        "CLIP_SHA256": result["检查"]["CLIP_SHA256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    archive = 核验归档(args.archive.resolve(), args.expected_archive_sha256)
    evidence_dir = args.evidence_dir.resolve()
    manifest = 核验文件清单(evidence_dir)
    rows, checks = 核验结果(evidence_dir)
    payload = {
        "状态": "通过",
        "复核方式": "从下载归档、文件清单和十份原始log.txt独立复算，不信任结果表中的手填数值",
        "归档": archive,
        "文件清单": manifest,
        "检查": checks,
        "逐分支独立复算": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"状态": "通过", "复核分支": len(rows), "输出": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
