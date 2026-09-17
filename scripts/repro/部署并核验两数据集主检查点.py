#!/usr/bin/env python3
"""安全解压并逐文件核验 DTD 与 FGVC-Aircraft 历史主检查点。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import uuid


def 文件摘要(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def 写入JSON(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def 部署(archive: Path, manifest_path: Path, target: Path) -> dict:
    archive = archive.resolve()
    manifest_path = manifest_path.resolve()
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"目标目录已经存在，拒绝覆盖：{target}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_package = manifest["恢复包"]
    actual_package_hash = 文件摘要(archive)
    if archive.stat().st_size != expected_package["字节数"]:
        raise ValueError("恢复包字节数与清单不一致")
    if actual_package_hash != expected_package["sha256"]:
        raise ValueError("恢复包 SHA-256 与清单不一致")

    expected_files = manifest["文件"]
    expected_names = set(expected_files)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("恢复包包含重复路径")
        if set(names) != expected_names:
            missing = sorted(expected_names - set(names))
            extra = sorted(set(names) - expected_names)
            raise ValueError(f"恢复包文件集合不一致；缺失={missing[:3]}，多余={extra[:3]}")
        for member in members:
            relative = PurePosixPath(member.name)
            if (
                not member.isfile()
                or relative.is_absolute()
                or ".." in relative.parts
                or len(relative.parts) < 5
                or relative.parts[0] != "formal"
                or relative.parts[1] not in {"dtd", "fgvc_aircraft"}
                or relative.parts[2] != "baseline"
                or relative.parts[3] not in {"seed1", "seed2", "seed3"}
            ):
                raise ValueError(f"恢复包包含不允许的成员：{member.name}")

        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            for member in members:
                destination = staging.joinpath(*PurePosixPath(member.name).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"无法读取恢复包成员：{member.name}")
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)

            for relative, expected in expected_files.items():
                path = staging.joinpath(*PurePosixPath(relative).parts)
                if path.stat().st_size != expected["字节数"]:
                    raise ValueError(f"文件字节数不一致：{relative}")
                if 文件摘要(path) != expected["sha256"]:
                    raise ValueError(f"文件 SHA-256 不一致：{relative}")
            staging.replace(target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    receipt = {
        "状态": "部署成功，90个文件与6个epoch-20检查点全部通过SHA-256核验",
        "训练次数": 0,
        "恢复包": str(archive),
        "恢复包sha256": actual_package_hash,
        "目标目录": str(target),
        "文件数量": len(expected_files),
        "检查点": manifest["检查点"],
        "下一步": "运行DTD与FGVC-Aircraft共12次冻结推理；禁止重新训练或覆盖历史检查点。",
    }
    receipt_path = target.parent / "两数据集主检查点恢复回执.json"
    写入JSON(receipt_path, receipt)
    return {**receipt, "回执": str(receipt_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(部署(args.archive, args.manifest, args.target_root), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
