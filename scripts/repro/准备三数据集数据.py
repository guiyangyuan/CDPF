#!/usr/bin/env python3
"""下载并核验 Caltech101、OxfordPets、Food101 的官方数据与 Zhou 固定划分。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import zipfile


数据集 = {
    "Caltech101": {
        "目录": "caltech-101",
        "图像目录": "101_ObjectCategories",
        "归档": [
            {
                "文件": "caltech-101.zip",
                "网址": "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1",
                "MD5": "3138e1922a9193bfa496528edbbc45d0",
                "格式": "zip",
                "解压目标": "dataset",
            }
        ],
        "划分文件": "split_zhou_Caltech101.json",
        "划分网址": "https://drive.usercontent.google.com/download?id=1hyarUivQE36mY6jSomru6Fjd-JzwcCzN&export=download&confirm=t",
        "划分数量": {"train": 4128, "val": 1649, "test": 2465},
        "类别数": 100,
    },
    "OxfordPets": {
        "目录": "oxford_pets",
        "图像目录": "images",
        "归档": [
            {
                "文件": "oxford-pets-images.tar.gz",
                "网址": "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz",
                "MD5": "5c4f3ee8e5d25df40f4fd59a7f44e54c",
                "格式": "tar",
                "解压目标": "dataset",
            },
            {
                "文件": "oxford-pets-annotations.tar.gz",
                "网址": "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz",
                "MD5": "95a8c909bbe2e81eed6a22bccdf3f68",
                "格式": "tar",
                "解压目标": "dataset",
            },
        ],
        "划分文件": "split_zhou_OxfordPets.json",
        "划分网址": "https://drive.usercontent.google.com/download?id=1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs&export=download&confirm=t",
        "划分SHA256": "d7f3ffe65b3919f95132e51422894ae18686b83c331fd8c0adcdc3dc2de6301e",
        "划分数量": {"train": 2944, "val": 736, "test": 3669},
        "类别数": 37,
    },
    "Food101": {
        "目录": "food-101",
        "图像目录": "images",
        "归档": [
            {
                "文件": "food-101.tar.gz",
                "网址": "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
                "MD5": "85eeb15f3717b99a5da872d97d918f87",
                "格式": "tar",
                "解压目标": "root",
            }
        ],
        "划分文件": "split_zhou_Food101.json",
        "划分网址": "https://drive.usercontent.google.com/download?id=1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl&export=download&confirm=t",
        "划分数量": {"train": 50500, "val": 20200, "test": 30300},
        "类别数": 101,
    },
}


def 文件摘要(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def 原子写json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def 下载(url: str, target: Path) -> None:
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    command = [
        "curl", "-L", "--fail", "--retry", "5", "--retry-delay", "5",
        "--connect-timeout", "30", "-C", "-", "--output", str(partial), url,
    ]
    subprocess.run(command, check=True)
    partial.replace(target)


def _安全成员(name: str) -> bool:
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts


def 解压归档(archive: Path, fmt: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if fmt == "zip":
        with zipfile.ZipFile(archive) as bundle:
            unsafe = [item.filename for item in bundle.infolist() if not _安全成员(item.filename)]
            if unsafe:
                raise ValueError(f"ZIP含不安全路径：{unsafe[:3]}")
            bundle.extractall(destination)
    elif fmt == "tar":
        with tarfile.open(archive, "r:gz") as bundle:
            unsafe = [item.name for item in bundle.getmembers() if not _安全成员(item.name)]
            if unsafe:
                raise ValueError(f"TAR含不安全路径：{unsafe[:3]}")
            bundle.extractall(destination)
    else:
        raise ValueError(f"未知归档格式：{fmt}")


def 补齐Caltech内容(dataset_root: Path) -> None:
    """CaltechDATA 的 ZIP 内部仍包含一层图像 TAR，需要再展开一次。"""
    image_root = dataset_root / "101_ObjectCategories"
    if image_root.is_dir():
        return
    candidates = (
        dataset_root / "101_ObjectCategories.tar.gz",
        dataset_root / "caltech-101" / "101_ObjectCategories.tar.gz",
    )
    nested = next((path for path in candidates if path.is_file()), None)
    if nested is None:
        raise FileNotFoundError("Caltech101官方ZIP中缺少101_ObjectCategories.tar.gz")
    解压归档(nested, "tar", dataset_root)
    if not image_root.is_dir():
        raise FileNotFoundError("Caltech101嵌套图像归档解压后仍缺少101_ObjectCategories")


def 下载并解压(data_root: Path, archive_root: Path) -> list[dict]:
    receipts = []
    for name, spec in 数据集.items():
        dataset_root = data_root / spec["目录"]
        image_root = dataset_root / spec["图像目录"]
        for archive_spec in spec["归档"]:
            archive = archive_root / archive_spec["文件"]
            if not image_root.is_dir() or (name == "OxfordPets" and not (dataset_root / "annotations").is_dir()):
                下载(archive_spec["网址"], archive)
                actual_md5 = 文件摘要(archive, "md5")
                if actual_md5 != archive_spec["MD5"]:
                    raise ValueError(f"{archive.name} MD5错误：{actual_md5}")
                destination = data_root if archive_spec["解压目标"] == "root" else dataset_root
                解压归档(archive, archive_spec["格式"], destination)
            if archive.is_file():
                receipts.append(
                    {
                        "数据集": name,
                        "文件": str(archive),
                        "字节数": archive.stat().st_size,
                        "MD5": 文件摘要(archive, "md5"),
                        "SHA256": 文件摘要(archive, "sha256"),
                    }
                )
        if name == "Caltech101":
            补齐Caltech内容(dataset_root)
        split_path = dataset_root / spec["划分文件"]
        下载(spec["划分网址"], split_path)
    return receipts


def 核验一个(data_root: Path, name: str, spec: dict) -> dict:
    dataset_root = data_root / spec["目录"]
    image_root = dataset_root / spec["图像目录"]
    split_path = dataset_root / spec["划分文件"]
    if not image_root.is_dir() or not split_path.is_file():
        raise FileNotFoundError(f"{name}目录或固定划分缺失：{dataset_root}")
    split_sha = 文件摘要(split_path, "sha256")
    if spec.get("划分SHA256") and split_sha != spec["划分SHA256"]:
        raise ValueError(f"{name}固定划分SHA256错误：{split_sha}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if set(payload) != {"train", "val", "test"}:
        raise ValueError(f"{name}固定划分字段错误")
    counts = {part: len(payload[part]) for part in ("train", "val", "test")}
    if counts != spec["划分数量"]:
        raise ValueError(f"{name}固定划分数量错误：{counts}")
    rows = [row for part in ("train", "val", "test") for row in payload[part]]
    if any(not isinstance(row, list) or len(row) != 3 for row in rows):
        raise ValueError(f"{name}固定划分行格式错误")
    paths = [row[0] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{name}训练/验证/测试图像路径重复")
    labels = {int(row[1]) for row in rows}
    if labels != set(range(spec["类别数"])):
        raise ValueError(f"{name}类别标签不是连续的0..{spec['类别数'] - 1}")
    missing = [path for path in paths if not (image_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"{name}缺少{len(missing)}张划分图像，第一张：{missing[0]}")
    if name == "OxfordPets":
        annotations = dataset_root / "annotations"
        for filename in ("trainval.txt", "test.txt"):
            if not (annotations / filename).is_file():
                raise FileNotFoundError(f"OxfordPets缺少标注：{filename}")
    return {
        "数据集": name,
        "目录": str(dataset_root),
        "类别数": len(labels),
        "划分数量": counts,
        "唯一划分图像数": len(paths),
        "划分SHA256": split_sha,
        "所有路径存在": True,
        "划分互斥": True,
    }


def 核验(data_root: Path) -> dict:
    results = [核验一个(data_root, name, spec) for name, spec in 数据集.items()]
    return {"状态": "通过", "数据根目录": str(data_root.resolve()), "数据集": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("计划")
    check_parser = subparsers.add_parser("核验")
    download_parser = subparsers.add_parser("下载并核验")
    for item in (check_parser, download_parser):
        item.add_argument("--data-root", type=Path, required=True)
        item.add_argument("--receipt", type=Path, required=True)
    download_parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "计划":
        print(json.dumps(数据集, ensure_ascii=False, indent=2))
        return 0
    if args.command == "下载并核验":
        archives = 下载并解压(args.data_root.resolve(), args.archive_root.resolve())
    else:
        archives = []
    result = 核验(args.data_root.resolve())
    result["归档"] = archives
    result["磁盘"] = shutil.disk_usage(args.data_root.resolve())._asdict()
    原子写json(args.receipt, result)
    print(json.dumps({"状态": "通过", "数据集数": 3, "收据": str(args.receipt)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
