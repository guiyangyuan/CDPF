#!/usr/bin/env python3
"""下载并严格核验 StanfordCars 与 SUN397 的 Zhou 固定划分图像。

SUN397 原始压缩包约 39 GB。为适配 AutoDL 50 GB 数据盘，本脚本流式读取
压缩包，只落盘固定 Base-to-Novel 划分实际使用的 39,700 张图像，不保存整包。
"""

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
    "StanfordCars": {
        "目录": "stanford_cars",
        "图像目录": ".",
        "划分文件": "split_zhou_StanfordCars.json",
        "划分网址": "https://drive.usercontent.google.com/download?id=1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT&export=download&confirm=t",
        "划分SHA256": "40acee7aa65c8e4727a32d90e884b7c3f8960b523422317e163f7af65432959d",
        "划分数量": {"train": 6509, "val": 1635, "test": 8041},
        "类别数": 196,
        "归档网址": "https://github.com/zhengli97/PromptKD/releases/download/datasets/stanford_cars.zip",
        "归档文件": "stanford_cars.zip",
        "归档字节数": 1957803555,
        "归档方式": "下载ZIP后只解压固定划分所需图像；成功后默认删除ZIP",
    },
    "SUN397": {
        "目录": "sun397",
        "图像目录": "SUN397",
        "划分文件": "split_zhou_SUN397.json",
        "划分网址": "https://drive.usercontent.google.com/download?id=1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq&export=download&confirm=t",
        "划分SHA256": "9847f9ef611008073ac42adee6614bbb83f877cfbcfbd3115fc8284e0c8c1c47",
        "划分数量": {"train": 15880, "val": 3970, "test": 19850},
        "类别数": 397,
        "归档网址": "https://hf-mirror.com/datasets/Veiiaj3dad/CLIP-Prompt-learning-sun397/resolve/main/SUN397.tar.gz?download=true",
        "原始归档网址": "https://huggingface.co/datasets/Veiiaj3dad/CLIP-Prompt-learning-sun397/resolve/main/SUN397.tar.gz?download=true",
        "归档字节数": 39077296924,
        "归档SHA256": "f404130965a7ad77bed5ececc71e720ab50f3cc1e1bb257257610d38f3b928ec",
        "归档方式": "流式读取整包，仅落盘固定划分所需图像，不保存39GB压缩包",
    },
}


def 文件摘要(path: Path, algorithm: str = "sha256") -> str:
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
        "curl", "-L", "--fail", "--retry", "8", "--retry-all-errors", "--retry-delay", "10",
        "--connect-timeout", "30", "-C", "-", "--output", str(partial), url,
    ]
    subprocess.run(command, check=True)
    partial.replace(target)


def 读取并核验划分(dataset_root: Path, spec: dict) -> tuple[dict, list[str]]:
    split_path = dataset_root / spec["划分文件"]
    if not split_path.is_file():
        raise FileNotFoundError(f"缺少固定划分：{split_path}")
    actual_sha = 文件摘要(split_path)
    if actual_sha != spec["划分SHA256"]:
        raise ValueError(f"固定划分SHA256错误：{split_path}，实际为{actual_sha}")
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if set(payload) != {"train", "val", "test"}:
        raise ValueError(f"固定划分字段错误：{split_path}")
    counts = {name: len(payload[name]) for name in ("train", "val", "test")}
    if counts != spec["划分数量"]:
        raise ValueError(f"固定划分数量错误：{counts}")
    rows = [row for name in ("train", "val", "test") for row in payload[name]]
    if any(not isinstance(row, list) or len(row) != 3 for row in rows):
        raise ValueError("固定划分行格式错误")
    paths = [str(row[0]) for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("训练、验证、测试划分存在重复图像")
    labels = {int(row[1]) for row in rows}
    if labels != set(range(spec["类别数"])):
        raise ValueError("固定划分类别标签不连续")
    return payload, paths


def 下载划分(data_root: Path) -> None:
    for spec in 数据集.values():
        dataset_root = data_root / spec["目录"]
        dataset_root.mkdir(parents=True, exist_ok=True)
        split_path = dataset_root / spec["划分文件"]
        if split_path.is_file() and 文件摘要(split_path) != spec["划分SHA256"]:
            raise ValueError(f"已有划分文件摘要不匹配，拒绝覆盖：{split_path}")
        下载(spec["划分网址"], split_path)
        读取并核验划分(dataset_root, spec)


def _zip相对图像路径(name: str) -> str | None:
    parts = Path(name).parts
    for marker in ("cars_train", "cars_test"):
        if marker in parts:
            index = parts.index(marker)
            return Path(*parts[index:]).as_posix()
    return None


def _写入图像(source, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with partial.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
    partial.replace(target)


def _提取ZIP内TGZ(bundle: zipfile.ZipFile, item: zipfile.ZipInfo, dataset_root: Path, missing: set[str]) -> int:
    extracted = 0
    with bundle.open(item) as compressed, tarfile.open(fileobj=compressed, mode="r|gz") as nested:
        for member in nested:
            relative = _zip相对图像路径(member.name.removeprefix("./"))
            if relative not in missing or not member.isfile():
                continue
            source = nested.extractfile(member)
            if source is None:
                raise IOError(f"无法读取StanfordCars成员：{member.name}")
            with source:
                _写入图像(source, dataset_root / relative)
            missing.remove(relative)
            extracted += 1
    return extracted


def 准备StanfordCars(data_root: Path, archive_root: Path, keep_archive: bool) -> dict:
    spec = 数据集["StanfordCars"]
    dataset_root = data_root / spec["目录"]
    _, paths = 读取并核验划分(dataset_root, spec)
    missing = {path for path in paths if not (dataset_root / path).is_file()}
    if not missing:
        return {"数据集": "StanfordCars", "状态": "已完整，跳过下载", "图像数": len(paths)}

    archive = archive_root / spec["归档文件"]
    下载(spec["归档网址"], archive)
    if archive.stat().st_size != spec["归档字节数"]:
        raise ValueError(f"StanfordCars归档大小错误：{archive.stat().st_size}")
    archive_sha = 文件摘要(archive)
    extracted = 0
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            if Path(item.filename).name in {"cars_train.tgz", "cars_test.tgz"}:
                extracted += _提取ZIP内TGZ(bundle, item, dataset_root, missing)
                continue
            relative = _zip相对图像路径(item.filename)
            if relative not in missing or item.is_dir():
                continue
            with bundle.open(item) as source:
                _写入图像(source, dataset_root / relative)
            missing.remove(relative)
            extracted += 1
    remaining = [path for path in paths if not (dataset_root / path).is_file()]
    if remaining:
        raise FileNotFoundError(f"StanfordCars归档缺少{len(remaining)}张划分图像，第一张：{remaining[0]}")
    receipt = {
        "数据集": "StanfordCars",
        "状态": "下载并解压完成",
        "图像数": len(paths),
        "本次新增图像": extracted,
        "归档字节数": archive.stat().st_size,
        "归档SHA256": archive_sha,
        "来源": spec["归档网址"],
    }
    if not keep_archive:
        archive.unlink()
        receipt["归档处理"] = "验证与解压后删除以节省数据盘"
    return receipt


class 摘要读取器:
    def __init__(self, raw) -> None:
        self.raw = raw
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self.raw.read(size)
        self.digest.update(data)
        self.bytes_read += len(data)
        return data


def 准备SUN397(data_root: Path, input_path: Path | None = None) -> dict:
    spec = 数据集["SUN397"]
    dataset_root = data_root / spec["目录"]
    image_root = dataset_root / spec["图像目录"]
    _, paths = 读取并核验划分(dataset_root, spec)
    wanted = {f"SUN397/{path}" for path in paths}
    existing = {name for name in wanted if (dataset_root / name).is_file()}
    if len(existing) == len(wanted):
        return {"数据集": "SUN397", "状态": "已完整，跳过下载", "图像数": len(paths)}
    free_bytes = shutil.disk_usage(data_root).free
    if free_bytes < 18 * 1024**3:
        raise OSError(f"SUN397流式解压前至少需要18GiB空闲空间，当前仅{free_bytes / 1024**3:.2f}GiB")
    image_root.mkdir(parents=True, exist_ok=True)
    process = None
    if input_path is None:
        command = [
            "curl", "-L", "--fail", "--retry", "8", "--retry-all-errors", "--retry-delay", "10",
            "--connect-timeout", "30", spec["归档网址"],
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE)
        if process.stdout is None:
            raise RuntimeError("无法读取SUN397下载流")
        raw = process.stdout
        source_name = spec["归档网址"]
    else:
        raw = input_path.resolve().open("rb", buffering=0)
        source_name = str(input_path.resolve())
    reader = 摘要读取器(raw)
    found: set[str] = set()
    written = 0
    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as bundle:
            for member in bundle:
                normalized = member.name.removeprefix("./")
                if normalized not in wanted or not member.isfile():
                    continue
                found.add(normalized)
                target = dataset_root / normalized
                if target.is_file() and target.stat().st_size > 0:
                    continue
                source = bundle.extractfile(member)
                if source is None:
                    raise IOError(f"无法读取SUN397成员：{normalized}")
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(target.suffix + ".part")
                with source, partial.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=4 * 1024 * 1024)
                partial.replace(target)
                written += 1
        for _ in iter(lambda: reader.read(4 * 1024 * 1024), b""):
            pass
    except BaseException:
        if process is not None:
            process.terminate()
            process.wait()
        raise
    finally:
        raw.close()
    if process is not None:
        returncode = process.wait()
        if returncode != 0:
            raise RuntimeError(f"SUN397下载失败，curl退出码{returncode}")
    if reader.bytes_read != spec["归档字节数"]:
        raise ValueError(f"SUN397下载字节数错误：{reader.bytes_read}")
    actual_sha = reader.digest.hexdigest()
    if actual_sha != spec["归档SHA256"]:
        raise ValueError(f"SUN397流式归档SHA256错误：{actual_sha}")
    absent_in_archive = wanted - found
    if absent_in_archive:
        raise FileNotFoundError(f"SUN397归档缺少{len(absent_in_archive)}张划分图像")
    remaining = [path for path in paths if not (image_root / path).is_file()]
    if remaining:
        raise FileNotFoundError(f"SUN397流式解压后缺少{len(remaining)}张图像，第一张：{remaining[0]}")
    return {
        "数据集": "SUN397",
        "状态": "流式下载并选择性解压完成",
        "图像数": len(paths),
        "本次新增图像": written,
        "归档字节数": reader.bytes_read,
        "归档SHA256": actual_sha,
        "来源": source_name,
        "归档处理": "未落盘，只保存固定划分使用的图像",
    }


def 核验一个(data_root: Path, name: str, spec: dict) -> dict:
    dataset_root = data_root / spec["目录"]
    image_root = dataset_root / spec["图像目录"]
    payload, paths = 读取并核验划分(dataset_root, spec)
    missing = [path for path in paths if not (image_root / path).is_file()]
    empty = [path for path in paths if (image_root / path).is_file() and (image_root / path).stat().st_size == 0]
    if missing or empty:
        problem = missing[0] if missing else empty[0]
        raise FileNotFoundError(f"{name}图像不完整：缺失{len(missing)}、空文件{len(empty)}，第一项：{problem}")
    split_sets = {part: {row[0] for row in payload[part]} for part in ("train", "val", "test")}
    if any(split_sets[a] & split_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError(f"{name}划分不互斥")
    return {
        "数据集": name,
        "目录": str(dataset_root),
        "类别数": spec["类别数"],
        "划分数量": spec["划分数量"],
        "唯一划分图像数": len(paths),
        "划分SHA256": spec["划分SHA256"],
        "所有路径存在且非空": True,
        "划分互斥": True,
    }


def 核验(data_root: Path) -> dict:
    rows = [核验一个(data_root, name, spec) for name, spec in 数据集.items()]
    disk = shutil.disk_usage(data_root)
    return {
        "状态": "通过",
        "数据根目录": str(data_root.resolve()),
        "数据集": rows,
        "磁盘": {"总字节": disk.total, "已用字节": disk.used, "空闲字节": disk.free},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("计划")
    check_parser = subparsers.add_parser("核验")
    download_parser = subparsers.add_parser("下载并核验")
    for item in (check_parser, download_parser):
        item.add_argument("--data-root", type=Path, required=True)
        item.add_argument("--receipt", type=Path, required=True)
    download_parser.add_argument("--archive-root", type=Path, required=True)
    download_parser.add_argument("--sun-input", type=Path)
    download_parser.add_argument("--保留汽车归档", action="store_true")
    args = parser.parse_args()
    if args.command == "计划":
        print(json.dumps(数据集, ensure_ascii=False, indent=2))
        return 0
    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    download_receipts = []
    if args.command == "下载并核验":
        archive_root = args.archive_root.resolve()
        archive_root.mkdir(parents=True, exist_ok=True)
        下载划分(data_root)
        download_receipts.append(准备StanfordCars(data_root, archive_root, args.保留汽车归档))
        download_receipts.append(准备SUN397(data_root, args.sun_input))
    result = 核验(data_root)
    result["下载"] = download_receipts
    result["来源与存储策略"] = {
        name: {key: spec[key] for key in spec if key in {"归档网址", "原始归档网址", "归档字节数", "归档SHA256", "归档方式"}}
        for name, spec in 数据集.items()
    }
    原子写json(args.receipt.resolve(), result)
    print(json.dumps({"状态": "通过", "数据集数": 2, "收据": str(args.receipt.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
