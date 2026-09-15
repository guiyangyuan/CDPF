#!/usr/bin/env python3
"""下载并核验 OxfordFlowers 官方图像、标签、类别名和固定 Zhou 划分。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


文件 = {
    "图像归档": {
        "名称": "102flowers.tgz",
        "网址": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
        "算法": "md5",
        "摘要": "52808999861908f626f3c1f4e79d11fa",
    },
    "图像标签": {
        "名称": "imagelabels.mat",
        "网址": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
        "算法": "md5",
        "摘要": "e0620be6f572b9609742df49c70aed4d",
    },
    "类别名称": {
        "名称": "cat_to_name.json",
        "网址": "https://raw.githubusercontent.com/linzhiqiu/cross_modal_adaptation/main/splits/cat_to_name.json",
        "算法": "sha256",
        "摘要": "0469f1ad7fe0072d98155c49aa58649d998edbbe3326229003aa5f7228525236",
    },
    "固定划分": {
        "名称": "split_zhou_OxfordFlowers.json",
        "网址": "https://raw.githubusercontent.com/linzhiqiu/cross_modal_adaptation/main/splits/split_zhou_OxfordFlowers.json",
        "算法": "sha256",
        "摘要": "2f50c42fc1a17ca13c271cb4269b1347bdb43d2be672e34f74a4e4cbfbcca794",
    },
}


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(url: str, path: Path) -> None:
    temporary = path.with_name(path.name + ".part")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "8",
            "--retry-all-errors",
            "--retry-delay",
            "3",
            "--connect-timeout",
            "20",
            "--speed-time",
            "120",
            "--speed-limit",
            "1024",
            "--continue-at",
            "-",
            "--user-agent",
            "textconsistency-dataset-preparation/1.0",
            "--output",
            str(temporary),
            url,
        ],
        check=True,
    )
    temporary.replace(path)


def ensure_file(spec: dict, path: Path) -> dict:
    if path.is_file() and digest(path, spec["算法"]) != spec["摘要"]:
        raise ValueError(f"已有文件摘要不匹配，拒绝覆盖: {path}")
    if not path.is_file():
        download(spec["网址"], path)
    actual = digest(path, spec["算法"])
    if actual != spec["摘要"]:
        raise ValueError(f"下载文件摘要不匹配: {path}; {actual}")
    return {"路径": str(path), "字节数": path.stat().st_size, spec["算法"].upper(): actual, "网址": spec["网址"]}


def extract_images(archive: Path, dataset_dir: Path) -> None:
    image_dir = dataset_dir / "jpg"
    if image_dir.is_dir() and len(list(image_dir.glob("*.jpg"))) == 8189:
        return
    if image_dir.exists():
        raise ValueError(f"图像目录存在但数量不完整，拒绝覆盖: {image_dir}")
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        unsafe = [member.name for member in members if Path(member.name).is_absolute() or ".." in Path(member.name).parts]
        if unsafe:
            raise ValueError(f"图像归档含不安全路径: {unsafe[0]}")
        if not members or any(Path(member.name).parts[0] != "jpg" for member in members):
            raise ValueError("图像归档结构不是预期的jpg目录")
        bundle.extractall(dataset_dir, filter="data")


def verify_dataset(dataset_dir: Path, receipts: dict) -> dict:
    image_dir = dataset_dir / "jpg"
    image_names = {path.name for path in image_dir.glob("*.jpg")}
    if len(image_names) != 8189:
        raise ValueError(f"图像数量应为8189，实际为{len(image_names)}")

    class_names = json.loads((dataset_dir / "cat_to_name.json").read_text())
    if set(class_names) != {str(index) for index in range(1, 103)}:
        raise ValueError("类别名文件不是完整的102类")
    split = json.loads((dataset_dir / "split_zhou_OxfordFlowers.json").read_text())
    if set(split) != {"train", "val", "test"}:
        raise ValueError("固定划分必须只含train/val/test")

    paths = {}
    label_names = set()
    missing = []
    for name in ("train", "val", "test"):
        paths[name] = set()
        for row in split[name]:
            if len(row) != 3:
                raise ValueError(f"{name}划分行格式错误")
            relative_path, label, classname = row
            paths[name].add(relative_path)
            label_names.add((int(label), classname))
            if Path(relative_path).name not in image_names:
                missing.append(relative_path)
    if missing:
        raise FileNotFoundError(f"固定划分缺少{len(missing)}张图像，首项: {missing[0]}")
    overlaps = {
        "训练验证": len(paths["train"] & paths["val"]),
        "训练测试": len(paths["train"] & paths["test"]),
        "验证测试": len(paths["val"] & paths["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"固定划分存在路径重叠: {overlaps}")
    if len(set.union(*paths.values())) != 8189:
        raise ValueError("固定划分没有恰好覆盖8189张图像")
    if len(label_names) != 102:
        raise ValueError(f"固定划分类别数应为102，实际为{len(label_names)}")

    return {
        "状态": "通过",
        "数据集": "OxfordFlowers",
        "目录": str(dataset_dir),
        "文件": receipts,
        "图像数": len(image_names),
        "类别数": len(label_names),
        "划分数量": {name: len(split[name]) for name in ("train", "val", "test")},
        "划分互不重叠": True,
        "划分覆盖全部图像": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    dataset_dir = data_root / "oxford_flowers"
    archive_dir = data_root / "archives"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "图像归档": archive_dir / 文件["图像归档"]["名称"],
        "图像标签": dataset_dir / 文件["图像标签"]["名称"],
        "类别名称": dataset_dir / 文件["类别名称"]["名称"],
        "固定划分": dataset_dir / 文件["固定划分"]["名称"],
    }
    receipts = {name: ensure_file(文件[name], path) for name, path in targets.items()}
    extract_images(targets["图像归档"], dataset_dir)
    result = verify_dataset(dataset_dir, receipts)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
