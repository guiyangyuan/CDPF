import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[2]


def 加载模块():
    path = ROOT / "scripts/repro/准备剩余两数据集数据.py"
    spec = importlib.util.spec_from_file_location("剩余两数据集准备", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.数据集 = copy.deepcopy(module.数据集)
    return module


def 写划分(path: Path, rows: dict) -> str:
    payload = json.dumps(rows, ensure_ascii=False).encode()
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


class 剩余两数据集准备测试(unittest.TestCase):
    def test_汽车ZIP只提取固定划分图像(self):
        module = 加载模块()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            archive_root = root / "archives"
            archive_root.mkdir()
            split = {
                "train": [["cars_train/0001.jpg", 0, "car zero"]],
                "val": [],
                "test": [["cars_test/0002.jpg", 1, "car one"]],
            }
            split_path = data_root / "stanford_cars/split.json"
            split_sha = 写划分(split_path, split)
            archive = archive_root / "cars.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("stanford_cars/cars_train/0001.jpg", b"train-image")
                bundle.writestr("stanford_cars/cars_test/0002.jpg", b"test-image")
                bundle.writestr("stanford_cars/unneeded.txt", b"unused")
            module.数据集["StanfordCars"] = {
                "目录": "stanford_cars", "图像目录": ".", "划分文件": "split.json",
                "划分SHA256": split_sha, "划分数量": {"train": 1, "val": 0, "test": 1},
                "类别数": 2, "归档网址": "unused", "归档文件": "cars.zip",
                "归档字节数": archive.stat().st_size,
            }
            receipt = module.准备StanfordCars(data_root, archive_root, keep_archive=True)
            self.assertEqual(receipt["图像数"], 2)
            self.assertEqual((data_root / "stanford_cars/cars_train/0001.jpg").read_bytes(), b"train-image")
            self.assertEqual((data_root / "stanford_cars/cars_test/0002.jpg").read_bytes(), b"test-image")
            self.assertFalse((data_root / "stanford_cars/unneeded.txt").exists())
            self.assertTrue(module.核验一个(data_root, "StanfordCars", module.数据集["StanfordCars"])["划分互斥"])

    def test_汽车ZIP内TGZ只提取固定划分图像(self):
        module = 加载模块()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            archive_root = root / "archives"
            archive_root.mkdir()
            split = {
                "train": [["cars_train/0001.jpg", 0, "car zero"]],
                "val": [],
                "test": [["cars_test/0002.jpg", 1, "car one"]],
            }
            split_path = data_root / "stanford_cars/split.json"
            split_sha = 写划分(split_path, split)
            nested_archives = {}
            for partition, members in {
                "cars_train": (("cars_train/0001.jpg", b"train-image"), ("cars_train/unused.jpg", b"unused")),
                "cars_test": (("cars_test/0002.jpg", b"test-image"),),
            }.items():
                payload = io.BytesIO()
                with tarfile.open(fileobj=payload, mode="w:gz") as bundle:
                    for name, content in members:
                        info = tarfile.TarInfo(name)
                        info.size = len(content)
                        bundle.addfile(info, io.BytesIO(content))
                nested_archives[partition] = payload.getvalue()
            archive = archive_root / "cars.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for partition, payload in nested_archives.items():
                    bundle.writestr(f"stanford_cars/extra_stuff/{partition}.tgz", payload)
            module.数据集["StanfordCars"] = {
                "目录": "stanford_cars", "图像目录": ".", "划分文件": "split.json",
                "划分SHA256": split_sha, "划分数量": {"train": 1, "val": 0, "test": 1},
                "类别数": 2, "归档网址": "unused", "归档文件": "cars.zip",
                "归档字节数": archive.stat().st_size,
            }
            receipt = module.准备StanfordCars(data_root, archive_root, keep_archive=True)
            self.assertEqual(receipt["本次新增图像"], 2)
            self.assertEqual((data_root / "stanford_cars/cars_train/0001.jpg").read_bytes(), b"train-image")
            self.assertEqual((data_root / "stanford_cars/cars_test/0002.jpg").read_bytes(), b"test-image")
            self.assertFalse((data_root / "stanford_cars/cars_train/unused.jpg").exists())

    def test_SUN整包流式读取但只保存固定划分(self):
        module = 加载模块()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            split = {
                "train": [["a/a1.jpg", 0, "scene zero"]],
                "val": [],
                "test": [["b/b1.jpg", 1, "scene one"]],
            }
            split_path = data_root / "sun397/split.json"
            split_sha = 写划分(split_path, split)
            archive = root / "sun.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                for name, payload in (
                    ("SUN397/a/a1.jpg", b"scene-a"),
                    ("SUN397/b/b1.jpg", b"scene-b"),
                    ("SUN397/c/unused.jpg", b"unused"),
                ):
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    bundle.addfile(info, io.BytesIO(payload))
            module.数据集["SUN397"] = {
                "目录": "sun397", "图像目录": "SUN397", "划分文件": "split.json",
                "划分SHA256": split_sha, "划分数量": {"train": 1, "val": 0, "test": 1},
                "类别数": 2, "归档网址": archive.as_uri(), "归档字节数": archive.stat().st_size,
                "归档SHA256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
            disk = SimpleNamespace(free=30 * 1024**3)
            with mock.patch.object(module.shutil, "disk_usage", return_value=disk):
                receipt = module.准备SUN397(data_root)
            self.assertEqual(receipt["图像数"], 2)
            self.assertEqual((data_root / "sun397/SUN397/a/a1.jpg").read_bytes(), b"scene-a")
            self.assertEqual((data_root / "sun397/SUN397/b/b1.jpg").read_bytes(), b"scene-b")
            self.assertFalse((data_root / "sun397/SUN397/c/unused.jpg").exists())
            self.assertTrue(module.核验一个(data_root, "SUN397", module.数据集["SUN397"])["划分互斥"])


if __name__ == "__main__":
    unittest.main()
