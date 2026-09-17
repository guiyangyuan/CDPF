import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def 加载模块(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ImageNet投稿准备测试(unittest.TestCase):
    def test_重复显示名称仍保留四个synset身份(self):
        module = 加载模块(
            ROOT / "trainers/textconsistency_metadata.py",
            "ImageNet文本元数据",
        )
        manifest = module.view_order_manifest(
            ["missile", "missile", "sunglasses", "sunglasses"],
            [20, 20, 20, 20],
            "contiguous",
            4,
            0,
        )
        self.assertEqual(len(manifest["classes"]), 4)
        self.assertEqual(
            [row["classname"] for row in manifest["classes"]],
            ["missile", "missile", "sunglasses", "sunglasses"],
        )
        self.assertEqual([row["class_index"] for row in manifest["classes"]], [0, 1, 2, 3])

    def test_官方类别名称逐synset覆盖描述且仅有两个重复名称(self):
        module = 加载模块(
            ROOT / "scripts/repro/准备ImageNet投稿实验.py",
            "ImageNet投稿准备",
        )
        classnames = module.读取类别名称(ROOT / "assets/imagenet/classnames.txt")
        templates = json.loads((ROOT / "templates/imagenet_prompts_full.json").read_text())
        result = module.核验描述覆盖(classnames, templates, n_views=20)
        self.assertEqual(result["synset数"], 1000)
        self.assertEqual(result["唯一显示名称数"], 998)
        self.assertEqual(result["缺少描述的synset"], [])
        self.assertEqual(result["少于20条描述的synset"], [])
        self.assertEqual(result["重复显示名称"], {
            "missile": ["n03773504", "n04008634"],
            "sunglasses": ["n04355933", "n04356056"],
        })

    def test_数据目录核验拒绝缺失类别并接受完整小型夹具(self):
        module = 加载模块(
            ROOT / "scripts/repro/准备ImageNet投稿实验.py",
            "ImageNet目录核验",
        )
        classnames = {
            "n00000001": "class one",
            "n00000002": "class two",
            "n00000003": "class three",
            "n00000004": "class four",
        }
        with tempfile.TemporaryDirectory() as temporary:
            image_root = Path(temporary)
            for split in ("train", "val"):
                for synset in classnames:
                    folder = image_root / split / synset
                    folder.mkdir(parents=True)
                    count = 2 if split == "train" else 1
                    for index in range(count):
                        (folder / f"{index}.JPEG").write_bytes(b"image")
            result = module.核验图像目录(
                image_root,
                classnames,
                expected_train_images=8,
                expected_val_images=4,
                minimum_train_per_class=2,
                expected_val_per_class=1,
            )
            self.assertEqual(result["训练图像数"], 8)
            self.assertEqual(result["验证图像数"], 4)
            (image_root / "val/n00000004").rename(image_root / "val/n99999999")
            with self.assertRaisesRegex(ValueError, "类别目录"):
                module.核验图像目录(
                    image_root,
                    classnames,
                    expected_train_images=8,
                    expected_val_images=4,
                    minimum_train_per_class=2,
                    expected_val_per_class=1,
                )

    def test_执行计划固定三次训练六次融合且不搜索超参数(self):
        module = 加载模块(
            ROOT / "scripts/repro/准备ImageNet投稿实验.py",
            "ImageNet执行计划",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = module.生成执行计划(
                ROOT,
                root / "data",
                root / "results",
                Path("/opt/conda/bin/python"),
            )
        self.assertEqual(plan["协议"], "ImageNet投稿主实验-v1")
        self.assertEqual(plan["训练数"], 3)
        self.assertEqual(plan["冻结融合评估数"], 6)
        self.assertEqual(plan["命令数"], 16)
        self.assertEqual(plan["固定协议"]["seeds"], [1, 2, 3])
        self.assertEqual(plan["固定协议"]["alpha"], 0.5)
        self.assertEqual(plan["固定协议"]["类别划分"], "按synset排序前500类为Base，后500类为Novel")
        self.assertIn("不搜索", plan["固定协议"]["选择限制"])


if __name__ == "__main__":
    unittest.main()
