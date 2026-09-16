import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def 加载模块(filename: str, name: str):
    path = ROOT / "scripts/repro" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class 三数据集扩展测试(unittest.TestCase):
    def test_主方法计划固定九次训练且不按结果筛选(self):
        module = 加载模块("运行三数据集主方法.py", "三数据集主方法")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = module.生成计划(root / "data", root / "results", Path("python"))
        self.assertEqual(plan["数据集"], ["Caltech101", "OxfordPets", "Food101"])
        self.assertEqual(plan["训练数"], 9)
        self.assertEqual(plan["内部命令数"], 48)
        self.assertEqual(plan["固定协议"]["主方法"],
                         "0.5*softmax(CoOp logits)+0.5*softmax(冻结均值描述 logits)")
        self.assertIn("全部执行", plan["固定协议"]["选择限制"])
        self.assertEqual([run["训练数"] for run in plan["运行"]], [3, 3, 3])
        self.assertEqual([run["内部命令数"] for run in plan["运行"]], [16, 16, 16])

    def test_联合上下文基线为三数据集三种子(self):
        module = 加载模块("运行三数据集联合上下文基线.py", "三数据集联合上下文基线")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = module.生成计划(ROOT, root / "data", root / "results", Path("python"))
        self.assertEqual(plan["协议"], "投稿CoCoOp新增三数据集标准基线-v1")
        self.assertEqual(plan["运行数"], 9)
        self.assertEqual(plan["命令数"], 27)
        self.assertEqual({run["数据集"] for run in plan["运行"]},
                         {"Caltech101", "OxfordPets", "Food101"})
        self.assertTrue(all(run["阶段"] == "新增三数据集" for run in plan["运行"]))

    def test_零样本基线覆盖六个独立分支(self):
        module = 加载模块("运行三数据集零样本基线.py", "三数据集零样本基线")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = module.生成计划(ROOT, root / "data", root / "results", Path("python"))
        self.assertEqual(plan["协议"], "投稿Zero-shot-CLIP新增三数据集-v1")
        self.assertEqual(plan["命令数"], 6)
        self.assertEqual({(row["数据集"], row["分支"]) for row in plan["命令"]}, {
            (dataset, split)
            for dataset in ("Caltech101", "OxfordPets", "Food101")
            for split in ("base", "new")
        })

    def test_三个入口均接入主方法配置与描述文件(self):
        run_pilot = (ROOT / "scripts/repro/run_pilot.py").read_text(encoding="utf-8")
        evaluator = (ROOT / "scripts/repro/evaluate_frozen_fusion.py").read_text(encoding="utf-8")
        preflight = (ROOT / "scripts/repro/preflight.py").read_text(encoding="utf-8")
        templates = (ROOT / "trainers/textconsistency_common.py").read_text(encoding="utf-8")
        for key, dataset, template in (
            ("caltech101", "Caltech101", "cal_prompts_full.json"),
            ("oxford_pets", "OxfordPets", "pets_prompts_full.json"),
            ("food101", "Food101", "food_prompts_full.json"),
        ):
            self.assertIn(f"'{key}'", run_pilot)
            self.assertIn(f'"{key}"', evaluator)
            self.assertIn(f"'{key}'", preflight)
            self.assertIn(f"'{dataset}': 'templates/{template}'", templates)

    def test_数据准备固定官方归档和划分数量(self):
        module = 加载模块("准备三数据集数据.py", "三数据集数据准备")
        self.assertEqual(set(module.数据集), {"Caltech101", "OxfordPets", "Food101"})
        self.assertEqual(module.数据集["Caltech101"]["划分数量"],
                         {"train": 4128, "val": 1649, "test": 2465})
        self.assertEqual(module.数据集["OxfordPets"]["划分数量"],
                         {"train": 2944, "val": 736, "test": 3669})
        self.assertEqual(module.数据集["Food101"]["划分数量"],
                         {"train": 50500, "val": 20200, "test": 30300})
        self.assertEqual(module.数据集["OxfordPets"]["划分SHA256"],
                         "d7f3ffe65b3919f95132e51422894ae18686b83c331fd8c0adcdc3dc2de6301e")

    def test_Caltech官方ZIP内层图像归档会继续展开(self):
        module = 加载模块("准备三数据集数据.py", "Caltech嵌套归档")
        with tempfile.TemporaryDirectory() as temporary:
            dataset_root = Path(temporary) / "caltech-101"
            nested = dataset_root / "caltech-101" / "101_ObjectCategories.tar.gz"
            nested.parent.mkdir(parents=True)
            payload = b"image"
            with tarfile.open(nested, "w:gz") as bundle:
                info = tarfile.TarInfo("101_ObjectCategories/airplanes/image_0001.jpg")
                info.size = len(payload)
                bundle.addfile(info, io.BytesIO(payload))
            module.补齐Caltech内容(dataset_root)
            self.assertEqual(
                (dataset_root / "101_ObjectCategories/airplanes/image_0001.jpg").read_bytes(), payload
            )


if __name__ == "__main__":
    unittest.main()
