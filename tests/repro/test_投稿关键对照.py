import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def 加载(filename: str, name: str):
    path = ROOT / "scripts/repro" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class 投稿关键对照测试(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "计划模块依赖 GPU 环境中的 torch")
    def test_计划只复用历史检查点并完整覆盖(self):
        module = 加载("运行投稿关键对照.py", "运行关键对照")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = module.生成计划(
                root / "release", root / "server", root / "data", root / "output", Path("python"))
        self.assertEqual(plan["训练次数"], 0)
        self.assertEqual(plan["推理命令数"], 60)
        self.assertEqual(plan["全局alpha网格"], [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertEqual(len({(row["数据集键"], row["种子"], row["分支"])
                              for row in plan["命令"]}), 60)
        self.assertTrue(all("--eval-only" not in row["命令"] for row in plan["命令"]))
        self.assertTrue(all("--baseline-dir" in row["命令"] for row in plan["命令"]))

    def test_汇总验证端点并报告描述语义增益(self):
        summary = 加载("汇总类别名称与权重稳健性.py", "汇总关键对照")
        ALPHA_GRID = summary.ALPHA_GRID
        alpha_branch_name = summary.alpha_branch_name
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            for dataset in summary.数据集顺序:
                for seed in (1, 2, 3):
                    for split in ("base", "new"):
                        total = 100
                        student = 70 + seed
                        classname = 68
                        class_fusion = 72 + seed
                        description = 74
                        main = 76 + seed
                        metrics = {
                            "student": {"correct": student, "total": total, "accuracy": float(student)},
                            "classname": {"correct": classname, "total": total, "accuracy": float(classname)},
                            "classname_probability_fusion": {"correct": class_fusion, "total": total,
                                                               "accuracy": float(class_fusion)},
                            "description_mean": {"correct": description, "total": total,
                                                   "accuracy": float(description)},
                            "mean_probability_fusion": {"correct": main, "total": total,
                                                          "accuracy": float(main)},
                        }
                        for alpha in ALPHA_GRID:
                            branch = alpha_branch_name("description_probability_alpha", alpha)
                            correct = student if alpha == 0 else description if alpha == 1 else main
                            metrics[branch] = {"correct": correct, "total": total, "accuracy": float(correct)}
                        internal = summary.指标数据集名称[dataset]
                        path = root / internal / f"seed{seed}" / split / "metrics.json"
                        path.parent.mkdir(parents=True)
                        path.write_text(json.dumps({
                            "status": "passed", "dataset": internal, "seed": seed,
                            "subsample": split, "split": "test", "metrics": metrics,
                        }), encoding="utf-8")
            output = Path(temporary) / "output"
            result = summary.生成结果(root, output)
            self.assertEqual(result["状态"], "通过")
            self.assertGreater(result["核心比较"]["TextConsistency减CoOp+类别名称_HM_pp"], 0)
            self.assertEqual(result["核心比较"]["TextConsistency胜出数据集数"], 10)
            self.assertEqual(result["核心比较"]["TextConsistency胜出配对种子数"], 30)
            self.assertTrue((output / "关键对照逐种子.csv").is_file())
            self.assertTrue((output / "统一权重稳健性.csv").is_file())

    def test_阶段汇总明确使用实际数据集和配对总数(self):
        summary = 加载("汇总类别名称与权重稳健性.py", "汇总关键对照子集")
        ALPHA_GRID = summary.ALPHA_GRID
        alpha_branch_name = summary.alpha_branch_name
        datasets = ["UCF101", "EuroSAT"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            for dataset in datasets:
                for seed in (1, 2, 3):
                    for split in ("base", "new"):
                        metrics = {
                            "student": {"correct": 70, "total": 100, "accuracy": 70.0},
                            "classname": {"correct": 68, "total": 100, "accuracy": 68.0},
                            "classname_probability_fusion": {"correct": 72, "total": 100, "accuracy": 72.0},
                            "description_mean": {"correct": 74, "total": 100, "accuracy": 74.0},
                            "mean_probability_fusion": {"correct": 76, "total": 100, "accuracy": 76.0},
                        }
                        for alpha in ALPHA_GRID:
                            correct = 70 if alpha == 0 else 74 if alpha == 1 else 76
                            metrics[alpha_branch_name("description_probability_alpha", alpha)] = {
                                "correct": correct, "total": 100, "accuracy": float(correct)}
                        path = root / dataset / f"seed{seed}" / split / "metrics.json"
                        path.parent.mkdir(parents=True)
                        path.write_text(json.dumps({
                            "status": "passed", "dataset": dataset, "seed": seed,
                            "subsample": split, "split": "test", "metrics": metrics,
                        }), encoding="utf-8")
            output = Path(temporary) / "output"
            result = summary.生成结果(root, output, datasets)
            self.assertEqual(result["数据集"], datasets)
            self.assertEqual(result["核心比较"]["配对种子总数"], 6)
            report = (output / "结果报告.md").read_text(encoding="utf-8")
            self.assertIn("胜出数据集：2/2", report)
            self.assertIn("胜出配对种子：6/6", report)

    def test_DTD汇总把评估器内部名称映射为论文名称(self):
        summary = 加载("汇总类别名称与权重稳健性.py", "汇总DTD别名")
        ALPHA_GRID = summary.ALPHA_GRID
        alpha_branch_name = summary.alpha_branch_name
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            for seed in (1, 2, 3):
                for split in ("base", "new"):
                    metrics = {
                        "student": {"correct": 70, "total": 100, "accuracy": 70.0},
                        "classname": {"correct": 68, "total": 100, "accuracy": 68.0},
                        "classname_probability_fusion": {"correct": 72, "total": 100, "accuracy": 72.0},
                        "description_mean": {"correct": 74, "total": 100, "accuracy": 74.0},
                        "mean_probability_fusion": {"correct": 76, "total": 100, "accuracy": 76.0},
                    }
                    for alpha in ALPHA_GRID:
                        correct = 70 if alpha == 0 else 74 if alpha == 1 else 76
                        metrics[alpha_branch_name("description_probability_alpha", alpha)] = {
                            "correct": correct, "total": 100, "accuracy": float(correct)}
                    path = root / "DescribableTextures" / f"seed{seed}" / split / "metrics.json"
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps({
                        "status": "passed", "dataset": "DescribableTextures", "seed": seed,
                        "subsample": split, "split": "test", "metrics": metrics,
                    }), encoding="utf-8")
            result = summary.生成结果(root, Path(temporary) / "output", ["DTD"])
            self.assertEqual(result["数据集"], ["DTD"])
            self.assertEqual(result["核心比较"]["配对种子总数"], 3)


if __name__ == "__main__":
    unittest.main()
