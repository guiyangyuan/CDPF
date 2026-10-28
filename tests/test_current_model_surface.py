import ast
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def model_source_digest() -> str:
    files = [ROOT / "train.py"]
    for folder in ("trainers", "datasets", "clip", "Dassl.pytorch/dassl"):
        files.extend(sorted((ROOT / folder).rglob("*.py")))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class CurrentModelSurface(unittest.TestCase):
    def test_only_current_and_baseline_trainers_are_registered(self):
        modules = imported_modules(ROOT / "train.py")
        legacy_stem = "textconsistency_" + "v2_fg"
        forbidden = {
            "trainers.coop_" + legacy_stem,
            "trainers.cocoop_" + legacy_stem,
        }
        self.assertTrue(forbidden.isdisjoint(modules))
        self.assertTrue({
            "trainers.coop",
            "trainers.coop_textconsistency_ablation",
            "trainers.cocoop",
            "trainers.zsclip",
        }.issubset(modules))

    def test_current_trainer_and_evaluator_are_self_contained(self):
        legacy_stem = "textconsistency_" + "v2_fg"
        for prefix in ("coop_", "cocoop_"):
            self.assertFalse((ROOT / "trainers" / f"{prefix}{legacy_stem}.py").exists())

        trainer = ROOT / "trainers/coop_textconsistency_ablation.py"
        modules = imported_modules(trainer)
        self.assertNotIn("trainers.coop_" + legacy_stem, modules)
        names = {
            node.name
            for node in ast.walk(ast.parse(trainer.read_text(encoding="utf-8")))
            if isinstance(node, ast.ClassDef)
        }
        self.assertTrue({"PromptLearner", "TextEncoder", "CoOp_TextConsistency_Ablation"}.issubset(names))

        evaluator = ROOT / "scripts/repro/evaluate_frozen_fusion.py"
        compile(evaluator.read_text(encoding="utf-8"), str(evaluator), "exec")

    def test_training_entrypoint_imports_only_current_method_and_standard_baselines(self):
        modules = {name for name in imported_modules(ROOT / "train.py") if name.startswith("trainers.")}
        self.assertEqual(modules, {
            "trainers.coop",
            "trainers.coop_textconsistency_ablation",
            "trainers.cocoop",
            "trainers.zsclip",
        })

    def test_runner_only_builds_the_current_coop_baseline(self):
        path = ROOT / "scripts/repro/run_pilot.py"
        spec = importlib.util.spec_from_file_location("current_runner", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            commands = module.build_commands(
                Path(temporary) / "data",
                Path(temporary) / "output",
                [1],
                20,
                ["baseline"],
                device="cuda",
                python="python",
                dataset="dtd",
                kd_weight=0.0,
            )
            with self.assertRaisesRegex(ValueError, "Only the current CoOp baseline"):
                module.build_commands(
                    Path(temporary) / "data",
                    Path(temporary) / "output",
                    [1],
                    20,
                    ["hierarchical"],
                    device="cuda",
                    python="python",
                    dataset="dtd",
                    kd_weight=1.0,
                )
        self.assertEqual(len(commands), 3)
        self.assertTrue(all("CoOp_TextConsistency_Ablation" in command for command in commands))

    def test_frozen_fusion_evaluator_reports_only_current_branches(self):
        source = (ROOT / "scripts/repro/evaluate_frozen_fusion.py").read_text(encoding="utf-8")
        for branch in ("student", "description_mean", "simple_fusion", "mean_probability_fusion"):
            self.assertIn(f'"{branch}"', source)
        self.assertNotIn("description_instance", source)
        self.assertNotIn("adk_probability_fusion", source)

    def test_active_reproduction_scripts_pin_the_clean_source_tree(self):
        digest = model_source_digest()
        active_scripts = (
            "scripts/repro/evaluate_frozen_fusion.py",
            "scripts/repro/执行扩展验证.py",
            "scripts/repro/核验扩展验证结果.py",
            "scripts/repro/运行三数据集主方法.py",
        )
        for relative in active_scripts:
            self.assertIn(digest, (ROOT / relative).read_text(encoding="utf-8"), relative)


if __name__ == "__main__":
    unittest.main()
