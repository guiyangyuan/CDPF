import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/repro" / f"{name}.py")
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


class CrossDatasetPlanTests(unittest.TestCase):
    def test_plan_has_exact_stages_and_command_counts_without_output(self):
        study = module("cross_dataset_study")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "never-created"
            plan = study.build_plan(ROOT, "/data", output, "/python")
            self.assertFalse(output.exists())
            self.assertEqual([s["name"] for s in plan["stages"]],
                             ["smoke_dtd", "smoke_fgvc_aircraft", "formal_dtd", "formal_fgvc_aircraft"])
            self.assertEqual([len(s["commands"]) for s in plan["stages"]], [17, 17, 51, 51])
            self.assertEqual(sum("--eval-only" not in c for s in plan["stages"] for c in s["commands"]), 24)
            self.assertTrue(all(str(ROOT / "train.py") in c for s in plan["stages"] for c in s["commands"]))

    def test_execute_rejects_existing_root_before_cuda(self):
        study = module("cross_dataset_study")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); preparation = root / "prep"; preparation.mkdir()
            plan = {"output_root": str(root), "source_root": str(ROOT)}
            with self.assertRaises(FileExistsError): study.execute(plan, preparation)

    def test_execute_rejects_no_cuda_without_creating_root(self):
        study = module("cross_dataset_study")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "study"; prep = Path(temp) / "prep"; prep.mkdir()
            plan = study.build_plan(ROOT, "/data", output, "/python")
            clip = Path(temp) / "clip"; clip.write_bytes(b"fixture")
            datasets = {key: {} for key in study.DATASETS}
            prepared = {"protocol": plan, "files_sha256": plan["files_sha256"], "datasets": datasets,
                        "clip": {"path": str(clip), "sha256": study.sha256(clip)},
                        "operational_scripts_sha256": study.operational_hashes()}
            (prep / "preparation.json").write_text(json.dumps(prepared))
            with patch.object(study, "dataset_evidence", side_effect=lambda source, data, key: datasets[key]), \
                    patch("torch.cuda.is_available", return_value=False), self.assertRaisesRegex(RuntimeError, "CUDA"):
                study.execute(plan, prep)
            self.assertFalse(output.exists())

    def test_gpu_guard_rejects_active_low_memory_process_and_accepts_idle(self):
        study = module("cross_dataset_study")
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                               mem_get_info=lambda index: (99, 100))
        torch = SimpleNamespace(cuda=cuda)
        active = lambda *a, **k: SimpleNamespace(returncode=0, stdout="GPU-uuid, 123\n")
        with self.assertRaisesRegex(RuntimeError, "Active GPU"):
            study.assert_gpus_idle(torch, active)
        idle = lambda *a, **k: SimpleNamespace(returncode=0, stdout="")
        study.assert_gpus_idle(torch, idle)

    def test_external_source_and_operational_dependencies_are_frozen(self):
        audit = module("cross_dataset_audit")
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp); dependency = source / "scripts/repro/structure_spec.py"
            dependency.parent.mkdir(parents=True); dependency.write_text("fixed")
            here = ROOT / "scripts/repro"
            prepared = {"files_sha256": {"scripts/repro/structure_spec.py": audit.digest(dependency)},
                        "operational_scripts_sha256": {
                            "cross_dataset_study.py": audit.digest(here / "cross_dataset_study.py"),
                            "cross_dataset_audit.py": audit.digest(here / "cross_dataset_audit.py")}}
            audit.verify_frozen(source, prepared)
            dependency.write_text("changed")
            with self.assertRaisesRegex(ValueError, "dependency changed"):
                audit.verify_frozen(source, prepared)

    def test_stage_protocol_preflight_exist_before_first_raw_command_and_audit_logs_named(self):
        study = module("cross_dataset_study")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); stage_root = root / "smoke/dtd"
            stage = {"name": "smoke_dtd", "kind": "smoke", "dataset": "dtd", "seeds": [1],
                     "epochs": 1, "commands": [["python", "train.py"]]}
            plan = {"source_root": str(ROOT), "fixed_settings": study.FIXED,
                    "view_orders": {v: {"policy": "contiguous", "seed": 0} for v in study.VARIANTS},
                    "stages": [stage]}
            prepared = {"model_source_sha256": "model", "datasets": {"dtd": {"loader": "fixed"}}}
            status = {"state": "running", "commands": [], "completed": []}
            calls = []
            def fake_run(command, stdout, stderr, **kwargs):
                self.assertTrue((stage_root / "protocol.json").is_file())
                self.assertTrue((stage_root / "data_preflight.json").is_file())
                calls.append((str(stdout), str(stderr)))
                return SimpleNamespace(returncode=0)
            with patch.object(study, "run_logged", side_effect=fake_run):
                study.execute_stages(plan, prepared, root / "prep", root, status)
            self.assertTrue(any(path.endswith("smoke_dtd/audit.stdout.log") for path, _ in calls))
            self.assertTrue(any(path.endswith("final_audit/audit.stdout.log") for path, _ in calls))

    def test_launch_exception_records_failed_status(self):
        study = module("cross_dataset_study")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "study"; prep = Path(temp) / "prep"; prep.mkdir()
            plan = study.build_plan(ROOT, "/data", output, "/python")
            clip = Path(temp) / "clip"; clip.write_bytes(b"fixture")
            datasets = {key: {} for key in study.DATASETS}
            prepared = {"protocol": plan, "files_sha256": plan["files_sha256"], "datasets": datasets,
                        "clip": {"path": str(clip), "sha256": study.sha256(clip)},
                        "operational_scripts_sha256": study.operational_hashes()}
            (prep / "preparation.json").write_text(json.dumps(prepared))
            with patch.object(study, "dataset_evidence", side_effect=lambda source, data, key: datasets[key]), \
                    patch.object(study, "assert_gpus_idle"), \
                    patch.object(study, "execute_stages", side_effect=OSError("launch failed")), \
                    self.assertRaisesRegex(OSError, "launch failed"):
                study.execute(plan, prep)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["error"]["type"], "OSError")


class CrossDatasetAuditTests(unittest.TestCase):
    def test_final_rejects_incomplete_and_uses_all_rows(self):
        audit = module("cross_dataset_audit")
        with self.assertRaisesRegex(ValueError, "84 formal evaluations"):
            audit.final_analysis([], [])
        rows = []
        for dataset in audit.DATASET_NAMES:
            for variant, branches in audit.BRANCHES.items():
                for branch in branches:
                    for seed in (1, 2, 3):
                        rows.append({"dataset": dataset, "variant": variant, "branch": branch,
                                     "seed": seed, "base": 60 + seed, "novel": 50 + seed, "hm": 55 + seed})
        result = audit.final_analysis(rows, [{}] * 14)
        self.assertTrue(result["passed"]); self.assertEqual(len(result["seed_branch_rows"]), 42)
        self.assertEqual(len(result["comparisons"]), 4)

    def test_checkpoint_rejects_hash_tamper_and_parameter_count(self):
        import torch
        audit = module("cross_dataset_audit")
        manifest = {"dataset": "DescribableTextures", "seed": 1, "shots": 16, "code_sha256": "x",
                    "train_samples_sha256": "t", "effective_losses": {"kd": 0.0},
                    "model_signature": {"trainer": "x"}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pth.tar-1"
            torch.save({"epoch": 1, "repro_signature": manifest["model_signature"], "run_provenance": manifest,
                        "state_dict": {"ctx": torch.ones(1)}}, path)
            actual = audit.digest(path); path.with_name(path.name + ".sha256").write_text(actual + "\n")
            metric = {"checkpoints": [{"path": str(path.resolve()), "sha256": actual, "epoch": 1}]}
            with self.assertRaisesRegex(ValueError, "parameter count"):
                audit.checkpoint_audit(path, metric, manifest, "baseline", 1)
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "hash/path/epoch"):
                audit.checkpoint_audit(path, metric, manifest, "baseline", 1)

    def test_log_count_and_loader_derived_counts(self):
        audit = module("cross_dataset_audit")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "log"; path.write_text("* total: 3,344\n* correct: 1,672\n")
            self.assertEqual(audit.parse_log(path), (1672, 3344))
        prepared = {"datasets": {"fgvc_aircraft": {"loader_evidence": [
            {"seed": 1, "splits": {"new": {"test_count": 3333}}}]}}}
        self.assertEqual(audit.expected_loader(prepared, "fgvc_aircraft", 1, "new")["test_count"], 3333)

    def test_metric_rejects_test_hash_and_split_tampering(self):
        audit = module("cross_dataset_audit")
        expected = {"train_sha256": "train", "val_sha256": "val", "test_sha256": "test"}
        valid = {"split": "test", "train_samples_sha256": "train",
                 "val_samples_sha256": "val", "test_samples_sha256": "test"}
        audit.validate_metric_samples(valid, expected)
        for field, value in (("split", "val"), ("test_samples_sha256", "changed"),
                             ("train_samples_sha256", "changed")):
            changed = dict(valid); changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                audit.validate_metric_samples(changed, expected)

    def test_config_family_keeps_optimizer_drift_visible(self):
        audit = module("cross_dataset_audit")
        base = {"OPTIM": {"LR": .002}, "TRAINER": {"NAME": "baseline", "TEXTCONSISTENCY": {
            "AGGREGATION": "mean", "KD_WEIGHT": 0.0, "USE_FEATURE_ADAPTER": False,
            "ABLATION_GROUP": "baseline", "N_CTX": 4}}}
        full = json.loads(json.dumps(base)); full["TRAINER"]["NAME"] = "full"
        full["TRAINER"]["TEXTCONSISTENCY"].update(AGGREGATION="hierarchical", KD_WEIGHT=1.0,
                                                   USE_FEATURE_ADAPTER=True, ABLATION_GROUP="full")
        self.assertEqual(audit.family_config(base), audit.family_config(full))
        full["OPTIM"]["LR"] = .02
        self.assertNotEqual(audit.family_config(base), audit.family_config(full))

    def test_read_only_audit_rejects_order_protocol_mutation(self):
        audit = module("cross_dataset_audit")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); target = root / "smoke/dtd"; target.mkdir(parents=True)
            stage = {"name": "smoke_dtd", "seeds": [1], "epochs": 1}
            prepared = {"model_source_sha256": "model", "protocol": {"stages": [stage],
                        "fixed_settings": {}, "view_orders": {v: {"policy": "contiguous", "seed": 0}
                                                               for v in audit.VARIANTS}},
                        "datasets": {"dtd": {}}}
            scoped = {"dataset": "DescribableTextures", "seeds": [1], "epochs": 1,
                      "variants": list(audit.VARIANTS), "shots": 16, "kd_weight": 1.0,
                      "code_sha256": "model", "fixed_settings": {},
                      "view_orders": json.loads(json.dumps(prepared["protocol"]["view_orders"]))}
            scoped["view_orders"]["hierarchical_adapter"] = {"policy": "shuffle", "seed": 9}
            (target / "protocol.json").write_text(json.dumps(scoped)); (target / "data_preflight.json").write_text("{}")
            with patch.object(audit, "load"), self.assertRaisesRegex(ValueError, "protocol identity"):
                audit.audit_dataset(ROOT, root, prepared, "smoke", "dtd")


if __name__ == "__main__": unittest.main()
