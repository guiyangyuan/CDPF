import json
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts.repro import h1_study as study
from scripts.repro import h1_validation


def fake_prepared(root, name, dataset, mode, seed=1):
    directory = root / name; directory.mkdir()
    source = study.ROOT
    data = root / "data"; data.mkdir(exist_ok=True)
    config = directory / "future_config.yaml"; config.write_text("fixed: true\n")
    historical = directory / "historical.json"; historical.write_text("{}\n")
    npz = directory / "historical.npz"; npz.write_bytes(b"npz")
    receipt = {
        "status": "prepared", "schema_version": 1, "dataset_key": dataset,
        "dataset": h1_validation.probe.DATASETS[dataset][0], "seed": seed, "mode": mode,
        "protocol": h1_validation.PROTOCOL, "source_sha256": h1_validation.CURRENT_SOURCE_SHA256,
        "loss_identity": h1_validation.loss_identity(mode), "source_root": str(source),
        "data_root": str(data), "future_config_path": str(config),
        "future_config_sha256": "a" * 64, "prepared_identity_sha256": name.ljust(64, "0")[:64],
        "historical": {
            "export": str(historical),
            "export_sha256": hashlib.sha256(historical.read_bytes()).hexdigest(),
            "npz": str(npz), "npz_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        },
        "tool": {"h1_validation_sha256": "d" * 64},
    }
    path = directory / "prepared.json"; path.write_text(json.dumps(receipt))
    return path, receipt


def four_receipts(root):
    specs = (("anchor", "dtd", "full_kl"), ("ucf", "ucf101", "weighted_nckd"),
             ("dtd", "dtd", "weighted_nckd"),
             ("aircraft", "fgvc_aircraft", "weighted_nckd"))
    values = [fake_prepared(root, *spec) for spec in specs]
    values[2][1]["historical"] = dict(values[0][1]["historical"])
    return [value[0] for value in values], {str(value[0].resolve()): value[1] for value in values}


class PlanTests(unittest.TestCase):
    def test_plan_freezes_exact_order_budget_commands_and_hashes_without_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); paths, records = four_receipts(root)
            plan_dir, output_root = root / "plan", root / "future-study"
            with mock.patch.object(study, "load_prepared", side_effect=lambda p: records[str(Path(p).resolve())]):
                plan = study.create_plan(paths, output_root, plan_dir, python=sys.executable)

            self.assertEqual(plan["state"], "planned_not_executed")
            self.assertEqual(plan["training_budget"], {"maximum": 4, "planned": 4})
            self.assertEqual([case["case_id"] for case in plan["cases"]],
                             ["anchor-dtd", "candidate-ucf101", "candidate-dtd",
                              "candidate-fgvc_aircraft"])
            self.assertEqual(sum(stage["kind"] == "train" for stage in plan["stages"]), 4)
            self.assertEqual(sum(stage["kind"] == "export" for stage in plan["stages"]), 4)
            self.assertEqual(plan["stages"][0]["case_id"], "anchor-dtd")
            self.assertEqual(plan["stages"][2]["kind"], "anchor_audit")
            self.assertEqual(plan["stages"][-1]["kind"], "aggregate")
            self.assertFalse(output_root.exists())
            self.assertTrue((plan_dir / "plan.json").is_file())
            self.assertEqual(plan["plan_identity_sha256"], study.plan_digest(plan))

    def test_plan_rejects_wrong_seed_mode_order_duplicates_and_preexisting_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); paths, records = four_receipts(root)
            mutations = []
            wrong_seed = dict(records[str(paths[1].resolve())]); wrong_seed["seed"] = 2
            mutations.append((paths, {**records, str(paths[1].resolve()): wrong_seed}, "seed1"))
            wrong_mode = dict(records[str(paths[0].resolve())]); wrong_mode["mode"] = "weighted_nckd"
            mutations.append((paths, {**records, str(paths[0].resolve()): wrong_mode}, "order|mode"))
            mutations.append(([paths[1], paths[0], *paths[2:]], records, "order"))
            mutations.append(([paths[0], paths[1], paths[1], paths[3]], records, "once|duplicate"))
            for index, (given, loaded, message) in enumerate(mutations):
                with self.subTest(index=index), mock.patch.object(
                        study, "load_prepared", side_effect=lambda p, values=loaded: values[str(Path(p).resolve())]):
                    with self.assertRaisesRegex(ValueError, message):
                        study.create_plan(given, root / f"out-{index}", root / f"plan-{index}")
            (root / "occupied").mkdir()
            with mock.patch.object(study, "load_prepared", side_effect=lambda p: records[str(Path(p).resolve())]):
                with self.assertRaises(FileExistsError):
                    study.create_plan(paths, root / "occupied", root / "unused-plan")

    def test_plan_rejects_outputs_overlapping_authenticated_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); paths, records = four_receipts(root)
            protected_source = Path(records[str(paths[0].resolve())]["source_root"])
            with mock.patch.object(study, "load_prepared", side_effect=lambda p: records[str(Path(p).resolve())]):
                with self.assertRaisesRegex(ValueError, "disjoint"):
                    study.create_plan(paths, protected_source / "study", root / "plan")

    def test_plan_rejects_a_different_source_checkout_even_with_claimed_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); paths, records = four_receipts(root)
            other = root / "other-source"; other.mkdir()
            (other / "train.py").write_text("print('unbound source')\n")
            for record in records.values():
                record["source_root"] = str(other)
            with mock.patch.object(study, "load_prepared",
                                   side_effect=lambda p: records[str(Path(p).resolve())]):
                with self.assertRaisesRegex(ValueError, "current source"):
                    study.create_plan(paths, root / "study", root / "plan")

    def test_saved_plan_revalidates_then_rejects_changed_inputs_or_skipped_anchor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); paths, records = four_receipts(root)
            loader = lambda p: records[str(Path(p).resolve())]
            with mock.patch.object(study, "load_prepared", side_effect=loader):
                plan = study.create_plan(paths, root / "study", root / "plan")
                self.assertEqual(study._validate_plan_for_run(root / "plan/plan.json"), plan)

                historical = Path(records[str(paths[1].resolve())]["historical"]["export"])
                historical.write_text("changed historical bytes\n")
                with self.assertRaisesRegex(ValueError, "changed"):
                    study._validate_plan_for_run(root / "plan/plan.json")
                historical.write_text("{}\n")

                Path(paths[1]).write_text(Path(paths[1]).read_text() + "\n")
                with self.assertRaisesRegex(ValueError, "changed"):
                    study._validate_plan_for_run(root / "plan/plan.json")

            Path(paths[1]).write_text(json.dumps(records[str(paths[1].resolve())]))
            tampered = dict(plan); tampered["stages"] = plan["stages"][3:]
            tampered["plan_identity_sha256"] = study.plan_digest(tampered)
            (root / "plan/plan.json").write_text(json.dumps(tampered))
            with mock.patch.object(study, "load_prepared", side_effect=loader):
                with self.assertRaisesRegex(ValueError, "commands|paths|inputs"):
                    study._validate_plan_for_run(root / "plan/plan.json")


class ExecutorTests(unittest.TestCase):
    def fake_plan(self, root, commands):
        stages = []
        for index, (case_id, kind, command) in enumerate(commands):
            stages.append({"stage_id": f"{index:02d}-{kind}", "case_id": case_id,
                           "kind": kind, "argv": command,
                           "output_dir": str(root / "study" / f"stage-{index}")})
        return {"state": "planned_not_executed", "schema_version": 1,
                "output_root": str(root / "study"), "stages": stages}

    def command(self, marker, name, output, exit_code=0):
        code = ("from pathlib import Path; "
                f"Path({str(marker)!r}).open('a').write({(name + os.linesep)!r}); "
                f"Path({str(output)!r}).mkdir(parents=True); "
                f"Path({str(output / 'artifact.json')!r}).write_text("
                f"{json.dumps({'stage': name})!r}); "
                f"raise SystemExit({exit_code})")
        return [sys.executable, "-c", code]

    def accept_stage(self, stage, *_):
        return {"output": study._directory_manifest(Path(stage["output_dir"]))}

    def test_executor_runs_anchor_before_candidates_and_writes_auditable_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); marker = root / "order.txt"
            specs = [("anchor-dtd", "train"), ("anchor-dtd", "export"),
                     ("anchor-dtd", "anchor_audit"),
                     ("candidate-ucf101", "train"), ("candidate-ucf101", "export"),
                     ("candidate-ucf101", "candidate_analysis"),
                     ("candidate-dtd", "train"), ("candidate-dtd", "export"),
                     ("candidate-dtd", "candidate_analysis"),
                     ("candidate-fgvc_aircraft", "train"),
                     ("candidate-fgvc_aircraft", "export"),
                     ("candidate-fgvc_aircraft", "candidate_analysis"),
                     ("all-candidates", "aggregate")]
            commands = []
            for index, (case_id, kind) in enumerate(specs):
                output = root / "study" / f"stage-{index}"
                commands.append((case_id, kind, self.command(marker, kind, output)))
            plan = self.fake_plan(root, commands); plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan))
            result = study.execute_plan(
                plan_path, cuda_available=lambda: True,
                plan_validator=lambda _: plan, stage_validator=self.accept_stage)
            self.assertEqual(result["state"], "passed")
            self.assertEqual(marker.read_text().splitlines(), [kind for _, kind in specs])
            self.assertEqual(len(result["commands"]), 13)
            self.assertTrue((root / "study" / "status.json").is_file())

    def test_executor_refuses_no_cuda_before_spawning_any_command(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); marker = root / "spawned"
            output = root / "study/stage-0"
            plan = self.fake_plan(root, [("anchor-dtd", "train", self.command(marker, "x", output))])
            path = root / "plan.json"; path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                study.execute_plan(path, cuda_available=lambda: False,
                                   plan_validator=lambda _: plan, stage_validator=lambda *_: None)
            self.assertFalse(marker.exists())
            self.assertEqual(json.loads((root / "study/status.json").read_text())["state"], "failed")

    def test_anchor_failure_stops_before_candidate_and_executor_never_reuses_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); marker = root / "order.txt"
            commands = []
            for index, (case_id, kind, exit_code) in enumerate((
                    ("anchor-dtd", "train", 0), ("anchor-dtd", "export", 0),
                    ("anchor-dtd", "anchor_audit", 7),
                    ("candidate-ucf101", "train", 0))):
                output = root / "study" / f"stage-{index}"
                commands.append((case_id, kind, self.command(marker, kind, output, exit_code)))
            plan = self.fake_plan(root, commands); path = root / "plan.json"
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(RuntimeError, "anchor_audit"):
                study.execute_plan(path, cuda_available=lambda: True,
                                   plan_validator=lambda _: plan, stage_validator=self.accept_stage)
            self.assertEqual(marker.read_text().splitlines(), ["train", "export", "anchor_audit"])
            status = json.loads((root / "study/status.json").read_text())
            self.assertEqual(status["state"], "stopped")
            with self.assertRaises(FileExistsError):
                study.execute_plan(path, cuda_available=lambda: True,
                                   plan_validator=lambda _: plan, stage_validator=self.accept_stage)

    def test_candidate_failure_is_fail_fast_and_preserves_command_log(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); marker = root / "order.txt"
            commands = []
            for index, (case_id, code) in enumerate((
                    ("anchor-dtd", 0), ("candidate-ucf101", 3), ("candidate-dtd", 0))):
                output = root / "study" / f"stage-{index}"
                commands.append((case_id, "train", self.command(marker, case_id, output, code)))
            plan = self.fake_plan(root, commands); path = root / "plan.json"
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(RuntimeError, "candidate-ucf101"):
                study.execute_plan(path, cuda_available=lambda: True,
                                   plan_validator=lambda _: plan, stage_validator=self.accept_stage)
            self.assertEqual(marker.read_text().splitlines(), ["anchor-dtd", "candidate-ucf101"])
            logs = sorted((root / "study/logs").glob("*.json"))
            self.assertEqual(len(logs), 2)
            self.assertEqual(json.loads(logs[-1].read_text())["returncode"], 3)

    def test_spawn_failure_is_recorded_as_a_command_log(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = self.fake_plan(root, [("anchor-dtd", "train", [str(root / "missing")])])
            path = root / "plan.json"; path.write_text(json.dumps(plan))
            with self.assertRaises(FileNotFoundError):
                study.execute_plan(path, cuda_available=lambda: True,
                                   plan_validator=lambda _: plan, stage_validator=lambda *_: None)
            logs = list((root / "study/logs").glob("*.json"))
            self.assertEqual(len(logs), 1)
            self.assertEqual(json.loads(logs[0].read_text())["status"], "failed")

    def test_later_stage_mutating_prior_stage_file_fails_dynamic_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); marker = root / "order.txt"
            model = root / "study/stage-0"
            export = root / "study/stage-1"
            train = self.command(marker, "train", model)
            mutate_code = (
                "from pathlib import Path; "
                f"Path({str(marker)!r}).open('a').write('export\\n'); "
                f"Path({str(model / 'artifact.json')!r}).write_text('mutated'); "
                f"Path({str(export)!r}).mkdir(parents=True); "
                f"Path({str(export / 'artifact.json')!r}).write_text('export')")
            plan = self.fake_plan(root, [
                ("anchor-dtd", "train", train),
                ("anchor-dtd", "export", [sys.executable, "-c", mutate_code]),
            ])
            path = root / "plan.json"; path.write_text(json.dumps(plan))

            with self.assertRaisesRegex(ValueError, "changed during|dynamic"):
                study.execute_plan(path, cuda_available=lambda: True,
                                   plan_validator=lambda _: plan,
                                   stage_validator=self.accept_stage)

            log = json.loads((root / "study/logs/01-export.json").read_text())
            self.assertFalse(log["inputs_immutable"])


class StageEvidenceTests(unittest.TestCase):
    def test_export_rejects_same_dataset_mode_from_different_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "study/model"; model.mkdir(parents=True)
            (model / "checkpoint.bin").write_bytes(b"checkpoint")
            export = root / "study/export"; export.mkdir()
            result = export / "export.json"; result.write_text("{}")
            prepared = root / "prepared.json"; prepared.write_text("{}")
            train_stage = {"stage_id": "01-train", "case_id": "candidate-dtd",
                           "kind": "train", "output_dir": str(model)}
            export_stage = {"stage_id": "02-export", "case_id": "candidate-dtd",
                            "kind": "export", "output_dir": str(export),
                            "result_path": str(result)}
            case = {"case_id": "candidate-dtd", "dataset_key": "dtd", "seed": 1,
                    "mode": "weighted_nckd", "prepared": str(prepared.resolve()),
                    "prepared_sha256": study._sha256(prepared),
                    "prepared_identity_sha256": "p" * 64,
                    "source_sha256": "s" * 64}
            plan = {"cases": [case], "stages": [train_stage, export_stage]}
            checkpoint = {"model_dir": str(model.resolve()), "checkpoint_sha256": "x" * 64}
            receipt = {**case, "prepared": str(prepared.resolve()), "checkpoint": checkpoint}
            completed = {"01-train": {
                "checkpoint": {**checkpoint, "checkpoint_sha256": "y" * 64},
                "output": study._directory_manifest(model),
            }}

            with mock.patch.object(study, "load_export", return_value=(receipt, {})):
                with self.assertRaisesRegex(ValueError, "checkpoint|model"):
                    study._validate_stage_output(export_stage, plan, completed)

    def test_minimal_passed_anchor_cannot_release_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            export_dir = root / "export"; export_dir.mkdir()
            export_path = export_dir / "export.json"; export_path.write_text("{}")
            historical_path = root / "historical.json"; historical_path.write_text("{}")
            analysis_dir = root / "analysis"; analysis_dir.mkdir()
            result_path = analysis_dir / "analysis.json"
            result_path.write_text(json.dumps({
                "status": "passed", "analysis_kind": "anchor_audit", "dataset_key": "dtd",
            }))
            rows = [{"row": 0, "sample_id": "a", "label": 0, "classname": "zero"},
                    {"row": 1, "sample_id": "b", "label": 1, "classname": "one"}]
            receipt = {"dataset_key": "dtd", "seed": 1, "mode": "full_kl",
                       "rows": rows, "class_order": ["zero", "one"]}
            s = np.array([[2, 0], [0, 2]], dtype=np.float32)
            m = np.array([[1, 0], [0, 1]], dtype=np.float32)
            current = {"labels": np.array([0, 1], dtype=np.int64), "S": s, "T": s,
                       "M": m, "ensemble": s.copy()}
            historical = {"labels": current["labels"].copy(), "C": s, "S0": s,
                          "S1": s, "M": m, "T0": s, "T1": s}
            case = {"case_id": "anchor-dtd", "dataset_key": "dtd", "seed": 1,
                    "mode": "full_kl", "historical_export": str(historical_path.resolve())}
            export_stage = {"stage_id": "02-export", "case_id": "anchor-dtd",
                            "kind": "export", "output_dir": str(export_dir),
                            "result_path": str(export_path)}
            anchor_stage = {"stage_id": "03-anchor", "case_id": "anchor-dtd",
                            "kind": "anchor_audit", "output_dir": str(analysis_dir),
                            "result_path": str(result_path)}
            plan = {"cases": [case], "stages": [export_stage, anchor_stage]}

            with mock.patch.object(study, "load_export", return_value=(receipt, current)), \
                    mock.patch.object(study.h1_analysis.historical_analysis, "load_export",
                                      return_value=(receipt, historical)):
                with self.assertRaisesRegex(ValueError, "differs|forged|complete"):
                    study._validate_stage_output(anchor_stage, plan)


if __name__ == "__main__":
    unittest.main()
