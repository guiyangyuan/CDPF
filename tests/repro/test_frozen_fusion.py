import argparse
import builtins
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import torch
import yaml

from scripts.repro import evaluate_frozen_fusion as fusion


class FrozenFusionMath(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_fixed_logits_attention_and_probability_formulas(self):
        student = torch.tensor([[2.0, -1.0]])
        image = torch.tensor([[2.0, 0.0]])
        descriptions = torch.tensor([
            [[3.0, 0.0], [0.0, 4.0]],
            [[-5.0, 0.0], [20.0, 20.0]],
        ])
        mask = torch.tensor([[True, True], [True, False]])
        result = fusion.frozen_fusion_logits(student, image, descriptions, mask, 2.0)

        expected_mean_features = torch.tensor([[
            2 ** -0.5, 2 ** -0.5
        ], [-1.0, 0.0]])
        expected_mean = 2.0 * torch.tensor([[1.0, 0.0]]) @ expected_mean_features.t()
        torch.testing.assert_close(result["description_mean_features"], expected_mean_features)
        torch.testing.assert_close(result["description_mean"], expected_mean)
        torch.testing.assert_close(result["simple_fusion"], (student + expected_mean) / 2)

        expected_attention = torch.softmax(torch.tensor([2.0, 0.0]), dim=0)
        torch.testing.assert_close(result["description_attention"][0, 0], expected_attention)
        self.assertEqual(result["description_attention"][0, 1, 1].item(), 0.0)
        instance_text0 = torch.nn.functional.normalize(
            expected_attention @ torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=0)
        expected_instance = torch.tensor([[2.0 * instance_text0[0], -2.0]])
        torch.testing.assert_close(result["description_instance"], expected_instance)

        student_p = torch.softmax(student, dim=-1)
        mean_p = torch.softmax(expected_mean, dim=-1)
        instance_p = torch.softmax(expected_instance, dim=-1)
        torch.testing.assert_close(result["mean_probability_fusion"], 0.5 * student_p + 0.5 * mean_p)
        torch.testing.assert_close(
            result["adk_probability_fusion"], 0.5 * student_p + 0.25 * mean_p + 0.25 * instance_p)
        for name in ("mean_probability_fusion", "adk_probability_fusion"):
            torch.testing.assert_close(result[name].sum(dim=-1), torch.ones(1))

    def test_masked_values_cannot_affect_any_description_branch(self):
        torch.manual_seed(4)
        student = torch.randn(3, 2)
        image = torch.randn(3, 4)
        descriptions = torch.randn(2, 3, 4)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        changed = descriptions.clone()
        changed[~mask] = 1e8
        left = fusion.frozen_fusion_logits(student, image, descriptions, mask, 3.0)
        right = fusion.frozen_fusion_logits(student, image, changed, mask, 3.0)
        for name in ("description_mean", "simple_fusion", "description_instance",
                     "mean_probability_fusion", "adk_probability_fusion"):
            torch.testing.assert_close(left[name], right[name])

    def test_nonfinite_and_empty_descriptions_are_rejected(self):
        student, image = torch.ones(1, 1), torch.ones(1, 2)
        descriptions = torch.ones(1, 2, 2)
        with self.assertRaisesRegex(ValueError, "finite"):
            bad = descriptions.clone(); bad[0, 0, 0] = float("nan")
            fusion.frozen_fusion_logits(student, image, bad, torch.ones(1, 2, dtype=torch.bool), 1.0)
        with self.assertRaisesRegex(ValueError, "at least one"):
            fusion.frozen_fusion_logits(student, image, descriptions, torch.zeros(1, 2, dtype=torch.bool), 1.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            fusion.frozen_fusion_logits(student, image, descriptions, torch.ones(1, 2, dtype=torch.bool), 0.0)

    def test_description_terms_have_zero_gradient_to_coop_student_context(self):
        student = torch.randn(2, 3, requires_grad=True)
        descriptions = torch.randn(3, 2, 4, requires_grad=True)
        result = fusion.frozen_fusion_logits(
            student, torch.randn(2, 4), descriptions, torch.ones(3, 2, dtype=torch.bool), 2.0)
        gradient = torch.autograd.grad(
            result["description_mean"].sum() + result["description_instance"].sum(),
            student, allow_unused=True,
        )[0]
        self.assertIsNone(gradient)
        self.assertIsNotNone(torch.autograd.grad(result["description_instance"].sum(), descriptions)[0])


def _signature():
    return {**fusion.EXPECTED_SIGNATURE, "clip_sha256": "c" * 64}


def _manifest(dataset, seed, split):
    return {
        "seed": seed,
        "dataset": fusion.DATASETS[dataset][0],
        "shots": 16,
        "subsample": split,
        "model_signature": ({**fusion._expected_signature(dataset), "clip_sha256": "c" * 64}),
        "code_sha256": fusion.DATASETS[dataset][1],
        "effective_losses": dict(fusion.EXPECTED_LOSSES),
        "train_samples_sha256": "1" * 64,
        "val_samples_sha256": "2" * 64,
        "test_samples_sha256": "3" * 64,
        "trainable_parameters": fusion.TRAINABLE_PARAMETERS,
    }


def _baseline_fixture(root, dataset="dtd", seed=1, split="base"):
    baseline = Path(root) / "baseline" / f"seed{seed}"
    eval_dir = baseline / f"eval_{split}_student"
    checkpoint_dir = baseline / "textconsistency"
    eval_dir.mkdir(parents=True)
    checkpoint_dir.mkdir()
    train = _manifest(dataset, seed, "base")
    evaluated = _manifest(dataset, seed, split)
    (baseline / "run_manifest.json").write_text(json.dumps(train))
    (eval_dir / "run_manifest.json").write_text(json.dumps(evaluated))
    checkpoint = checkpoint_dir / "model.pth.tar-20"
    checkpoint.write_bytes(b"fixed checkpoint bytes")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (checkpoint_dir / "model.pth.tar-20.sha256").write_text(digest + "\n")
    metric = {
        **evaluated,
        "split": "test",
        "branch": "student",
        "ensemble_alpha": 0.5,
        "state": "loaded_checkpoint",
        "checkpoints": [{"path": "/historical/model.pth.tar-20", "sha256": digest, "epoch": 20}],
        "metrics": {"accuracy": 50.0},
    }
    (eval_dir / "metrics.json").write_text(json.dumps(metric))
    (eval_dir / "log.txt").write_text("* total: 2\n* correct: 1\n")
    config = {"SEED": seed, "OUTPUT_DIR": "/old", "DATASET": {"ROOT": "/old-data"}}
    (baseline / "effective_config.yaml").write_text(yaml.safe_dump(config))
    (eval_dir / "effective_config.yaml").write_text(yaml.safe_dump(config))
    return baseline


class FrozenFusionMetadata(unittest.TestCase):
    def test_old_ucf_encoder_gets_explicit_contiguous_order_evidence(self):
        encoder = SimpleNamespace(description_counts=[2, 1])
        result = fusion.description_order_evidence(encoder, ["jump", "run"])
        self.assertEqual(result["policy"], "contiguous")
        self.assertEqual(result["classes"], [
            {"classname": "jump", "count": 2, "order": [0, 1]},
            {"classname": "run", "count": 1, "order": [0]},
        ])
        self.assertEqual(len(result["sha256"]), 64)

    def test_new_encoder_order_evidence_is_preserved_verbatim(self):
        expected = {"version": 1, "sha256": "x"}
        encoder = SimpleNamespace(description_counts=[1], view_order_manifest=expected)
        self.assertIs(fusion.description_order_evidence(encoder, ["one"]), expected)

    def test_valid_relocated_checkpoint_is_accepted_without_loading_torch_or_clip(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline = _baseline_fixture(temp)
            real_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name == "clip" or name.startswith("clip."):
                    raise AssertionError("metadata validation imported CLIP")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=guarded_import), \
                    mock.patch.object(torch, "load", side_effect=AssertionError("loaded checkpoint tensors")):
                result = fusion.validate_baseline_metadata(
                    baseline, "dtd", 1, "base", fusion.DATASETS["dtd"][1])
            self.assertEqual(result["historical_student"]["correct"], 1)

    def test_wrong_seed_or_sample_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline = _baseline_fixture(temp)
            with self.assertRaisesRegex(ValueError, "seed"):
                fusion.validate_baseline_metadata(baseline, "dtd", 2, "base", fusion.DATASETS["dtd"][1])
            path = baseline / "eval_base_student/metrics.json"
            value = json.loads(path.read_text()); value["train_samples_sha256"] = "9" * 64
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "manifest/metric"):
                fusion.validate_baseline_metadata(baseline, "dtd", 1, "base", fusion.DATASETS["dtd"][1])

    def test_new_split_uses_its_own_saved_loader_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline = _baseline_fixture(temp, split="new")
            for relative in ("eval_new_student/run_manifest.json", "eval_new_student/metrics.json"):
                path = baseline / relative
                value = json.loads(path.read_text()); value["train_samples_sha256"] = "8" * 64
                path.write_text(json.dumps(value))
            result = fusion.validate_baseline_metadata(
                baseline, "dtd", 1, "new", fusion.DATASETS["dtd"][1])
            self.assertEqual(result["metric"]["train_samples_sha256"], "8" * 64)

    def test_tampered_checkpoint_and_sidecar_are_rejected(self):
        for target in ("checkpoint", "sidecar"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                baseline = _baseline_fixture(temp)
                checkpoint = baseline / "textconsistency/model.pth.tar-20"
                if target == "checkpoint":
                    checkpoint.write_bytes(b"tampered")
                else:
                    checkpoint.with_name(checkpoint.name + ".sha256").write_text("0" * 64)
                with self.assertRaisesRegex(ValueError, "sidecar"):
                    fusion.validate_baseline_metadata(
                        baseline, "dtd", 1, "base", fusion.DATASETS["dtd"][1])

    def test_nonstudent_or_non_epoch20_history_is_rejected(self):
        for field, value, pattern in (("branch", "teacher", "student"), ("epoch", 19, "epoch")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                baseline = _baseline_fixture(temp)
                path = baseline / "eval_base_student/metrics.json"
                metric = json.loads(path.read_text())
                if field == "epoch":
                    metric["checkpoints"][0]["epoch"] = value
                else:
                    metric[field] = value
                path.write_text(json.dumps(metric))
                with self.assertRaisesRegex(ValueError, pattern):
                    fusion.validate_baseline_metadata(
                        baseline, "dtd", 1, "base", fusion.DATASETS["dtd"][1])

    def test_wrong_dataset_source_release_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline = _baseline_fixture(temp)
            with self.assertRaisesRegex(ValueError, "requires frozen source"):
                fusion.validate_baseline_metadata(
                    baseline, "dtd", 1, "base", fusion.DATASETS["ucf101"][1])

    def test_source_derived_config_rejects_non_path_drift_but_allows_relocation(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline = _baseline_fixture(temp)
            expected = {"SEED": 1, "OUTPUT_DIR": "/new", "DATASET": {"ROOT": "/new-data"}}
            fusion.validate_baseline_metadata(
                baseline, "dtd", 1, "base", fusion.DATASETS["dtd"][1],
                expected_train_config=expected, expected_eval_config=expected)
            expected["SEED"] = 2
            with self.assertRaisesRegex(ValueError, "config"):
                fusion.validate_baseline_metadata(
                    baseline, "dtd", 1, "base", fusion.DATASETS["dtd"][1],
                    expected_train_config=expected)

    def test_ucf_legacy_declared_kd_is_narrowly_accepted_but_dtd_is_not(self):
        actual = {"TRAINER": {"TEXTCONSISTENCY": {"KD_WEIGHT": 1.0}}}
        expected = {"TRAINER": {"TEXTCONSISTENCY": {"KD_WEIGHT": 0.0}}}
        fact = fusion._validate_source_config(actual, expected, "ucf101", "training")
        self.assertEqual(fact["historical_declared_kd_weight"], 1.0)
        self.assertEqual(fact["effective_kd_weight"], 0.0)
        with self.assertRaisesRegex(ValueError, "source-derived"):
            fusion._validate_source_config(actual, expected, "dtd", "training")

    def test_nonzero_effective_kd_is_rejected_for_every_baseline(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline = _baseline_fixture(temp, dataset="ucf101")
            for relative in ("run_manifest.json", "eval_base_student/run_manifest.json",
                             "eval_base_student/metrics.json"):
                path = baseline / relative
                value = json.loads(path.read_text())
                value["effective_losses"]["kd"] = 1.0
                path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "effective_losses"):
                fusion.validate_baseline_metadata(
                    baseline, "ucf101", 1, "base", fusion.DATASETS["ucf101"][1])

    def test_checkpoint_tensor_audit_requires_finite_2048_parameter_student(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.pth.tar-20"
            manifest = _manifest("dtd", 1, "base")
            state = {
                "epoch": 20,
                "repro_signature": manifest["model_signature"],
                "run_provenance": dict(manifest),
                "state_dict": {"prompt_learner.ctx": torch.ones(4, 512)},
            }
            torch.save(state, path)
            fusion._validate_checkpoint_tensors(path, manifest)
            state["state_dict"]["prompt_learner.ctx"][0, 0] = float("nan")
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                fusion._validate_checkpoint_tensors(path, manifest)
            state["state_dict"] = {"prompt_learner.ctx": torch.ones(3, 512)}
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "2048"):
                fusion._validate_checkpoint_tensors(path, manifest)

    def test_historical_log_and_metric_must_agree(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline = _baseline_fixture(temp)
            (baseline / "eval_base_student/log.txt").write_text("* total: 3\n* correct: 1\n")
            with self.assertRaisesRegex(ValueError, "metric differs"):
                fusion.validate_baseline_metadata(
                    baseline, "dtd", 1, "base", fusion.DATASETS["dtd"][1])


class FrozenFusionCLI(unittest.TestCase):
    def test_existing_or_nested_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            existing = root / "existing"; existing.mkdir()
            with self.assertRaises(FileExistsError):
                fusion._ensure_disjoint_fresh_output(existing, ())
            with self.assertRaisesRegex(ValueError, "immutable input"):
                fusion._ensure_disjoint_fresh_output(root / "data/results", (root / "data",))

    def test_no_cuda_fails_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"; baseline = root / "baseline"; data = root / "data"
            source.mkdir(); baseline.mkdir(); data.mkdir()
            output = root / "output"
            args = argparse.Namespace(
                source_root=source, baseline_dir=baseline, dataset="dtd", seed=1, split="base",
                data_root=data, output_dir=output, expected_source_sha256=fusion.DATASETS["dtd"][1],
                validate_only=False,
            )
            cfgs = (object(), object(), {}, {}, object())
            with mock.patch.object(fusion, "model_source_digest", return_value=fusion.DATASETS["dtd"][1]), \
                    mock.patch.object(fusion, "build_source_configs", return_value=cfgs), \
                    mock.patch.object(fusion, "validate_baseline_metadata", return_value={}), \
                    mock.patch.object(torch.cuda, "is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "CUDA"):
                    fusion.evaluate(args)
            self.assertFalse(output.exists())

    def test_validate_only_returns_without_cuda_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"; baseline = root / "baseline"; data = root / "data"
            source.mkdir(); baseline.mkdir(); data.mkdir()
            output = root / "output"
            args = argparse.Namespace(
                source_root=source, baseline_dir=baseline, dataset="dtd", seed=1, split="base",
                data_root=data, output_dir=output, expected_source_sha256=fusion.DATASETS["dtd"][1],
                validate_only=True,
            )
            metadata = {"checkpoint_sha256": "a" * 64,
                        "historical_student": {"correct": 1, "total": 2, "accuracy": 50.0}}
            cfgs = (object(), object(), {}, {}, object())
            with mock.patch.object(fusion, "model_source_digest", return_value=fusion.DATASETS["dtd"][1]), \
                    mock.patch.object(fusion, "build_source_configs", return_value=cfgs), \
                    mock.patch.object(fusion, "validate_baseline_metadata", return_value=metadata), \
                    mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA queried")):
                result = fusion.evaluate(args)
            self.assertEqual(result["status"], "validated_metadata")
            self.assertFalse(output.exists())

    def test_cli_exposes_no_epoch_or_alpha_override(self):
        options = {option for action in fusion.build_parser()._actions for option in action.option_strings}
        self.assertNotIn("--epoch", options)
        self.assertNotIn("--alpha", options)
        self.assertEqual(fusion.EPOCH, 20)
        self.assertEqual(fusion.ALPHA, 0.5)


if __name__ == "__main__":
    unittest.main()
