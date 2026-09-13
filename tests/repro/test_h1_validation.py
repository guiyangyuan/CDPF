import copy
import contextlib
import io
import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.repro import h1_validation as h1
from scripts.repro import validation_probe as probe


ROOT = Path(__file__).resolve().parents[2]


class HistoricalReferenceContracts(unittest.TestCase):
    def test_fixed_reference_accepts_only_seed1_dataset_and_exact_export_hashes(self):
        snapshot = json.loads(h1.FIXED_REFERENCE_PATH.read_text())
        case = h1.validate_historical_reference(snapshot, "dtd", 1)
        self.assertEqual(case["historical_export_sha256"],
                         "a298b41a7d306bcf23a8ad12e8095ace8254d551cd8eadd843f2b7dbe246f2b1")

        wrong_seed = copy.deepcopy(snapshot)
        wrong_seed["seed"] = 2
        with self.assertRaisesRegex(ValueError, "seed"):
            h1.validate_historical_reference(wrong_seed, "dtd", 1)

        changed_sha = copy.deepcopy(snapshot)
        changed_sha["cases"][1]["historical_logits_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "reference"):
            h1.validate_historical_reference(changed_sha, "dtd", 1)

        with self.assertRaisesRegex(ValueError, "seed"):
            h1.validate_historical_reference(snapshot, "dtd", 2)

    def test_fixed_reference_file_hash_is_authenticated(self):
        with tempfile.TemporaryDirectory() as temp:
            changed = Path(temp) / "reference.json"
            changed.write_text(h1.FIXED_REFERENCE_PATH.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "reference snapshot"):
                h1.load_reference_snapshot(changed)


class PreparationContracts(unittest.TestCase):
    def test_cli_exposes_prepare_and_export(self):
        result = subprocess.run(
            [sys.executable, "scripts/repro/h1_validation.py", "--help"],
            cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("prepare", result.stdout)
        self.assertIn("export", result.stdout)

    def test_protocol_rejects_unknown_mode_seed_or_current_source(self):
        self.assertEqual(h1.loss_identity("weighted_nckd"), {
            "format": 1,
            "mode": "weighted_nckd",
            "temperature": 2.0,
            "effective_weight": 1.0,
            "active": True,
            "non_target_weighting": "teacher_probability_mass_outside_ground_truth",
        })
        with self.assertRaisesRegex(ValueError, "mode"):
            h1.loss_identity("renamed")
        prepared = {"status": "prepared", "schema_version": 1, "protocol": h1.PROTOCOL,
                    "seed": 1, "mode": "full_kl", "loss_identity": h1.loss_identity("full_kl"),
                    "source_sha256": h1.CURRENT_SOURCE_SHA256}
        wrong_seed = {**prepared, "seed": 2}
        with self.assertRaisesRegex(ValueError, "seed"):
            h1.validate_prepared_identity(wrong_seed)
        wrong_source = {**prepared, "source_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "source"):
            h1.validate_prepared_identity(wrong_source)

    def test_historical_config_rejects_arbitrary_yaml(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "effective_config.yaml"
            path.write_text("SEED: 1\n")
            with self.assertRaisesRegex(ValueError, "historical.*config"):
                h1.validate_historical_config(path, "dtd")

    def test_sample_binding_rejects_duplicates_and_reordering(self):
        rows = [
            {"sample_id": "d/a.png", "label": 0, "classname": "zero", "size": 1, "mtime_ns": 1},
            {"sample_id": "d/b.png", "label": 1, "classname": "one", "size": 1, "mtime_ns": 1},
        ]
        h1.validate_sample_binding(rows, copy.deepcopy(rows), "val")
        with self.assertRaisesRegex(ValueError, "order"):
            h1.validate_sample_binding(rows, list(reversed(rows)), "val")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            h1.validate_sample_binding(rows, [rows[0], rows[0]], "val")

    def test_image_content_receipt_decodes_and_binds_ordered_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, color in (("a.png", "red"), ("b.png", "blue")):
                Image.new("RGB", (2, 2), color=color).save(root / name)
            rows = [
                {"sample_id": "a.png", "label": 0, "classname": "zero"},
                {"sample_id": "b.png", "label": 1, "classname": "one"},
            ]
            receipt = h1.image_content_receipt(rows, root)
            self.assertEqual(receipt["count"], 2)
            self.assertEqual(receipt["decoded_modes"], ["RGB"])
            self.assertNotEqual(receipt["ordered_sha256"],
                                h1.image_content_receipt(list(reversed(rows)), root)["ordered_sha256"])
            Image.new("RGB", (2, 2), color="green").save(root / "a.png")
            with self.assertRaisesRegex(ValueError, "image content"):
                h1.validate_image_content(rows, root, receipt, "train")

    def test_train_command_uses_current_train_seed1_saved_config_and_fresh_model_dir(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"; source.mkdir(); (source / "train.py").write_text("pass\n")
            config = root / "prepared/future_config.yaml"; config.parent.mkdir(); config.write_text("SEED: 1\n")
            prepared = {"status": "prepared", "schema_version": 1, "protocol": h1.PROTOCOL,
                        "seed": 1, "mode": "weighted_nckd",
                        "loss_identity": h1.loss_identity("weighted_nckd"),
                        "source_sha256": h1.CURRENT_SOURCE_SHA256, "source_root": str(source),
                        "data_root": str(root / "data"), "future_config_path": str(config)}
            model_dir = root / "model"
            command = h1.build_train_command(prepared, model_dir, python="python-fixture")
            self.assertEqual(command, ["python-fixture", str((source / "train.py").resolve()),
                "--root", str((root / "data").resolve()), "--output-dir", str(model_dir.resolve()),
                "--seed", "1", "--config-file", str(config.resolve())])
            model_dir.mkdir()
            with self.assertRaisesRegex(FileExistsError, "fresh"):
                h1.build_train_command(prepared, model_dir)

    def test_existing_fewshot_cache_must_match_cache_free_train_and_val_order(self):
        from dassl.data.datasets import Datum
        from datasets.oxford_pets import OxfordPets

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, val = [], []
            for label, classname in enumerate(("zero", "one", "two", "three")):
                for split, target in (("train", train), ("val", val)):
                    path = root / f"{split}-{label}.png"
                    Image.new("RGB", (2, 2), color=(label * 50, 0, 0)).save(path)
                    target.append(Datum(impath=str(path), label=label, classname=classname))
            cache = root / "shot_16-seed_1.pkl"
            with cache.open("wb") as handle:
                pickle.dump({"train": train, "val": val}, handle)
            base_train, base_val = OxfordPets.subsample_classes(train, val, subsample="base")
            classnames = ["zero", "one"]
            expected_train = probe._sample_rows(base_train, root, classnames)
            expected_val = probe._sample_rows(base_val, root, classnames)
            receipt = h1.validate_fewshot_cache(
                cache, object(), root, expected_train, expected_val)
            self.assertEqual([row["sample_id"] for row in receipt["train"]],
                             ["train-0.png", "train-1.png"])
            with cache.open("wb") as handle:
                pickle.dump({"train": list(reversed(train)), "val": val}, handle)
            with self.assertRaisesRegex(ValueError, "order"):
                h1.validate_fewshot_cache(cache, object(), root, expected_train, expected_val)

    def test_prepare_never_constructs_model_optimizer_or_cuda(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"; data = root / "data"; output = root / "prepared"
            (source / "clip").mkdir(parents=True); (source / "templates").mkdir()
            clip = source / "clip/ViT-B-16.pt"; clip.write_bytes(b"clip")
            template = source / "templates/texture_prompts_full.json"; template.write_text("{}")
            historical_export = root / "historical-export.json"; historical_export.write_text("{}")
            historical_config = root / "historical-config.yaml"; historical_config.write_text("{}")
            metadata = data / "dtd/split.json"; metadata.parent.mkdir(parents=True); metadata.write_text("{}")
            image_dir = data / "dtd/images"; image_dir.mkdir()
            items = []
            for name, label, classname in (("train.png", 0, "zero"),
                                           ("val.png", 0, "zero"),
                                           ("test.png", 0, "zero")):
                path = image_dir / name; Image.new("RGB", (2, 2)).save(path)
                items.append(SimpleNamespace(impath=str(path), label=label, classname=classname))
            train, val, test = ([items[0]], [items[1]], [items[2]])
            train_sha, val_sha = probe.sample_digest(train, data), probe.sample_digest(val, data)
            case = {"train_samples_sha256": train_sha, "val_samples_sha256": val_sha,
                    "clip_sha256": probe.sha256_file(clip),
                    "template_sha256": probe.sha256_file(template),
                    "checkpoints": {"kd1": {"sha256": "c" * 64}}}
            samples = probe.validate_samples(train, val, test, data, ["zero"])
            historical = {"export": str(historical_export), "export_sha256": "e" * 64,
                          "npz": str(root / "historical.npz"), "npz_sha256": "n" * 64,
                          "rows": h1._sample_core(samples["val"]), "class_order": ["zero"],
                          "branches": ["C", "M", "S0", "S1", "T0", "T1"]}

            class Config:
                def dump(self):
                    return "FIXTURE: true\n"

            decoded_batch = [{"img": torch.zeros(1, 3, 2, 2), "label": torch.tensor([0]),
                              "impath": [str(image_dir / "val.png")]}]
            args = SimpleNamespace(source_root=source, data_root=data,
                historical_export=historical_export, historical_config=historical_config,
                dataset="dtd", mode="weighted_nckd", output_dir=output)
            with mock.patch.object(probe, "validate_inventory", return_value=case), \
                    mock.patch.object(h1, "load_reference_snapshot", return_value={}), \
                    mock.patch.object(h1, "validate_historical_reference", return_value={}), \
                    mock.patch.object(h1, "validate_historical_export", return_value=historical), \
                    mock.patch.object(h1, "validate_historical_config", return_value={}), \
                    mock.patch.object(probe, "model_source_digest", return_value=h1.CURRENT_SOURCE_SHA256), \
                    mock.patch.object(probe, "_source_modules", return_value=object()), \
                    mock.patch.object(h1, "build_future_config",
                                      return_value=(Config(), {"FIXTURE": True}, {"changed_fields": []})), \
                    mock.patch.object(probe, "rebuild_source_dataset",
                                      return_value=(train, val, test, ["zero"], [metadata])), \
                    mock.patch.object(h1, "validate_fewshot_cache",
                                      return_value={"path": "cache", "sha256": "f" * 64,
                                                    "train": h1._sample_core(samples["train"]),
                                                    "val": h1._sample_core(samples["val"])}), \
                    mock.patch.object(probe, "build_val_loader",
                                      return_value=(decoded_batch, "transform-fixture")), \
                    mock.patch("trainers.textconsistency_common.load_clip_checkpoint",
                               side_effect=AssertionError("model loaded")), \
                    mock.patch("torch.optim.SGD", side_effect=AssertionError("optimizer built")):
                receipt = h1.prepare_case(args)
            self.assertEqual(receipt["status"], "prepared")
            self.assertEqual(receipt["environment"]["device"], "cpu")
            self.assertTrue((output / "prepared.json").is_file())
            self.assertFalse((output / "future-model-placeholder").exists())


class ExportContracts(unittest.TestCase):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prompt_learner = torch.nn.Linear(2, 2, bias=False)
            self.view_aggregator = torch.nn.Linear(2, 2)
            self.logit_scale = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, images, return_attn=False):
            features = torch.nn.functional.normalize(images.float(), dim=-1)
            student = features @ torch.tensor([[2.0, 0.0], [0.0, 2.0]])
            teacher = features @ torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            return student, teacher, features, None

    def test_strict_reload_then_batch_export_retains_finite_s_t_m_and_ensemble(self):
        first = self.TinyModel()
        modules = torch.nn.ModuleDict({"prompt_learner": first.prompt_learner,
                                       "view_aggregator": first.view_aggregator})
        checkpoint = {"state_dict": {key: value.detach().clone()
                                     for key, value in modules.state_dict().items()}}
        second = self.TinyModel()
        with torch.no_grad():
            second.prompt_learner.weight.zero_()
        h1.strict_load_current_model(second, checkpoint)
        self.assertTrue(torch.equal(second.prompt_learner.weight, first.prompt_learner.weight))
        batch = {"img": torch.eye(2), "label": torch.tensor([0, 1]),
                 "impath": ["a.png", "b.png"]}
        descriptions = torch.tensor([[[1.0, 0.0], [0.0, 1.0]],
                                     [[0.0, 1.0], [9.0, 0.0]]])
        mask = torch.tensor([[True, True], [True, False]])
        result = h1.export_batches(second, [batch], descriptions, mask, ["zero", "one"], device="cpu")
        self.assertEqual(set(result["logits"]), set(h1.BRANCHES))
        np.testing.assert_allclose(result["logits"]["M"],
                                   [[2 ** -0.5, 0.0], [2 ** -0.5, 1.0]], rtol=1e-6)
        np.testing.assert_allclose(result["logits"]["ensemble"],
                                   0.5 * (result["logits"]["S"] + result["logits"]["T"]))
        self.assertTrue(all(np.isfinite(value).all() for value in result["logits"].values()))

        descriptions[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            h1.export_batches(second, [batch], descriptions, mask, ["zero", "one"], device="cpu")
        descriptions[0, 0, 0] = 1.0
        nan = torch.full((2, 2), float("nan"))
        with mock.patch.object(second, "forward",
                               return_value=(nan, torch.ones(2, 2), torch.eye(2), None)), \
                self.assertRaisesRegex(ValueError, "non-finite"):
            h1.export_batches(second, [batch], descriptions, mask, ["zero", "one"], device="cpu")

    def test_public_load_export_revalidates_rows_npz_prepared_and_checkpoint_bindings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared_path = root / "prepared.json"
            prepared_path.write_text("{}\n")
            prepared = {"status": "prepared", "dataset_key": "dtd", "dataset": "DescribableTextures",
                        "seed": 1, "mode": "weighted_nckd", "loss_identity": h1.loss_identity("weighted_nckd"),
                        "source_sha256": h1.CURRENT_SOURCE_SHA256,
                        "samples": {"classes": ["zero", "one"], "val": [
                            {"sample_id": "a.png", "label": 0, "classname": "zero"},
                            {"sample_id": "b.png", "label": 1, "classname": "one"}]},
                        "sample_hashes": {"val": "v"}, "prepared_identity_sha256": "p"}
            prepared["transform"] = {
                "decoded_val": {"count": 2, "tensor_shape": [2],
                                "tensor_dtype": "torch.float32",
                                "ordered_transform_sha256": "d"}}
            payload = {"logits": {name: np.ones((2, 2), dtype=np.float32) for name in h1.BRANCHES},
                       "labels": np.array([0, 1], dtype=np.int64), "sample_ids": ["a.png", "b.png"],
                       "classnames": ["zero", "one"], "forward_precision": "float32",
                       "decoded_val": {"count": 2, "tensor_shape": [2], "tensor_dtype": "torch.float32",
                                       "ordered_transform_sha256": "d"}}
            checkpoint = {"model_dir": str(root / "model"),
                          "checkpoint_sha256": "c" * 64,
                          "manifest_sha256": "m" * 64,
                          "effective_config_sha256": "e" * 64,
                          "loss_identity": h1.loss_identity("weighted_nckd"),
                          "source_sha256": h1.CURRENT_SOURCE_SHA256}
            output = root / "export"
            h1.write_export_artifact(output, payload, prepared_path, prepared, checkpoint)
            with self.assertRaisesRegex(FileExistsError, "fresh"):
                h1.write_export_artifact(output, payload, prepared_path, prepared, checkpoint)
            with mock.patch.object(h1, "load_prepared", return_value=prepared), \
                    mock.patch.object(h1, "validate_new_checkpoint",
                                      return_value=(checkpoint, {})):
                receipt, arrays = h1.load_export(output / "export.json")
                self.assertEqual(arrays["labels"].tolist(), [0, 1])
                changed = json.loads((output / "export.json").read_text())
                changed["rows"] = list(reversed(changed["rows"]))
                (output / "export.json").write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "rows"):
                    h1.load_export(output / "export.json")

    def test_new_checkpoint_rejects_old_source_or_changed_loss_identity_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "textconsistency").mkdir()
            config = {"OUTPUT_DIR": str(root)}
            (root / "effective_config.yaml").write_text("OUTPUT_DIR: " + str(root) + "\n")
            prepared = {
                "dataset": "DescribableTextures", "future_config": config,
                "loss_identity": h1.loss_identity("weighted_nckd"),
                "expected_model_signature": {"loss_identity": h1.loss_identity("weighted_nckd")},
                "sample_hashes": {"train": "train", "val": "val", "test": "test"},
                "expected_trainable_parameters": 1,
            }
            manifest = {
                "dataset": "DescribableTextures", "seed": 1, "shots": 16, "subsample": "base",
                "code_sha256": "old-source", "loss_identity": prepared["loss_identity"],
                "effective_losses": {"kd": 1.0, "adaptive_ce": 1.0,
                                     "diversity": 0.0, "contrastive": 0.0},
                "model_signature": prepared["expected_model_signature"],
                "train_samples_sha256": "train", "val_samples_sha256": "val",
                "test_samples_sha256": "test", "trainable_parameters": 1,
            }
            (root / "run_manifest.json").write_text(json.dumps(manifest))
            with mock.patch("torch.load", side_effect=AssertionError("old checkpoint loaded")), \
                    self.assertRaisesRegex(ValueError, "manifest identity"):
                h1.validate_new_checkpoint(prepared, root)
            manifest["code_sha256"] = h1.CURRENT_SOURCE_SHA256
            manifest["loss_identity"] = h1.loss_identity("full_kl")
            (root / "run_manifest.json").write_text(json.dumps(manifest))
            with mock.patch("torch.load", side_effect=AssertionError("wrong-loss checkpoint loaded")), \
                    self.assertRaisesRegex(ValueError, "manifest identity"):
                h1.validate_new_checkpoint(prepared, root)

    def test_cuda_cli_guard_records_failure_without_loading_prepared_or_model(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = root / "prepared.json"; prepared.write_text("{}")
            output = root / "export"
            with mock.patch("torch.cuda.is_available", return_value=False), \
                    mock.patch.object(h1, "load_prepared", side_effect=AssertionError("loaded prepared")), \
                    self.assertRaisesRegex(RuntimeError, "CUDA"):
                h1.export_case(SimpleNamespace(prepared=prepared, model_dir=root / "model",
                                               output_dir=output))
            self.assertEqual(json.loads((output / "failure.json").read_text())["status"], "failed")


class TinyCurrentModelIntegration(unittest.TestCase):
    def test_actual_tiny_customclip_checkpoint_strict_reload_and_export(self):
        from tests.repro.test_trainers import make_trainer
        from trainers.coop_textconsistency_v2_fg import CoOp_TextConsistency_V2_FG

        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            first = make_trainer(CoOp_TextConsistency_V2_FG, root / "first", kd_mode="weighted_nckd")
            images = torch.randn(3, 3, 32, 32)
            first.forward_backward({"img": images, "label": torch.tensor([0, 1, 2])})
            checkpoint = {"state_dict": {key: value.detach().clone()
                         for key, value in first.trainable_modules.state_dict().items()}}
            second = make_trainer(CoOp_TextConsistency_V2_FG, root / "second", kd_mode="weighted_nckd")
            h1.strict_load_current_model(second.model, checkpoint)
            encoder = second.model.multiview_encoder
            batch = {"img": images, "label": torch.tensor([0, 1, 2]),
                     "impath": ["a.png", "b.png", "c.png"]}
            result = h1.export_batches(second.model, [batch], encoder.get_view_features(),
                                       encoder.view_mask, second.dm.dataset.classnames, device="cpu")
            self.assertTrue(all(value.shape == (3, 3) for value in result["logits"].values()))
            self.assertTrue(all(np.isfinite(value).all() for value in result["logits"].values()))


if __name__ == "__main__":
    unittest.main()
