import subprocess
import sys
import hashlib
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.repro import validation_probe as probe


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)


class ValidationProbeCli(unittest.TestCase):
    def test_help_exposes_prepare_and_export_without_importing_clip(self):
        result = subprocess.run(
            [str(PYTHON), "scripts/repro/validation_probe.py", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("prepare", result.stdout)
        self.assertIn("export", result.stdout)


class ValidationProbeContracts(unittest.TestCase):
    @staticmethod
    def _role_config(role):
        is_coop = role == "coop"
        tc = SimpleNamespace(
            ABLATION_GROUP="baseline" if is_coop else "full",
            ADAPTIVE_CE_WEIGHT=1.0,
            AGGREGATION="mean" if is_coop else "hierarchical",
            CLIP_PATH="unused",
            CONSISTENCY_TEMP=1.0,
            CONSISTENCY_WEIGHT=1.0,
            CONTRASTIVE_WEIGHT=0.0,
            DEVICE="cpu",
            DIVERSITY_WEIGHT=0.0,
            ENSEMBLE_ALPHA=0.5,
            INFERENCE_BRANCH="student" if is_coop else "ensemble",
            KD_TEMP=2.0,
            KD_WEIGHT={"coop": 0.0, "kd0": 0.0, "kd1": 1.0}[role],
            N_CTX=4,
            N_GROUPS=4,
            N_VIEWS=20,
            ORDER_SEED=0,
            PREC="fp32",
            TEXT_BATCH_SIZE=32,
            USE_FEATURE_ADAPTER=not is_coop,
            VIEW_ORDER="contiguous",
        )
        return SimpleNamespace(
            SEED=1,
            USE_CUDA=False,
            TEST=SimpleNamespace(SPLIT="val"),
            DATASET=SimpleNamespace(NAME="DescribableTextures", NUM_SHOTS=16,
                                    SUBSAMPLE_CLASSES="base"),
            DATALOADER=SimpleNamespace(NUM_WORKERS=0,
                TEST=SimpleNamespace(SAMPLER="SequentialSampler", BATCH_SIZE=16)),
            MODEL=SimpleNamespace(BACKBONE=SimpleNamespace(NAME="ViT-B/16")),
            INPUT=SimpleNamespace(SIZE=(224, 224), INTERPOLATION="bicubic",
                PIXEL_MEAN=(0.48145466, 0.4578275, 0.40821073),
                PIXEL_STD=(0.26862954, 0.26130258, 0.27577711), NO_TRANSFORM=False,
                TRANSFORMS=("random_resized_crop", "random_flip", "normalize")),
            TRAINER=SimpleNamespace(NAME="CoOp_TextConsistency_Ablation" if is_coop
                else "CoOp_TextConsistency_V2_FG", TEXTCONSISTENCY=tc),
        )

    @staticmethod
    def _role_manifest(role):
        is_coop = role == "coop"
        return {
            "effective_losses": {"kd": {"coop": 0.0, "kd0": 0.0, "kd1": 1.0}[role],
                "adaptive_ce": 1.0, "diversity": 0.0, "contrastive": 0.0},
            "model_signature": {
                "format": 2,
                "trainer": "CoOp_TextConsistency_Ablation" if is_coop else "CoOp_TextConsistency_V2_FG",
                "n_ctx": 4, "aggregation": "none" if is_coop else "hierarchical",
                "feature_adapter": not is_coop, "n_groups": None if is_coop else 4,
                "n_views": 20, "ablation_group": "baseline" if is_coop else None,
                "clip_sha256": probe.CLIP_SHA256, "template_sha256": None if is_coop else "4" * 64,
                "view_order": "contiguous", "order_seed": 0, "order_version": 1,
            },
        }

    def test_inventory_rejects_missing_role_parameters_and_wrong_protocol(self):
        case = {
            "dataset": "dtd", "seed": 1, "split": "val", "candidate_classes": "base",
            "train_samples_sha256": "1" * 64, "val_samples_sha256": "2" * 64,
            "clip_sha256": probe.CLIP_SHA256, "template_sha256": "4" * 64,
            "coop_training_template_sha256": None,
            "checkpoints": {
                role: {
                    "workspace_relative_checkpoint": f"{role}/model.pth.tar-20",
                    "sha256": "5" * 64, "manifest_sha256": "6" * 64,
                    "model_source_sha256": probe.NEW_SOURCE_SHA256,
                    "effective_kd_weight": weight, "epoch": 20,
                }
                for role, weight in (("coop", 0.0), ("kd0", 0.0), ("kd1", 1.0))
            },
        }
        validated = probe.validate_inventory({"cases": [case]}, "dtd", 1)
        self.assertEqual(validated["split"], "val")
        del case["checkpoints"]["kd1"]["effective_kd_weight"]
        with self.assertRaisesRegex(ValueError, "schema"):
            probe.validate_inventory({"cases": [case]}, "dtd", 1)
        case["checkpoints"]["kd1"]["effective_kd_weight"] = 1.0
        case["candidate_classes"] = "new"
        with self.assertRaisesRegex(ValueError, "Base"):
            probe.validate_inventory({"cases": [case]}, "dtd", 1)

    def test_sample_validation_catches_missing_val_overlap_and_permutations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("a.png", "b.png", "c.png"):
                (root / name).write_bytes(b"image")
            train = [SimpleNamespace(impath=str(root / "a.png"), label=0, classname="zero")]
            val = [SimpleNamespace(impath=str(root / "b.png"), label=0, classname="zero")]
            test = [SimpleNamespace(impath=str(root / "c.png"), label=0, classname="zero")]
            result = probe.validate_samples(train, val, test, root, ["zero"])
            self.assertEqual(result["classes"], ["zero"])
            self.assertEqual(result["val"][0]["sample_id"], "b.png")
            with self.assertRaisesRegex(ValueError, "empty"):
                probe.validate_samples(train, [], test, root, ["zero"])
            with self.assertRaisesRegex(ValueError, "overlap"):
                probe.validate_samples(train, train, test, root, ["zero"])
            bad = [SimpleNamespace(impath=str(root / "b.png"), label=0, classname="wrong")]
            with self.assertRaisesRegex(ValueError, "class"):
                probe.validate_samples(train, bad, test, root, ["zero"])

    def test_historical_sample_digest_is_order_and_label_sensitive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "dataset").mkdir()
            for name in ("a.png", "b.png"):
                (root / "dataset" / name).write_bytes(b"x")
            rows = [
                SimpleNamespace(impath=str(root / "dataset/a.png"), label=0, classname="zero"),
                SimpleNamespace(impath=str(root / "dataset/b.png"), label=1, classname="one"),
            ]
            self.assertEqual(
                probe.sample_digest(rows, root),
                "6a9e6d9621742fc1cf1087f6e43a9c94464776c9f5be0030f2588551effa91b6",
            )
            self.assertNotEqual(probe.sample_digest(list(reversed(rows)), root), probe.sample_digest(rows, root))

    def test_effective_config_is_bound_to_authenticated_signature_and_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)
            (source / "clip").mkdir()
            clip = source / "clip/ViT-B-16.pt"
            clip.write_bytes(b"fixture")
            case = {"dataset": "dtd", "seed": 1, "clip_sha256": probe.CLIP_SHA256,
                    "template_sha256": "4" * 64}
            for role in probe.ROLES:
                cfg = self._role_config(role)
                cfg.TRAINER.TEXTCONSISTENCY.CLIP_PATH = str(clip)
                probe.validate_effective_role_config(
                    role, cfg, self._role_manifest(role), case, source, cuda=False)
            for role, field, changed in (
                ("coop", "N_CTX", 8),
                ("kd0", "N_VIEWS", 19),
                ("kd0", "N_GROUPS", 5),
                ("kd1", "USE_FEATURE_ADAPTER", False),
                ("kd1", "USE_VISUAL_ADAPTER", True),
            ):
                with self.subTest(role=role, field=field):
                    cfg = self._role_config(role)
                    cfg.TRAINER.TEXTCONSISTENCY.CLIP_PATH = str(clip)
                    setattr(cfg.TRAINER.TEXTCONSISTENCY, field, changed)
                    with self.assertRaisesRegex(ValueError, "effective config"):
                        probe.validate_effective_role_config(
                            role, cfg, self._role_manifest(role), case, source, cuda=False)
            for transforms in (
                ("random_resized_crop", "random_flip"),
                ("random_resized_crop", "random_flip", "normalize", "instance_norm"),
            ):
                with self.subTest(transforms=transforms):
                    cfg = self._role_config("coop")
                    cfg.TRAINER.TEXTCONSISTENCY.CLIP_PATH = str(clip)
                    cfg.INPUT.TRANSFORMS = transforms
                    with self.assertRaisesRegex(ValueError, "effective config"):
                        probe.validate_effective_role_config(
                            "coop", cfg, self._role_manifest("coop"), case, source, cuda=False)

    def test_strict_load_verifies_class_buffers_and_every_trainable_tensor(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.prompt_learner = torch.nn.Linear(2, 2, bias=False)
                self.prompt_learner.register_buffer("token_prefix", torch.ones(1, 1))
                self.view_aggregator = torch.nn.Linear(2, 2)

        model = Tiny()
        saved = torch.nn.ModuleDict({
            "prompt_learner": model.prompt_learner,
            "view_aggregator": model.view_aggregator,
        }).state_dict()
        checkpoint = {"state_dict": {key: value.clone() for key, value in saved.items()}}
        probe.strict_load_trainable_modules(model, checkpoint)
        missing = {"state_dict": dict(checkpoint["state_dict"])}
        missing["state_dict"].pop("view_aggregator.bias")
        with self.assertRaisesRegex(ValueError, "missing"):
            probe.strict_load_trainable_modules(model, missing)
        bad_buffer = {"state_dict": {key: value.clone() for key, value in checkpoint["state_dict"].items()}}
        bad_buffer["state_dict"]["prompt_learner.token_prefix"].zero_()
        with self.assertRaisesRegex(ValueError, "class-dependent"):
            probe.strict_load_trainable_modules(model, bad_buffer)

    def test_export_batches_uses_direct_frozen_forwards_and_fixed_mean_formula(self):
        class TinyModel(torch.nn.Module):
            def __init__(self, student, teacher, features):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.student = student
                self.teacher = teacher
                self.features = features
                self.logit_scale = torch.tensor(0.0)

            def forward(self, images, return_attn=False):
                if torch.is_grad_enabled() or any(p.requires_grad for p in self.parameters()):
                    raise AssertionError("export enabled backward")
                return self.student[:len(images)], self.teacher[:len(images)] if self.teacher is not None else None, self.features[:len(images)], None

        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        models = {
            "coop": TinyModel(torch.tensor([[3.0, 0.0], [0.0, 3.0]]), None, features),
            "kd0": TinyModel(torch.tensor([[2.0, 0.0], [0.0, 2.0]]), torch.tensor([[1.0, 0.0], [0.0, 1.0]]), features),
            "kd1": TinyModel(torch.tensor([[4.0, 0.0], [0.0, 4.0]]), torch.tensor([[2.0, 0.0], [0.0, 2.0]]), features),
        }
        descriptions = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 9.0]]])
        mask = torch.tensor([[True, True], [True, False]])
        batches = [{"img": torch.ones(2, 1), "label": torch.tensor([0, 1]), "impath": ["a", "b"]}]
        decoded = probe._decode_evidence(batches)
        with mock.patch("torch.optim.SGD", side_effect=AssertionError("optimizer created")), \
                mock.patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward called")):
            result = probe.export_batches(models, batches, descriptions, mask, ["zero", "one"])
        self.assertEqual(set(result["logits"]), set(probe.BRANCHES))
        np.testing.assert_allclose(result["logits"]["M"], [[2 ** -0.5, 0.0], [2 ** -0.5, 1.0]], rtol=1e-6)
        np.testing.assert_array_equal(result["labels"], [0, 1])
        self.assertTrue(result["teacher_logits_exact_equal"] is False)
        self.assertEqual(result["decoded_val"], decoded)
        mutations = (
            [{"img": torch.zeros(2, 1), "label": torch.tensor([0, 1]), "impath": ["a", "b"]}],
            [{"img": torch.ones(2, 1), "label": torch.tensor([1, 0]), "impath": ["a", "b"]}],
            [{"img": torch.ones(2, 1), "label": torch.tensor([0, 1]), "impath": ["b", "a"]}],
        )
        for mutated in mutations:
            with self.subTest(mutated=mutated[0]["impath"]), \
                    self.assertRaisesRegex(ValueError, "transform digest"):
                probe.validate_decoded_transform(probe._decode_evidence(mutated), decoded)

    def test_write_export_rejects_collision_and_writes_no_object_arrays(self):
        payload = {
            "logits": {name: np.ones((2, 2), dtype=np.float32) for name in probe.BRANCHES},
            "labels": np.array([0, 1], dtype=np.int64),
            "sample_ids": ["a", "b"], "classnames": ["zero", "one"],
            "teacher_exact_equal": True, "forward_precision": "float32",
        }
        metadata = {"dataset_key": "dtd", "seed": 1, "protocol": probe.PROTOCOL}
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "export"
            receipt = probe.write_export_artifact(target, payload, metadata)
            with np.load(target / "logits.npz", allow_pickle=False) as archive:
                self.assertTrue(all(archive[key].dtype != object for key in archive.files))
            self.assertEqual(receipt["status"], "passed")
            with self.assertRaisesRegex(FileExistsError, "fresh"):
                probe.write_export_artifact(target, payload, metadata)

    def test_export_rejects_nonfinite_or_wrong_shape_before_writing_success(self):
        payload = {
            "logits": {name: np.ones((2, 2), dtype=np.float32) for name in probe.BRANCHES},
            "labels": np.array([0, 1]), "sample_ids": ["a", "b"],
            "classnames": ["zero", "one"], "teacher_exact_equal": True,
            "forward_precision": "float32",
        }
        with tempfile.TemporaryDirectory() as temp:
            payload["logits"]["M"][0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "finite"):
                probe.write_export_artifact(Path(temp) / "nonfinite", payload, {})
            payload["logits"]["M"] = np.ones((1, 2), dtype=np.float32)
            with self.assertRaisesRegex(ValueError, "finite"):
                probe.write_export_artifact(Path(temp) / "shape", payload, {})

    def test_all_shared_clip_models_are_constructed_before_any_device_move(self):
        clip = SimpleNamespace(moved=False)

        class Tiny(torch.nn.Module):
            def __init__(self, cfg, classnames, shared_clip):
                super().__init__()
                if shared_clip.moved:
                    raise AssertionError("constructed after shared CLIP moved")
                self.shared_clip = shared_clip
                self.prompt_learner = torch.nn.Linear(1, 1, bias=False)
                self.view_aggregator = torch.nn.Identity()

            def to(self, device):
                self.shared_clip.moved = True
                return self

        states = {role: {"state_dict": {"prompt_learner.weight": torch.ones(1, 1)}}
                  for role in probe.ROLES}
        models = probe.instantiate_frozen_models(
            {role: Tiny for role in probe.ROLES}, {role: object() for role in probe.ROLES},
            ["one"], clip, states, "fixture-device")
        self.assertEqual(set(models), set(probe.ROLES))
        self.assertTrue(all(not parameter.requires_grad for model in models.values()
                            for parameter in model.parameters()))

    def test_cuda_rejection_records_failure_without_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = root / "prepared.json"
            prepared.write_text(json.dumps({
                "workspace_root": str(root / "workspace"), "source_root": str(root / "source"),
                "data_root": str(root / "data"),
                "checkpoints": {role: {"checkpoint": str(root / f"{role}.pt")} for role in probe.ROLES},
            }))
            output = root / "export"
            with mock.patch("torch.cuda.is_available", return_value=False), \
                    self.assertRaisesRegex(RuntimeError, "CUDA"):
                probe.export_case(SimpleNamespace(prepared=prepared, output_dir=output))
            failure = json.loads((output / "failure.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertIn("CUDA", failure["error"])

    def test_export_moves_loader_images_to_the_model_device(self):
        moved = []

        class Images:
            def detach(self):
                return torch.ones(1, 1)

            def to(self, device):
                moved.append(device)
                return torch.ones(1, 1)

        class Tiny(torch.nn.Module):
            def __init__(self, teacher):
                super().__init__(); self.weight = torch.nn.Parameter(torch.ones(()))
                self.teacher = teacher; self.logit_scale = torch.tensor(0.0)

            def forward(self, images, return_attn=False):
                feature = torch.tensor([[1.0, 0.0]])
                logits = torch.tensor([[1.0, 0.0]])
                return logits, logits if self.teacher else None, feature, None

        models = {"coop": Tiny(False), "kd0": Tiny(True), "kd1": Tiny(True)}
        probe.export_batches(models, [{"img": Images(), "label": torch.tensor([0]), "impath": ["a"]}],
            torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]), torch.ones(2, 1, dtype=torch.bool),
            ["zero", "one"], device="cuda-fixture")
        self.assertEqual(moved, ["cuda-fixture"])


if __name__ == "__main__":
    unittest.main()
