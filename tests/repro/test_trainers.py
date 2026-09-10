"""Integration tests with the real CLIP architecture at tiny dimensions.
Only loading external pretrained weights is substituted; trainers and losses are real.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from clip.model import CLIP
from dassl.config import get_cfg_default
from dassl.engine.trainer import TrainerBase
from train import extend_cfg
from trainers.coop_textconsistency_v2_fg import CoOp_TextConsistency_V2_FG
from trainers.cocoop_textconsistency_v2_fg import CoCoOp_TextConsistency_V2_FG
from trainers.coop_textconsistency_ablation import CoOp_TextConsistency_Ablation

ROOT = Path(__file__).resolve().parents[2]


def tiny_clip(cfg):
    torch.manual_seed(31)
    model = CLIP(32, 32, 1, 64, 16, 77, 49408, 64, 1, 1,
                 {'trainer': 'CoOp', 'vision_depth': 0, 'language_depth': 0, 'vision_ctx': 0, 'language_ctx': 0})
    model._repro_sha256 = 'tiny-test-model-not-pretrained'
    return model.float()


def make_trainer(trainer_cls, output, mode='hierarchical', adapter=True, group='full'):
    cfg = get_cfg_default(); extend_cfg(cfg)
    cfg.SEED = 1; cfg.USE_CUDA = False; cfg.DATASET.ROOT = str(output)
    cfg.DATASET.NAME = 'Caltech101'; cfg.DATASET.SUBSAMPLE_CLASSES = 'base'
    cfg.DATASET.NUM_SHOTS = 16; cfg.OUTPUT_DIR = str(output)
    cfg.TRAINER.NAME = trainer_cls.__name__
    cfg.MODEL.BACKBONE.NAME = 'ViT-B/16'
    cfg.OPTIM.MAX_EPOCH = 2; cfg.OPTIM.WARMUP_EPOCH = 0
    tc = cfg.TRAINER.TEXTCONSISTENCY
    tc.N_VIEWS = 5; tc.N_GROUPS = 2; tc.N_CTX = 4
    tc.KD_WEIGHT = 1.; tc.DEVICE = 'cpu'; tc.AGGREGATION = mode
    tc.USE_FEATURE_ADAPTER = adapter; tc.ABLATION_GROUP = group
    if group == 'baseline': tc.INFERENCE_BRANCH = 'student'
    trainer = trainer_cls.__new__(trainer_cls); TrainerBase.__init__(trainer)
    trainer.cfg = cfg
    names = list(json.loads((ROOT / 'templates/cal_prompts_full.json').read_text()))[:3]
    trainer.dm = SimpleNamespace(dataset=SimpleNamespace(classnames=names, train_x=[], val=[], test=[]))
    trainer.check_cfg(cfg)
    with patch('trainers.textconsistency_runtime.load_clip_checkpoint', side_effect=tiny_clip):
        trainer.build_model()
    trainer.batch_idx = 0; trainer.num_batches = 1
    return trainer


class TrainerIntegration(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.capture = contextlib.redirect_stdout(io.StringIO()); self.capture.__enter__()

    def tearDown(self):
        self.capture.__exit__(None, None, None)

    def test_real_training_updates_student_and_aggregator_while_clip_stays_frozen(self):
        for cls in (CoOp_TextConsistency_V2_FG, CoCoOp_TextConsistency_V2_FG):
            with self.subTest(trainer=cls.__name__), tempfile.TemporaryDirectory() as temp:
                trainer = make_trainer(cls, Path(temp))
                model = trainer.model
                before = model.prompt_learner.ctx.detach().clone()
                adapter_before = model.view_aggregator.fine_grained_proj[0].weight.detach().clone()
                trainer.forward_backward({'img': torch.randn(3, 3, 32, 32), 'label': torch.tensor([0, 1, 2])})
                self.assertFalse(torch.equal(before, model.prompt_learner.ctx))
                self.assertFalse(torch.equal(adapter_before, model.view_aggregator.fine_grained_proj[0].weight))
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.image_encoder.parameters()))
                self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.text_encoder.parameters()))

    def test_save_reload_reproduces_all_prediction_branches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = make_trainer(CoOp_TextConsistency_V2_FG, root / 'first')
            images = torch.randn(3, 3, 32, 32)
            first.forward_backward({'img': images, 'label': torch.tensor([0, 1, 2])})
            first.save_model(0, root / 'first')
            second = make_trainer(CoOp_TextConsistency_V2_FG, root / 'second')
            second.load_model(root / 'first', 1)
            for branch in ('student', 'teacher', 'ensemble'):
                for trainer in (first, second):
                    trainer.cfg.TRAINER.TEXTCONSISTENCY.INFERENCE_BRANCH = branch
                    trainer.model.eval()
                torch.testing.assert_close(first.model_inference(images), second.model_inference(images), rtol=0, atol=0)
            self.assertEqual(second.loaded_checkpoints[0]['epoch'], 1)

    def test_checkpoint_rejects_different_grouping_even_when_tensor_shapes_match(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trainer = make_trainer(CoOp_TextConsistency_V2_FG, root / 'one')
            trainer.save_model(0, root / 'one')
            trainer.repro_signature = {**trainer.repro_signature, 'n_groups': 3}
            with self.assertRaises(ValueError): trainer.load_model(root / 'one', 1)

    def test_checkpoint_cannot_be_relabelled_as_a_different_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trainer = make_trainer(CoOp_TextConsistency_V2_FG, root / 'one')
            trainer.save_model(0, root / 'one')
            trainer.run_provenance = {**trainer.run_provenance, 'seed': 2}
            with self.assertRaises(ValueError): trainer.load_model(root / 'one', 1)

    def test_cocoop_aggregation_obeys_configuration_for_every_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = make_trainer(CoCoOp_TextConsistency_V2_FG, Path(temp), 'flat', False)
            self.assertEqual(trainer.model.view_aggregator.mode, 'flat')
            self.assertIsNone(trainer.model.view_aggregator.fine_grained_proj)
            self.assertEqual(trainer.diversity_weight, 0.)

    def test_baseline_without_teacher_trains_and_predicts(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = make_trainer(CoOp_TextConsistency_Ablation, Path(temp), group='baseline')
            images = torch.randn(3, 3, 32, 32)
            result = trainer.forward_backward({'img': images, 'label': torch.tensor([0, 1, 2])})
            self.assertEqual(result['loss_adaptive_ce'], 0.)
            self.assertEqual(tuple(trainer.model_inference(images).shape), (3, 3))

if __name__ == '__main__':
    unittest.main()
