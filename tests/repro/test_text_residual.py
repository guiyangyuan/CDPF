"""Two short checks for the changed text correction path; no dataset scores."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from dassl.config import get_cfg_default
from dassl.engine.trainer import TrainerBase
import train
from trainers.coop_text_residual import CoOp_TextResidual, neighbor_inputs
from .test_trainers import tiny_clip
from .test_description_correction import make_correction
from scripts.repro.text_residual_pilot import collect_predictions

ROOT = Path(__file__).resolve().parents[2]


def make_text(output, mode='contrast', novel=False, adamw=False, detail=None):
    cfg = get_cfg_default(); train.extend_cfg(cfg)
    folder = 'text_residual_adamw' if adamw else 'text_residual'
    config = f'text_detail/{detail}' if detail else f'{folder}/{mode}'
    cfg.merge_from_file(str(ROOT / f'configs/repro/{config}.yaml'))
    cfg.SEED = 1; cfg.DATASET.NAME = 'Caltech101'
    cfg.DATASET.ROOT = str(output); cfg.OUTPUT_DIR = str(output)
    cfg.DATASET.SUBSAMPLE_CLASSES = 'new' if novel else 'base'
    cfg.OPTIM.WARMUP_EPOCH = 0
    cfg.TRAINER.TEXTCONSISTENCY.N_VIEWS = 5
    cfg.TRAINER.TEXTCONSISTENCY.TEXT_RESIDUAL_DIM = 8
    from trainers.coop_text_residual_adamw import CoOp_TextResidual_AdamW
    from trainers.coop_text_detail import (
        CoOp_TextDetail, CoOp_TextDetail_FusionAligned,
        CoOp_TextDetail_RadiusBounded,
        CoOp_TextDetail_SubspaceRadius,
        CoOp_TextDetail_SemanticGatedRadius,
    )
    cls = (CoOp_TextDetail_SemanticGatedRadius if detail == 'semantic_gated_radius_attentive'
           else CoOp_TextDetail_SubspaceRadius if detail == 'subspace_radius_attentive'
           else CoOp_TextDetail_RadiusBounded if detail == 'radius_bounded_attentive'
           else CoOp_TextDetail_FusionAligned if detail == 'fusion_aligned_attentive'
           else CoOp_TextDetail if detail
           else CoOp_TextResidual_AdamW if adamw else CoOp_TextResidual)
    trainer = cls.__new__(cls)
    TrainerBase.__init__(trainer); trainer.cfg = cfg
    names = list(json.loads((ROOT / 'templates/cal_prompts_full.json').read_text()))
    names = names[5:9] if novel else names[:3]
    trainer.dm = SimpleNamespace(dataset=SimpleNamespace(classnames=names, train_x=[], val=[], test=[]))
    trainer.check_cfg(cfg)
    with patch('trainers.textconsistency_runtime.load_clip_checkpoint', side_effect=tiny_clip):
        trainer.build_model()
    trainer.batch_idx = 0; trainer.num_batches = 3
    return trainer


class TextResidualSmoke(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.capture = contextlib.redirect_stdout(io.StringIO()); self.capture.__enter__()
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup(); self.capture.__exit__(None, None, None)

    def test_two_steps_and_text_geometry(self):
        baseline = make_correction(self.root / 'old', 'student')
        models = {mode: make_text(self.root / mode, mode) for mode in ('classname', 'description', 'contrast')}
        self.assertEqual(len({x.run_provenance['trainable_parameters'] for x in models.values()}), 1)
        batch = {'img': torch.randn(3, 3, 32, 32), 'label': torch.tensor([0, 1, 2])}
        for trainer in models.values():
            outputs = trainer.model(batch['img'])
            torch.testing.assert_close(outputs['Z'], baseline.model(batch['img'])['S'], rtol=0, atol=0)
            ce, correction = trainer.loss_terms(outputs, batch['label'])
            self.assertIsNone(torch.autograd.grad(correction, trainer.model.prompt_learner.ctx,
                                                 allow_unused=True, retain_graph=True)[0])
        for step in range(2):
            baseline.batch_idx = step; baseline.forward_backward(batch)
            for trainer in models.values():
                trainer.batch_idx = step
                metrics = trainer.forward_backward(batch)
                self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in metrics.values()))
                torch.testing.assert_close(trainer.model.prompt_learner.ctx,
                                           baseline.model.prompt_learner.ctx, rtol=0, atol=0)
                self.assertTrue(all(p.grad is None for p in trainer.model.image_encoder.parameters()))
                self.assertTrue(all(p.grad is None for p in trainer.model.text_encoder.parameters()))
                self.assertGreater(torch.count_nonzero(trainer.model.text_residual.up.weight).item(), 0)
        names = torch.eye(3)
        centered, neighbors = neighbor_inputs(names[:, None, :], names, 5)
        self.assertTrue(all(i not in row for i, row in enumerate(neighbors.tolist())))
        torch.testing.assert_close(centered[:, 0], names - (names.sum(0) - names) / 2)

    def test_checkpoint_and_new_class_text(self):
        first = make_text(self.root / 'first')
        images = torch.randn(3, 3, 32, 32)
        first.forward_backward({'img': images, 'label': torch.tensor([0, 1, 2])})
        first.save_model(0, self.root / 'first')
        same, novel = make_text(self.root / 'same'), make_text(self.root / 'novel', novel=True)
        novel_views = novel.model.residual_views.clone()
        for target in (same, novel):
            target.load_model(self.root / 'first', 1)
            torch.testing.assert_close(first.model.text_residual.up.weight,
                                       target.model.text_residual.up.weight, rtol=0, atol=0)
        for key, value in first.model(images).items():
            torch.testing.assert_close(value, same.model(images)[key], rtol=0, atol=0)
        torch.testing.assert_close(novel_views, novel.model.residual_views, rtol=0, atol=0)
        self.assertEqual(tuple(novel.model(images)['Z'].shape), (3, 4))
        exported, _ = collect_predictions(same.model, [{'img': images, 'label': torch.tensor([0, 1, 2]),
                                                        'impath': ['a', 'b', 'c']}], 'cpu')
        torch.testing.assert_close(torch.from_numpy(exported['Z']), same.model_inference(images), rtol=0, atol=0)
        wrong = make_text(self.root / 'wrong', 'description')
        with self.assertRaisesRegex(ValueError, 'signature mismatch'):
            wrong.load_model(self.root / 'first', 1)


if __name__ == '__main__':
    unittest.main()
