"""Short CPU checks: real tiny CLIP, two batches, one checkpoint round-trip.

No dataset score is produced. Only external pretrained weight loading is stubbed.
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
from dassl.config import get_cfg_default
from dassl.engine.trainer import TrainerBase
import train
from trainers.coop_textconsistency_correction import CoOp_TextConsistency_Correction
from trainers.coop_textconsistency_v2_fg import CoOp_TextConsistency_V2_FG
from .test_trainers import tiny_clip
from scripts.repro.description_correction_pilot import collect_predictions

ROOT = Path(__file__).resolve().parents[2]


def make_correction(output, variant='description', novel=False, legacy=False):
    cfg = get_cfg_default(); train.extend_cfg(cfg)
    cfg.merge_from_file(str(ROOT / f'configs/repro/description_correction/{variant}.yaml'))
    cfg.SEED = 1; cfg.DATASET.NAME = 'Caltech101'
    cfg.DATASET.ROOT = str(output); cfg.OUTPUT_DIR = str(output)
    cfg.DATASET.SUBSAMPLE_CLASSES = 'new' if novel else 'base'
    cfg.OPTIM.WARMUP_EPOCH = 0
    cfg.TRAINER.TEXTCONSISTENCY.N_VIEWS = 5
    cfg.TRAINER.TEXTCONSISTENCY.VISUAL_ADAPTER_DIM = 8
    trainer_cls = CoOp_TextConsistency_V2_FG if legacy else CoOp_TextConsistency_Correction
    cfg.TRAINER.NAME = trainer_cls.__name__
    trainer = trainer_cls.__new__(trainer_cls)
    TrainerBase.__init__(trainer); trainer.cfg = cfg
    names = list(json.loads((ROOT / 'templates/cal_prompts_full.json').read_text()))
    names = names[5:9] if novel else names[:3]
    trainer.dm = SimpleNamespace(dataset=SimpleNamespace(classnames=names, train_x=[], val=[], test=[]))
    trainer.check_cfg(cfg)
    with patch('trainers.textconsistency_runtime.load_clip_checkpoint', side_effect=tiny_clip):
        trainer.build_model()
    trainer.batch_idx = 0; trainer.num_batches = 3
    return trainer


class DescriptionCorrectionSmoke(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.capture = contextlib.redirect_stdout(io.StringIO()); self.capture.__enter__()
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup(); self.capture.__exit__(None, None, None)

    def test_two_steps_preserve_student_and_update_only_expected_parameters(self):
        baseline = make_correction(self.root / 'old', 'student', legacy=True)
        controls = {name: make_correction(self.root / name, name)
                    for name in ('student', 'classname', 'description')}
        self.assertEqual(controls['classname'].run_provenance['trainable_parameters'],
                         controls['description'].run_provenance['trainable_parameters'])
        batch = {'img': torch.randn(3, 3, 32, 32), 'label': torch.tensor([0, 1, 2])}
        before = {name: {k: p.detach().clone() for k, p in trainer.model.named_parameters()}
                  for name, trainer in controls.items()}
        original_s, original_m, _, _ = baseline.model(batch['img'])
        for name, trainer in controls.items():
            outputs = trainer.model(batch['img'])
            torch.testing.assert_close(outputs['S'], original_s, rtol=0, atol=0)
            torch.testing.assert_close(outputs['M'], original_m, rtol=0, atol=0)
            torch.testing.assert_close(outputs['Z'], outputs['S'], rtol=0, atol=0)
            self.assertEqual(torch.count_nonzero(outputs['R']).item(), 0)
            if name != 'student':
                ce, correction = trainer.loss_terms(outputs, batch['label'])
                ctx = trainer.model.prompt_learner.ctx
                self.assertIsNone(torch.autograd.grad(correction, ctx, allow_unused=True, retain_graph=True)[0])
                expected = torch.autograd.grad(ce, ctx, retain_graph=True)[0]
                actual = torch.autograd.grad(ce + correction, ctx)[0]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for step in range(2):
            baseline.batch_idx = step; baseline.num_batches = 3
            baseline.forward_backward(batch)
            for name, trainer in controls.items():
                trainer.batch_idx = step
                metrics = trainer.forward_backward(batch)
                self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in metrics.values()))
                torch.testing.assert_close(trainer.model.prompt_learner.ctx,
                                           baseline.model.prompt_learner.ctx, rtol=0, atol=0)
        for name, trainer in controls.items():
            for key, param in trainer.model.named_parameters():
                if key.startswith(('prompt_learner.', 'visual_adapter.')):
                    self.assertFalse(torch.equal(param, before[name][key]), key)
                else:
                    torch.testing.assert_close(param, before[name][key], rtol=0, atol=0)
                    self.assertIsNone(param.grad)
            if name != 'student':
                self.assertGreater(torch.count_nonzero(trainer.model(batch['img'])['R']).item(), 0)
        self.assertFalse(torch.equal(controls['classname'].model.visual_adapter.up.weight,
                                     controls['description'].model.visual_adapter.up.weight))

    def test_checkpoint_roundtrip_and_rebuilt_novel_text(self):
        first = make_correction(self.root / 'first')
        images = torch.randn(3, 3, 32, 32)
        first.forward_backward({'img': images, 'label': torch.tensor([0, 1, 2])})
        first.save_model(0, self.root / 'first')
        same = make_correction(self.root / 'same')
        novel = make_correction(self.root / 'novel', novel=True)
        novel_text = novel.model.correction_text.clone()
        for target in (same, novel):
            target.load_model(self.root / 'first', 1)
            torch.testing.assert_close(target.model.visual_adapter.up.weight,
                                       first.model.visual_adapter.up.weight, rtol=0, atol=0)
        for key, tensor in first.model(images).items():
            torch.testing.assert_close(tensor, same.model(images)[key], rtol=0, atol=0)
        exported, paths = collect_predictions(same.model, [{'img': images, 'label': torch.tensor([0, 1, 2]),
                                                           'impath': ['a.jpg', 'b.jpg', 'c.jpg']}], 'cpu')
        self.assertEqual(paths, ['a.jpg', 'b.jpg', 'c.jpg'])
        torch.testing.assert_close(torch.from_numpy(exported['Z']), first.model_inference(images), rtol=0, atol=0)
        self.assertEqual(tuple(novel.model_inference(images).shape), (3, 4))
        torch.testing.assert_close(novel.model.correction_text, novel_text, rtol=0, atol=0)
        wrong_text = make_correction(self.root / 'wrong_text', 'classname')
        with self.assertRaisesRegex(ValueError, 'signature mismatch'):
            wrong_text.load_model(self.root / 'first', 1)
        self.assertEqual(first.run_provenance['effective_losses']['correction_ce'], 1.0)
        self.assertEqual(first.repro_signature['correction']['text_source'], 'description')


if __name__ == '__main__':
    unittest.main()
