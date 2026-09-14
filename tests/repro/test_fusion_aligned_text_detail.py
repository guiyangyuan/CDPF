"""Targeted checks for the fusion-aligned text-detail objective."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from .test_text_residual import make_text


class FusionAlignedTextDetailTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.capture = contextlib.redirect_stdout(io.StringIO())
        self.capture.__enter__()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()
        self.capture.__exit__(None, None, None)

    def test_corrected_branch_and_loss_share_the_same_fusion_anchor(self):
        trainer = make_text(self.root / 'aligned', detail='fusion_aligned_attentive')
        batch = {'img': torch.randn(3, 3, 32, 32), 'label': torch.tensor([0, 1, 2])}
        outputs = trainer.model(batch['img'])

        torch.testing.assert_close(outputs['Z'], outputs['F'] + outputs['R'], rtol=0, atol=0)
        student, correction = trainer.loss_terms(outputs, batch['label'])
        torch.testing.assert_close(student, F.cross_entropy(outputs['S'], batch['label']))
        torch.testing.assert_close(correction, F.cross_entropy(outputs['F'].detach() + outputs['R'], batch['label']))
        self.assertIsNone(torch.autograd.grad(correction, trainer.model.prompt_learner.ctx,
                                             allow_unused=True, retain_graph=True)[0])
        self.assertEqual(trainer.model.correction_identity['inference'], 'F+R')
        self.assertEqual(trainer.model.correction_identity['objective'],
                         'CE(S,y)+CE(stopgrad(F)+R,y)')

    def test_zero_residual_starts_at_fusion_without_changing_student_update(self):
        trainer = make_text(self.root / 'aligned', detail='fusion_aligned_attentive')
        original = make_text(self.root / 'original', detail='attentive')
        batch = {'img': torch.randn(3, 3, 32, 32), 'label': torch.tensor([0, 1, 2])}
        before = trainer.model(batch['img'])
        torch.testing.assert_close(before['R'], torch.zeros_like(before['R']), rtol=0, atol=0)
        torch.testing.assert_close(before['Z'], before['F'], rtol=0, atol=0)
        trainer.forward_backward(batch)
        original.forward_backward(batch)
        torch.testing.assert_close(trainer.model.prompt_learner.ctx,
                                   original.model.prompt_learner.ctx, rtol=0, atol=0)
        self.assertGreater(torch.count_nonzero(trainer.model.text_residual.up.weight).item(), 0)


if __name__ == '__main__':
    unittest.main()
