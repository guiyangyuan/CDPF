"""Short checks for optimizer separation, student parity, and full state backup."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import torch
from .test_text_residual import make_text


class SplitOptimizerSmoke(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.capture = contextlib.redirect_stdout(io.StringIO()); self.capture.__enter__()
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup(); self.capture.__exit__(None, None, None)

    def test_two_steps_preserve_student_and_only_schedule_sgd(self):
        baseline = make_text(self.root / 'sgd')
        models = [make_text(self.root / mode, mode, adamw=True)
                  for mode in ('classname', 'description', 'contrast')]
        batch = {'img': torch.randn(3, 3, 32, 32), 'label': torch.tensor([0, 1, 2])}
        original = baseline.model(batch['img'])
        for trainer in models:
            outputs = trainer.model(batch['img'])
            torch.testing.assert_close(outputs['S'], original['S'], rtol=0, atol=0)
            torch.testing.assert_close(outputs['Z'], original['S'], rtol=0, atol=0)
            opt = trainer.optim
            self.assertIs(trainer.sched.optimizer, opt.student)
            self.assertEqual(trainer.repro_signature['format'], 5)
            self.assertEqual(trainer.run_provenance['trainable_parameters'], baseline.run_provenance['trainable_parameters'])
        for _ in range(2):
            baseline.num_batches = 1; baseline.forward_backward(batch)
            for trainer in models:
                trainer.num_batches = 1
                result = trainer.forward_backward(batch)
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in result.values()))
                torch.testing.assert_close(trainer.model.prompt_learner.ctx,
                                           baseline.model.prompt_learner.ctx, rtol=0, atol=0)
                self.assertEqual(trainer.optim.residual.param_groups[0]['lr'], .001)
                self.assertEqual(trainer.optim.student.param_groups[0]['lr'], baseline.optim.param_groups[0]['lr'])
                self.assertTrue(all(p.grad is None for p in trainer.model.image_encoder.parameters()))
                self.assertGreater(torch.count_nonzero(trainer.model.text_residual.up.weight).item(), 0)

    def test_checkpoint_restores_both_optimizers_for_same_next_update(self):
        first = make_text(self.root / 'first', adamw=True)
        batch = {'img': torch.randn(3, 3, 32, 32), 'label': torch.tensor([0, 1, 2])}
        first.num_batches = 1; first.forward_backward(batch)
        first.save_model(0, self.root / 'first')
        saved = torch.load(self.root / 'first/textconsistency/model.pth.tar-1', weights_only=False)
        self.assertTrue(saved['optimizer']['student']['state'])
        self.assertTrue(saved['optimizer']['residual']['state'])
        same = make_text(self.root / 'same', adamw=True)
        same.load_model(self.root / 'first', 1)
        same.optim.load_state_dict(saved['optimizer']); same.sched.load_state_dict(saved['scheduler'])
        for name, value in first.model(batch['img']).items():
            torch.testing.assert_close(value, same.model(batch['img'])[name], rtol=0, atol=0)
        same.num_batches = 1
        first.forward_backward(batch); same.forward_backward(batch)
        for name, value in first.trainable_modules.state_dict().items():
            torch.testing.assert_close(value, same.trainable_modules.state_dict()[name], rtol=0, atol=0)
        for state in same.optim.residual.state.values():
            self.assertEqual(state['step'].item(), 2)
        self.assertEqual(same.optim.residual.param_groups[0]['lr'], .001)
        wrong = make_text(self.root / 'old_sgd')
        with self.assertRaisesRegex(ValueError, 'signature mismatch'):
            wrong.load_model(self.root / 'first', 1)


if __name__ == '__main__':
    unittest.main()
