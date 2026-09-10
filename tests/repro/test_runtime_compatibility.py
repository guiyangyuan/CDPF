import unittest
import torch
from dassl.config import get_cfg_default
from dassl.optim import build_lr_scheduler


class RuntimeCompatibility(unittest.TestCase):
    def test_warmup_scheduler_starts_low_then_restores_training_rate(self):
        cfg = get_cfg_default().OPTIM
        cfg.LR = 0.002; cfg.MAX_EPOCH = 20; cfg.LR_SCHEDULER = 'cosine'
        cfg.WARMUP_EPOCH = 1; cfg.WARMUP_TYPE = 'constant'; cfg.WARMUP_CONS_LR = 0.00001
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([parameter], lr=0.002)
        scheduler = build_lr_scheduler(optimizer, cfg)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.00001)
        parameter.sum().backward(); optimizer.step(); scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 0.002)

if __name__ == '__main__': unittest.main()
