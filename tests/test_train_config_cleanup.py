import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dassl.config import get_cfg_default
import train


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train.py"

REMOVED_CONFIG_TOKENS = (
    "cfg.TRAINER.MAPLE",
    "cfg.TRAINER.PROMPTSRC",
    "cfg.TRAINER.ATPROMPT",
    "cfg.TRAINER.ANCHOROPT",
    "cfg.TRAINER.ADAPTIVEANCHOR",
    "cfg.TRAINER.ANCHORSELECT",
    "cfg.TRAINER.ANCHORCL",
    "cfg.TRAINER.TEXTMIX",
    "cfg.TRAINER.DYNAMICANCHOR",
    "cfg.TRAINER.IVLP",
    "cfg.TRAINER.LINEAR_PROBE",
    "cfg.TRAINER.FILM",
    "cfg.TRAINER.NAMES_TO_UPDATE",
    "choose_attribute_for_atprompt",
)


REQUIRED_CONFIG_TOKENS = (
    "cfg.TRAINER.COOP",
    "cfg.TRAINER.COCOOP",
    "cfg.TRAINER.TEXTCONSISTENCY",
    "cfg.TRAINER.TEXTCONSISTENCY.ABLATION_GROUP",
    "cfg.DATASET.SUBSAMPLE_CLASSES",
)


class TrainConfigCleanupTest(unittest.TestCase):
    def test_kd_loss_mode_defaults_to_historical_full_kl(self):
        cfg = get_cfg_default(); train.extend_cfg(cfg)
        self.assertEqual(cfg.TRAINER.TEXTCONSISTENCY.KD_LOSS_MODE, 'full_kl')

    def test_train_py_exposes_only_supported_public_config_blocks(self):
        source = TRAIN.read_text(encoding="utf-8")
        for token in REMOVED_CONFIG_TOKENS:
            with self.subTest(token=token):
                self.assertNotIn(token, source)

        for token in REQUIRED_CONFIG_TOKENS:
            with self.subTest(token=token):
                self.assertIn(token, source)

    def test_nondefault_kd_mode_is_rejected_for_unsupported_trainer(self):
        cfg = get_cfg_default(); train.extend_cfg(cfg)
        cfg.TRAINER.NAME = 'CoOp'
        cfg.TRAINER.TEXTCONSISTENCY.KD_LOSS_MODE = 'weighted_nckd'
        args = SimpleNamespace(eval_only=False, no_train=True, model_dir='', load_epoch=None)
        with patch('train.setup_cfg', return_value=cfg), patch('train.build_trainer') as build:
            with self.assertRaisesRegex(ValueError, 'only supported'):
                train.main(args)
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
