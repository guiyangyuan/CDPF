import unittest
from pathlib import Path


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
    def test_train_py_exposes_only_supported_public_config_blocks(self):
        source = TRAIN.read_text(encoding="utf-8")
        for token in REMOVED_CONFIG_TOKENS:
            with self.subTest(token=token):
                self.assertNotIn(token, source)

        for token in REQUIRED_CONFIG_TOKENS:
            with self.subTest(token=token):
                self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
