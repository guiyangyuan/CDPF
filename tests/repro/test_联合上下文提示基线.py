"""CoCoOp投稿基线的核心行为与冻结计划检查。"""

import importlib.util
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
from trainers.cocoop import CoCoOp


ROOT = Path(__file__).resolve().parents[2]


def tiny_clip(_cfg):
    torch.manual_seed(31)
    model = CLIP(
        32, 32, 1, 64, 16, 77, 49408, 64, 1, 1,
        {"trainer": "CoCoOp", "vision_depth": 0, "language_depth": 0, "vision_ctx": 0, "language_ctx": 0},
    )
    return model.float()


def make_trainer(output: Path) -> CoCoOp:
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.SEED = 1
    cfg.USE_CUDA = False
    cfg.DATASET.ROOT = str(output)
    cfg.DATASET.NAME = "Caltech101"
    cfg.DATASET.SUBSAMPLE_CLASSES = "base"
    cfg.DATASET.NUM_SHOTS = 16
    cfg.OUTPUT_DIR = str(output)
    cfg.INPUT.SIZE = (32, 32)
    cfg.TRAINER.NAME = "CoCoOp"
    cfg.TRAINER.COCOOP.N_CTX = 4
    cfg.TRAINER.COCOOP.CTX_INIT = "a photo of a"
    cfg.TRAINER.COCOOP.PREC = "fp32"
    cfg.MODEL.BACKBONE.NAME = "ViT-B/16"
    cfg.OPTIM.NAME = "sgd"
    cfg.OPTIM.LR = 0.002
    cfg.OPTIM.MAX_EPOCH = 10
    cfg.OPTIM.WARMUP_EPOCH = 0
    trainer = CoCoOp.__new__(CoCoOp)
    TrainerBase.__init__(trainer)
    trainer.cfg = cfg
    trainer.dm = SimpleNamespace(dataset=SimpleNamespace(classnames=["red object", "blue object", "green object"]))
    trainer.check_cfg(cfg)
    with patch("trainers.cocoop.load_clip_to_cpu", side_effect=tiny_clip):
        trainer.build_model()
    trainer.batch_idx = 0
    trainer.num_batches = 1
    return trainer


class CoCoOp基线检查(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_只更新上下文与条件网络并可从第十轮重载(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = make_trainer(root / "第一次")
            trainable = {name for name, value in first.model.named_parameters() if value.requires_grad}
            self.assertTrue(trainable)
            self.assertTrue(all(name.startswith("prompt_learner.") for name in trainable))
            self.assertIn("prompt_learner.ctx", trainable)
            self.assertTrue(any(name.startswith("prompt_learner.meta_net.") for name in trainable))
            before = {name: value.detach().clone() for name, value in first.model.prompt_learner.named_parameters()}
            images = torch.randn(1, 3, 32, 32)
            result = first.forward_backward({"img": images, "label": torch.tensor([1])})
            self.assertGreater(result["loss"], 0.0)
            self.assertTrue(any(not torch.equal(before[name], value) for name, value in first.model.prompt_learner.named_parameters()))
            self.assertTrue(all(not value.requires_grad and value.grad is None for value in first.model.image_encoder.parameters()))
            self.assertTrue(all(not value.requires_grad and value.grad is None for value in first.model.text_encoder.parameters()))

            first.save_model(9, root / "检查点")
            checkpoint = root / "检查点/prompt_learner/model.pth.tar-10"
            self.assertTrue(checkpoint.is_file())
            second = make_trainer(root / "第二次")
            second.load_model(root / "检查点", epoch=10)
            first.model.eval()
            second.model.eval()
            torch.testing.assert_close(first.model(images), second.model(images), rtol=0, atol=0)

    def test_计划固定五数据集三种子与同一检查点(self):
        module_path = ROOT / "scripts/repro/运行联合上下文提示基线.py"
        spec = importlib.util.spec_from_file_location("联合上下文提示计划", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            temp = Path(temporary)
            plan = module.生成计划(ROOT, temp / "数据", temp / "输出", Path("/usr/bin/python3"))
        self.assertEqual(plan["协议"], "投稿CoCoOp五数据集标准基线-v1")
        self.assertEqual(plan["运行数"], 15)
        self.assertEqual(plan["命令数"], 45)
        self.assertEqual(sum(item["阶段"] == "核心三数据集" for item in plan["运行"]), 9)
        self.assertEqual(sum(item["阶段"] == "扩展两数据集" for item in plan["运行"]), 6)
        self.assertEqual(plan["固定协议"]["seeds"], [1, 2, 3])
        for item in plan["运行"]:
            self.assertTrue(item["检查点"].endswith("prompt_learner/model.pth.tar-10"))
            self.assertIn("a photo of a", item["命令"]["训练"])
            self.assertNotIn("--eval-only", item["命令"]["训练"])
            for split in ("base", "new"):
                command = item["命令"][split]
                self.assertIn("--eval-only", command)
                self.assertEqual(command[command.index("--load-epoch") + 1], "10")
                self.assertEqual(command[-1], split)


if __name__ == "__main__":
    unittest.main()
