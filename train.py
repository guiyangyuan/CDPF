import argparse
import os
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

from dassl.utils import setup_logger, set_random_seed
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop
import trainers.coop_textconsistency
import trainers.coop_textconsistency_avg
import trainers.coop_textconsistency_v2
import trainers.coop_textconsistency_v2_fg
import trainers.coop_textconsistency_correction
import trainers.coop_textconsistency_ablation

import trainers.cocoop
import trainers.cocoop_textconsistency_v2_fg

import trainers.zsclip
from trainers.visual_adapter import validate_visual_adapter_config

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed is not None:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    # cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.CTX_INIT = False  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    
    cfg.TRAINER.COOP.W = 2.0
    
    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for TextConsistency (Multi-View Text Consistency Learning)
    cfg.TRAINER.TEXTCONSISTENCY = CN()
    cfg.TRAINER.TEXTCONSISTENCY.N_CTX = 4
    cfg.TRAINER.TEXTCONSISTENCY.PREC = "fp32"
    cfg.TRAINER.TEXTCONSISTENCY.N_VIEWS = 5           # 每类使用的描述数量
    cfg.TRAINER.TEXTCONSISTENCY.CONSISTENCY_WEIGHT = 1.0  # 一致性损失权重
    cfg.TRAINER.TEXTCONSISTENCY.CONSISTENCY_TEMP = 1.0    # 一致性损失温度
    cfg.TRAINER.TEXTCONSISTENCY.KD_WEIGHT = 0.0       # KD损失权重
    cfg.TRAINER.TEXTCONSISTENCY.KD_TEMP = 2.0         # KD温度
    cfg.TRAINER.TEXTCONSISTENCY.KD_LOSS_MODE = "full_kl"
    cfg.TRAINER.TEXTCONSISTENCY.ADAPTIVE_CE_WEIGHT = 1.0  # 自适应分支分类损失权重
    cfg.TRAINER.TEXTCONSISTENCY.DIVERSITY_WEIGHT = 0.0    # V2: attention多样性正则化权重
    cfg.TRAINER.TEXTCONSISTENCY.CONTRASTIVE_WEIGHT = 0.0  # FG: 对比学习权重
    cfg.TRAINER.TEXTCONSISTENCY.N_GROUPS = 4          # FG: 层次化注意力分组数
    cfg.TRAINER.TEXTCONSISTENCY.ABLATION_GROUP = "full"  # Paper Table 4.3 group

    cfg.TRAINER.TEXTCONSISTENCY.AGGREGATION = "hierarchical"
    cfg.TRAINER.TEXTCONSISTENCY.VIEW_ORDER = "contiguous"
    cfg.TRAINER.TEXTCONSISTENCY.ORDER_SEED = 0
    cfg.TRAINER.TEXTCONSISTENCY.USE_FEATURE_ADAPTER = True
    cfg.TRAINER.TEXTCONSISTENCY.USE_VISUAL_ADAPTER = False
    cfg.TRAINER.TEXTCONSISTENCY.VISUAL_ADAPTER_DIM = 64
    cfg.TRAINER.TEXTCONSISTENCY.VISUAL_ADAPTER_SCALE = 0.2
    cfg.TRAINER.TEXTCONSISTENCY.CORRECTION_TEXT = "description"
    cfg.TRAINER.TEXTCONSISTENCY.INFERENCE_BRANCH = "ensemble"
    cfg.TRAINER.TEXTCONSISTENCY.ENSEMBLE_ALPHA = 0.5
    cfg.TRAINER.TEXTCONSISTENCY.DEVICE = "auto"
    cfg.TRAINER.TEXTCONSISTENCY.CLIP_PATH = "clip/ViT-B-16.pt"
    cfg.TRAINER.TEXTCONSISTENCY.TEXT_BATCH_SIZE = 32
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    validate_visual_adapter_config(cfg)
    kd_mode_trainers = {
        "CoOp_TextConsistency_V2_FG",
        "CoCoOp_TextConsistency_V2_FG",
        "CoOp_TextConsistency_Ablation",
    }
    if (cfg.TRAINER.TEXTCONSISTENCY.KD_LOSS_MODE != "full_kl"
            and cfg.TRAINER.NAME not in kd_mode_trainers):
        raise ValueError("A non-default KD loss mode is only supported by validated TextConsistency trainers")
    legacy_trainers = {"CoOp_TextConsistency", "CoOp_TextConsistency_Avg", "CoOp_TextConsistency_V2"}
    if cfg.TRAINER.NAME in legacy_trainers:
        raise ValueError("Legacy trainer is not validated; use CoOp_TextConsistency_V2_FG with AGGREGATION mean/flat/hierarchical")
    if "TextConsistency" in cfg.TRAINER.NAME:
        directory = Path(cfg.OUTPUT_DIR)
        if cfg.RESUME:
            raise ValueError("Use a fresh output for the corrected protocol; legacy resume is not supported")
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Output already contains files; choose a fresh directory: {directory}")
        torch.use_deterministic_algorithms(True)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test(is_final=True)
        return

    if not args.no_train:
        trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)
