#!/bin/bash
# CoOp Baseline 测试
DATA=/root/lanyun-fs/DATA
TRAINER=CoOp
CFG=vit_b16
SHOTS=16

DATASET=$1

# 数据集参数 (与AnchorOPT一致)
case $DATASET in
    "caltech101") NCTX=4; EPO=100 ;;
    "oxford_pets") NCTX=4; EPO=100 ;;
    "stanford_cars") NCTX=8; EPO=100 ;;
    "oxford_flowers") NCTX=4; EPO=100 ;;
    "food101") NCTX=4; EPO=20 ;;
    "fgvc_aircraft") NCTX=4; EPO=100 ;;
    "sun397") NCTX=8; EPO=20 ;;
    "dtd") NCTX=4; EPO=100 ;;
    "eurosat") NCTX=4; EPO=100 ;;
    "ucf101") NCTX=8; EPO=100 ;;
    *) echo "Unknown dataset"; exit 1 ;;
esac

echo "=== CoOp Baseline on $DATASET ==="

for SEED in 1 2 3; do
  DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/baseline/seed${SEED}
  
  # Train
  CUDA_VISIBLE_DEVICES=0 python train.py --root ${DATA} --seed ${SEED} \
    --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml --output-dir ${DIR} \
    TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC False TRAINER.COOP.CLASS_TOKEN_POSITION end \
    DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base OPTIM.MAX_EPOCH ${EPO}

  # Test base
  CUDA_VISIBLE_DEVICES=0 python train.py --root ${DATA} --seed ${SEED} \
    --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml --output-dir ${DIR}/evaluation \
    --model-dir ${DIR} --eval-only \
    TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC False TRAINER.COOP.CLASS_TOKEN_POSITION end \
    DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base

  # Test novel
  CUDA_VISIBLE_DEVICES=0 python train.py --root ${DATA} --seed ${SEED} \
    --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml --output-dir ${DIR}/evaluation \
    --model-dir ${DIR} --eval-only \
    TRAINER.COOP.N_CTX ${NCTX} TRAINER.COOP.CSC False TRAINER.COOP.CLASS_TOKEN_POSITION end \
    DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES new
done
echo "=== Done ==="
