#!/bin/bash

# TextConsistency V2 + Fine-Grained
# 使用方法: bash scripts/coop/textconsistency_v2_fg_test.sh dtd 701

DATA=/root/lanyun-fs/DATA
TRAINER=CoOp_TextConsistency_V2_FG
CFG=vit_b16
SHOTS=16

DATASET=$1
EXP=$2

# 数据集基础参数
case $DATASET in
    "dtd") NCTX=4; EPO=100 ;;
    "ucf101") NCTX=8; EPO=100 ;;
    "fgvc_aircraft") NCTX=4; EPO=100 ;;
    "oxford_flowers") NCTX=4; EPO=100 ;;
    *) echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

# Fine-Grained增强配置
case $EXP in
    "701") N_VIEWS=28; KD_W=3.0; KD_T=3.0; CONTRAST_W=0.15; DIV_W=0.08; N_GROUPS=4 ;;
    "901") N_VIEWS=40; KD_W=85.0; KD_T=4.5; CONTRAST_W=0.1; DIV_W=0.05; N_GROUPS=5 ;;
    "902") N_VIEWS=40; KD_W=85.0; KD_T=4.5; CONTRAST_W=0.2; DIV_W=0.03; N_GROUPS=5 ;;
    "903") N_VIEWS=40; KD_W=85.0; KD_T=4.5; CONTRAST_W=0.3; DIV_W=0.02; N_GROUPS=5 ;;
    "501") N_VIEWS=24; KD_W=2.5; KD_T=2.5; CONTRAST_W=0.1; DIV_W=0.05; N_GROUPS=4 ;;
    "302") N_VIEWS=24; KD_W=1.5; KD_T=2.5; CONTRAST_W=0.1; DIV_W=0.05; N_GROUPS=4 ;;
    "303") N_VIEWS=24; KD_W=1.5; KD_T=2.5; CONTRAST_W=0.05; DIV_W=0.02; N_GROUPS=3 ;;
    "304") N_VIEWS=24; KD_W=1.5; KD_T=2.5; CONTRAST_W=0.02; DIV_W=0.01; N_GROUPS=3 ;;
    "305") N_VIEWS=26; KD_W=2.0; KD_T=2.5; CONTRAST_W=0.0; DIV_W=0.0; N_GROUPS=3 ;;
    *) echo "Unknown exp: $EXP"; exit 1 ;;
esac

echo "============================================================================"
echo "TextConsistency V2 + Fine-Grained"
echo "============================================================================"
echo "数据集: $DATASET | 实验: EXP$EXP"
echo "N_VIEWS=$N_VIEWS KD_W=$KD_W KD_T=$KD_T CONTRAST_W=$CONTRAST_W DIV_W=$DIV_W N_GROUPS=$N_GROUPS"
echo "============================================================================"

EXP_DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/EXP${EXP}_FG

for SEED in 1 2 3; do
  DIR=${EXP_DIR}/seed${SEED}
  
  echo "训练 Seed ${SEED}/3..."
  
  CUDA_VISIBLE_DEVICES=0 python train.py --root ${DATA} --seed ${SEED} \
    --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml --output-dir ${DIR} \
    TRAINER.TEXTCONSISTENCY.N_CTX ${NCTX} \
    TRAINER.TEXTCONSISTENCY.N_VIEWS ${N_VIEWS} \
    TRAINER.TEXTCONSISTENCY.KD_WEIGHT ${KD_W} \
    TRAINER.TEXTCONSISTENCY.KD_TEMP ${KD_T} \
    TRAINER.TEXTCONSISTENCY.CONTRASTIVE_WEIGHT ${CONTRAST_W} \
    TRAINER.TEXTCONSISTENCY.DIVERSITY_WEIGHT ${DIV_W} \
    TRAINER.TEXTCONSISTENCY.N_GROUPS ${N_GROUPS} \
    TRAINER.TEXTCONSISTENCY.PREC fp32 \
    DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base OPTIM.MAX_EPOCH ${EPO}

  CUDA_VISIBLE_DEVICES=0 python train.py --root ${DATA} --seed ${SEED} \
    --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml --output-dir ${DIR}/evaluation \
    --model-dir ${DIR} --eval-only \
    TRAINER.TEXTCONSISTENCY.N_CTX ${NCTX} \
    TRAINER.TEXTCONSISTENCY.N_VIEWS ${N_VIEWS} \
    TRAINER.TEXTCONSISTENCY.KD_WEIGHT ${KD_W} \
    TRAINER.TEXTCONSISTENCY.KD_TEMP ${KD_T} \
    TRAINER.TEXTCONSISTENCY.CONTRASTIVE_WEIGHT ${CONTRAST_W} \
    TRAINER.TEXTCONSISTENCY.DIVERSITY_WEIGHT ${DIV_W} \
    TRAINER.TEXTCONSISTENCY.N_GROUPS ${N_GROUPS} \
    TRAINER.TEXTCONSISTENCY.PREC fp32 \
    DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base

  CUDA_VISIBLE_DEVICES=0 python train.py --root ${DATA} --seed ${SEED} \
    --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml --output-dir ${DIR}/evaluation \
    --model-dir ${DIR} --eval-only \
    TRAINER.TEXTCONSISTENCY.N_CTX ${NCTX} \
    TRAINER.TEXTCONSISTENCY.N_VIEWS ${N_VIEWS} \
    TRAINER.TEXTCONSISTENCY.KD_WEIGHT ${KD_W} \
    TRAINER.TEXTCONSISTENCY.KD_TEMP ${KD_T} \
    TRAINER.TEXTCONSISTENCY.CONTRASTIVE_WEIGHT ${CONTRAST_W} \
    TRAINER.TEXTCONSISTENCY.DIVERSITY_WEIGHT ${DIV_W} \
    TRAINER.TEXTCONSISTENCY.N_GROUPS ${N_GROUPS} \
    TRAINER.TEXTCONSISTENCY.PREC fp32 \
    DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES new
done

# 计算平均
BASE_SCORES=""
NOVEL_SCORES=""
for SEED in 1 2 3; do
  BASE_FILE=$(ls ${EXP_DIR}/seed${SEED}/evaluation/log.txt 2>/dev/null)
  NOVEL_FILE=$(ls ${EXP_DIR}/seed${SEED}/evaluation/log.txt-* 2>/dev/null | head -1)
  BASE_ACC=$(grep "accuracy" "$BASE_FILE" 2>/dev/null | head -1 | grep -oP '\d+\.\d+')
  NOVEL_ACC=$(grep "accuracy" "$NOVEL_FILE" 2>/dev/null | head -1 | grep -oP '\d+\.\d+')
  BASE_SCORES="${BASE_SCORES} ${BASE_ACC}"
  NOVEL_SCORES="${NOVEL_SCORES} ${NOVEL_ACC}"
done

AVG_BASE=$(echo $BASE_SCORES | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; printf "%.2f", sum/NF}')
AVG_NOVEL=$(echo $NOVEL_SCORES | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; printf "%.2f", sum/NF}')
HM=$(echo "$AVG_BASE $AVG_NOVEL" | awk '{printf "%.2f", 2*$1*$2/($1+$2)}')

echo "============================================================================"
echo "Base: ${AVG_BASE}% | Novel: ${AVG_NOVEL}% | HM: ${HM}%"
echo "============================================================================"

echo "========================================" > ${EXP_DIR}/summary.txt
echo "EXP${EXP}_FG on ${DATASET}" >> ${EXP_DIR}/summary.txt
echo "----------------------------------------" >> ${EXP_DIR}/summary.txt
echo "Avg Base:  ${AVG_BASE}%" >> ${EXP_DIR}/summary.txt
echo "Avg Novel: ${AVG_NOVEL}%" >> ${EXP_DIR}/summary.txt
echo "HM:        ${HM}%" >> ${EXP_DIR}/summary.txt
echo "========================================" >> ${EXP_DIR}/summary.txt
