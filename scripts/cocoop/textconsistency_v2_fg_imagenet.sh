#!/bin/bash

# CoCoOp + TextConsistency V2 + FG - ImageNet Training Script
# 用于在ImageNet上训练模型，然后可以用于cross-dataset和domain generalization实验

# 数据集路径
DATA=/root/lanyun-fs/DATA

# GPU设置
GPU=0

# 训练参数
TRAINER=CoCoOp_TextConsistency_V2_FG
DATASET=imagenet
BATCH=1

# ImageNet配置（参考AnchorOPT论文）
# ImageNet: N_CTX=6, EPOCHS=20, CE=100.0, KD=1000.0, KD_Temp=1.0
N_CTX=6
EPOCHS=20
N_VIEWS=20
KD_W=1000.0
KD_T=1.0

# 实验ID
EXP_ID=CoCoOp_TC_V2_FG_001

# 配置文件
CFG=vit_b16_c4_ep10_batch1

echo "========================================="
echo "🚀 Training ImageNet with CoCoOp + TextConsistency V2 + FG"
echo "========================================="
echo "Configuration:"
echo "  N_CTX: ${N_CTX}"
echo "  EPOCHS: ${EPOCHS}"
echo "  N_VIEWS: ${N_VIEWS}"
echo "  KD_WEIGHT: ${KD_W}"
echo "  KD_TEMP: ${KD_T}"
echo "  BATCH_SIZE: ${BATCH}"
echo "========================================="
echo ""

# 训练3个seeds
for SEED in 1 2 3
do
    DIR=/root/lanyun-fs/output/${DATASET}/CoCoOp_TextConsistency_V2_FG/vit_b16_c${N_CTX}_ep${EPOCHS}_batch${BATCH}/${EXP_ID}_v${N_VIEWS}_w${KD_W}_t${KD_T}/seed${SEED}
    
    if [ -d "$DIR" ]; then
        echo "⏭️  Results exist at ${DIR}, skip training seed ${SEED}"
        continue
    fi
    
    echo "🚀 Training ImageNet with CoCoOp (seed ${SEED})..."
    
    CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoCoOp/${CFG}.yaml \
    --output-dir ${DIR} \
    TRAINER.TEXTCONSISTENCY.N_CTX ${N_CTX} \
    TRAINER.TEXTCONSISTENCY.N_VIEWS ${N_VIEWS} \
    TRAINER.TEXTCONSISTENCY.KD_WEIGHT ${KD_W} \
    TRAINER.TEXTCONSISTENCY.KD_TEMP ${KD_T} \
    DATASET.NUM_SHOTS 16 \
    DATASET.SUBSAMPLE_CLASSES all \
    OPTIM.MAX_EPOCH ${EPOCHS}
    
    echo "✅ Training completed for seed ${SEED}"
    echo ""
done

echo "========================================="
echo "📊 ImageNet Training Summary"
echo "========================================="

# 提取所有seeds的结果
RESULT_DIR=/root/lanyun-fs/output/${DATASET}/CoCoOp_TextConsistency_V2_FG/vit_b16_c${N_CTX}_ep${EPOCHS}_batch${BATCH}/${EXP_ID}_v${N_VIEWS}_w${KD_W}_t${KD_T}

echo "Results saved in: ${RESULT_DIR}"
echo ""
echo "Extracting accuracies from all seeds..."

for SEED in 1 2 3
do
    LOG_FILE="${RESULT_DIR}/seed${SEED}/log.txt"
    if [ -f "$LOG_FILE" ]; then
        ACC=$(tail -20 "$LOG_FILE" | grep "accuracy:" | tail -1 | grep -oP '\d+\.\d+')
        if [ -n "$ACC" ]; then
            echo "  Seed ${SEED}: ${ACC}%"
        else
            echo "  Seed ${SEED}: Failed to extract accuracy"
        fi
    else
        echo "  Seed ${SEED}: Log file not found"
    fi
done

echo ""
echo "========================================="
echo "✅ ImageNet training completed!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Cross-dataset evaluation:"
echo "   bash scripts/cocoop/textconsistency_v2_fg_xd_eval.sh caltech101 1"
echo ""
echo "2. Domain generalization evaluation:"
echo "   bash scripts/cocoop/textconsistency_v2_fg_xd_eval.sh imagenet_a 1"
echo ""
