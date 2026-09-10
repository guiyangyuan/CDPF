#!/bin/bash
# ============================================================================
# 一次性训练所有数据集（最佳配置）
# ============================================================================

DATA=/root/lanyun-fs/DATA
CFG=vit_b16
SHOTS=16
TRAINER=CoOp_TextConsistency_V2

# 训练函数
train_dataset() {
    local DATASET=$1
    local NCTX=$2
    local EPO=$3
    local N_VIEWS=$4
    local KD_W=$5
    local KD_T=$6
    local EXP_ID=$7
    
    echo "============================================================================"
    echo ">>> Training ${DATASET} (EXP${EXP_ID})"
    echo "Config: N_VIEWS=${N_VIEWS}, KD_W=${KD_W}, KD_T=${KD_T}, NCTX=${NCTX}, EPO=${EPO}"
    echo "============================================================================"
    
    EXP_DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/EXP${EXP_ID}_V${N_VIEWS}_KD${KD_W}
    
    for SEED in 1 2 3; do
        DIR=${EXP_DIR}/seed${SEED}
        
        # Train
        CUDA_VISIBLE_DEVICES=0 python train.py --root ${DATA} --seed ${SEED} \
            --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/CoOp/${CFG}.yaml --output-dir ${DIR} \
            TRAINER.TEXTCONSISTENCY.N_CTX ${NCTX} \
            TRAINER.TEXTCONSISTENCY.N_VIEWS ${N_VIEWS} \
            TRAINER.TEXTCONSISTENCY.KD_WEIGHT ${KD_W} \
            TRAINER.TEXTCONSISTENCY.KD_TEMP ${KD_T} \
            TRAINER.TEXTCONSISTENCY.PREC fp32 \
            DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base OPTIM.MAX_EPOCH ${EPO}
        
        # Test base
        CUDA_VISIBLE_DEVICES=0 python train.py --root ${DATA} --seed ${SEED} \
            --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/CoOp/${CFG}.yaml --output-dir ${DIR}/evaluation \
            --model-dir ${DIR} --eval-only \
            TRAINER.TEXTCONSISTENCY.N_CTX ${NCTX} \
            TRAINER.TEXTCONSISTENCY.N_VIEWS ${N_VIEWS} \
            TRAINER.TEXTCONSISTENCY.KD_WEIGHT ${KD_W} \
            TRAINER.TEXTCONSISTENCY.KD_TEMP ${KD_T} \
            TRAINER.TEXTCONSISTENCY.PREC fp32 \
            DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base
        
        # Test novel
        CUDA_VISIBLE_DEVICES=0 python train.py --root ${DATA} --seed ${SEED} \
            --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/CoOp/${CFG}.yaml --output-dir ${DIR}/evaluation \
            --model-dir ${DIR} --eval-only \
            TRAINER.TEXTCONSISTENCY.N_CTX ${NCTX} \
            TRAINER.TEXTCONSISTENCY.N_VIEWS ${N_VIEWS} \
            TRAINER.TEXTCONSISTENCY.KD_WEIGHT ${KD_W} \
            TRAINER.TEXTCONSISTENCY.KD_TEMP ${KD_T} \
            TRAINER.TEXTCONSISTENCY.PREC fp32 \
            DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES new
    done
    
    # 计算平均分数
    BASE_SCORES=""
    NOVEL_SCORES=""
    for SEED in 1 2 3; do
        BASE_FILE="${EXP_DIR}/seed${SEED}/evaluation/log.txt"
        NOVEL_FILE=$(ls ${EXP_DIR}/seed${SEED}/evaluation/log.txt-* 2>/dev/null | head -1)
        BASE_ACC=$(grep "accuracy" "$BASE_FILE" 2>/dev/null | head -1 | grep -oP '\d+\.\d+')
        NOVEL_ACC=$(grep "accuracy" "$NOVEL_FILE" 2>/dev/null | head -1 | grep -oP '\d+\.\d+')
        BASE_SCORES="${BASE_SCORES} ${BASE_ACC}"
        NOVEL_SCORES="${NOVEL_SCORES} ${NOVEL_ACC}"
    done
    
    AVG_BASE=$(echo $BASE_SCORES | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; printf "%.2f", sum/NF}')
    AVG_NOVEL=$(echo $NOVEL_SCORES | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; printf "%.2f", sum/NF}')
    HM=$(echo "$AVG_BASE $AVG_NOVEL" | awk '{printf "%.2f", 2*$1*$2/($1+$2)}')
    
    echo "HM: ${HM}%" > ${EXP_DIR}/summary.txt
    echo "Result: Base=${AVG_BASE}%, Novel=${AVG_NOVEL}%, HM=${HM}%"
    echo ""
}

echo "============================================================================"
echo "开始训练所有数据集（最佳配置）"
echo "============================================================================"
echo ""

# 训练所有数据集
train_dataset "caltech101" 4 50 20 100.0 4.0 1
train_dataset "stanford_cars" 4 50 20 0.5 2.0 101
train_dataset "oxford_pets" 4 50 20 100.0 4.0 201
train_dataset "oxford_flowers" 4 50 20 0.5 2.0 302
train_dataset "food101" 4 20 20 100.0 1.5 401
train_dataset "fgvc_aircraft" 4 50 20 2.0 2.0 501
train_dataset "sun397" 8 20 20 0.5 1.2 601
train_dataset "dtd" 4 50 25 2.0 2.0 701
train_dataset "eurosat" 4 100 20 0.3 2.0 801
train_dataset "ucf101" 4 50 25 60.0 5.0 901

echo "============================================================================"
echo "所有数据集训练完成！"
echo "============================================================================"
