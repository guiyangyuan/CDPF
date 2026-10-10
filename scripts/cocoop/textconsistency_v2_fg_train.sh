#!/bin/bash

# CoCoOp + TextConsistency V2 + FG Training Script
# 参考AnchorOPT的训练参数：10 epochs, batch size 1
# 参考 docs/history/paper-v2-fg/结果.md 中的历史超参数

# 数据集路径
DATA=/root/lanyun-fs/DATA

# GPU设置
GPU=0

# 训练参数（参考AnchorOPT论文）
TRAINER=CoCoOp_TextConsistency_V2_FG
BATCH=1

# 配置文件（根据epochs动态选择）
CFG_EP10=vit_b16_c4_ep10_batch1
CFG_EP20=vit_b16_c4_ep10_batch1  # 使用相同的config，通过OPTIM.MAX_EPOCH覆盖

# 数据集配置（参考 docs/history/paper-v2-fg/结果.md + AnchorOPT论文设置）
# 格式: 数据集名称:实验ID:N_VIEWS:KD_W:KD_T:N_CTX:EPOCHS
declare -A DATASET_CONFIGS=(
    # 20 epochs数据集（大部分）
    ["caltech101"]="1:20:90.0:4.0:4:20"
    ["stanford_cars"]="101:24:0.6:1.9:4:20"
    ["oxford_pets"]="201:22:100.0:4.0:4:20"
    ["oxford_flowers"]="302:20:1.0:2.5:4:20"
    ["food101"]="401:20:100.0:1.5:4:20"
    ["fgvc_aircraft"]="501:22:2.0:2.0:4:20"
    ["dtd"]="701:24:2.5:2.5:4:20"
    ["eurosat"]="801:40:0.8:2.2:4:20"
    ["ucf101"]="901:40:80.0:4.2:4:20"
    # 10 epochs数据集（SUN397特殊，N_CTX=8）
    ["sun397"]="601:24:0.6:1.3:8:10"
)

# 函数：训练单个数据集
train_dataset() {
    local DATASET=$1
    local CONFIG=$2
    
    IFS=':' read -r EXP_ID N_VIEWS KD_W KD_T N_CTX EPOCHS <<< "$CONFIG"
    
    echo "========================================="
    echo "Training ${DATASET} (EXP${EXP_ID})"
    echo "N_VIEWS=${N_VIEWS}, KD_W=${KD_W}, KD_T=${KD_T}, N_CTX=${N_CTX}, EPOCHS=${EPOCHS}"
    echo "========================================="
    
    # 选择配置文件
    if [ "$EPOCHS" = "10" ]; then
        CFG=$CFG_EP10
    else
        CFG=$CFG_EP20
    fi
    
    # 训练Base类
    for SEED in 1 2 3
    do
        DIR=output/${DATASET}/CoCoOp_TextConsistency_V2_FG/vit_b16_c${N_CTX}_ep${EPOCHS}_batch${BATCH}/exp${EXP_ID}_v${N_VIEWS}_w${KD_W}_t${KD_T}/seed${SEED}
        
        if [ -d "$DIR" ]; then
            echo "Results exist at ${DIR}, skip training"
        else
            echo "Training base classes (seed ${SEED})..."
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
            DATASET.SUBSAMPLE_CLASSES base \
            OPTIM.MAX_EPOCH ${EPOCHS}
        fi
    done
    
    # 评估Novel类
    for SEED in 1 2 3
    do
        DIR=output/${DATASET}/CoCoOp_TextConsistency_V2_FG/vit_b16_c${N_CTX}_ep${EPOCHS}_batch${BATCH}/exp${EXP_ID}_v${N_VIEWS}_w${KD_W}_t${KD_T}/seed${SEED}
        
        echo "Evaluating novel classes (seed ${SEED})..."
        CUDA_VISIBLE_DEVICES=${GPU} python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/CoCoOp/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${DIR} \
        --load-epoch ${EPOCHS} \
        --eval-only \
        TRAINER.TEXTCONSISTENCY.N_CTX ${N_CTX} \
        TRAINER.TEXTCONSISTENCY.N_VIEWS ${N_VIEWS} \
        TRAINER.TEXTCONSISTENCY.KD_WEIGHT ${KD_W} \
        TRAINER.TEXTCONSISTENCY.KD_TEMP ${KD_T} \
        DATASET.NUM_SHOTS 16 \
        DATASET.SUBSAMPLE_CLASSES new
    done
    
    # 提取结果并生成summary.txt
    echo "Extracting results..."
    python Dassl.pytorch/tools/parse_test_res.py ${DIR%/seed${SEED}} --test-log
    
    # 生成summary.txt
    RESULT_DIR=${DIR%/seed${SEED}}
    if [ -f "${RESULT_DIR}/results.txt" ]; then
        echo "Generating summary.txt..."
        BASE_ACC=$(grep "base classes" "${RESULT_DIR}/results.txt" | awk '{print $NF}' | sed 's/%//')
        NOVEL_ACC=$(grep "new classes" "${RESULT_DIR}/results.txt" | awk '{print $NF}' | sed 's/%//')
        HM_ACC=$(grep "H" "${RESULT_DIR}/results.txt" | awk '{print $NF}' | sed 's/%//')
        
        cat > "${RESULT_DIR}/summary.txt" << EOF
Dataset: ${DATASET}
Experiment: EXP${EXP_ID}
Configuration: N_VIEWS=${N_VIEWS}, KD_W=${KD_W}, KD_T=${KD_T}, N_CTX=${N_CTX}, EPOCHS=${EPOCHS}

Results:
Base Accuracy: ${BASE_ACC}%
Novel Accuracy: ${NOVEL_ACC}%
Harmonic Mean: ${HM_ACC}%

Training: ${EPOCHS} epochs, batch size ${BATCH}
Seeds: 1, 2, 3
EOF
        echo "Summary saved to ${RESULT_DIR}/summary.txt"
        cat "${RESULT_DIR}/summary.txt"
    fi
    
    echo ""
}

# 主函数
main() {
    if [ $# -eq 0 ]; then
        echo "Training all datasets..."
        for DATASET in "${!DATASET_CONFIGS[@]}"; do
            train_dataset "$DATASET" "${DATASET_CONFIGS[$DATASET]}"
        done
    else
        DATASET=$1
        if [ -z "${DATASET_CONFIGS[$DATASET]}" ]; then
            echo "Error: Unknown dataset '$DATASET'"
            echo "Available datasets: ${!DATASET_CONFIGS[@]}"
            exit 1
        fi
        train_dataset "$DATASET" "${DATASET_CONFIGS[$DATASET]}"
    fi
}

# 运行
main "$@"
