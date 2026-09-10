#!/bin/bash
# ============================================================================
# TextConsistency V2 - 最佳配置训练脚本
# ============================================================================
# 本脚本包含所有数据集经过多轮调优后的最佳参数配置
# 使用方法: bash scripts/coop/textconsistency_v2_tune.sh <dataset> <exp_id>
# 示例: bash scripts/coop/textconsistency_v2_tune.sh caltech101 1
# ============================================================================

DATA=/root/lanyun-fs/DATA
TRAINER=CoOp_TextConsistency_V2
CFG=vit_b16
SHOTS=16

DATASET=$1
EXP=$2

# 数据集基础参数配置
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
    *) echo "Unknown dataset: $DATASET"; exit 1 ;;
esac

# ============================================================================
# 最佳配置参数 (经过多轮调优验证)
# ============================================================================
case $EXP in
    # ------------------------------------------------------------------------
    # Caltech101 - 最佳配置
    # ------------------------------------------------------------------------
    # HM: 96.12% | Base: 97.52% | Novel: 94.76%
    # 状态: ✅ 已达标
    "1") N_VIEWS=20; KD_W=100.0; KD_T=4.0; DIV_W=0.0 ;;
    
    # ------------------------------------------------------------------------
    # Stanford Cars - 最佳配置
    # ------------------------------------------------------------------------
    # HM: 74.89% | Base: 79.47% | Novel: 70.78%
    # 状态: ✅ 已达标
    "101") N_VIEWS=20; KD_W=0.5; KD_T=2.0; DIV_W=0.0 ;;
    
    # ------------------------------------------------------------------------
    # Oxford Pets - 最佳配置
    # ------------------------------------------------------------------------
    # HM: 96.37% | Base: 97.52% | Novel: 95.25%
    # 状态: ✅ 已达标
    "201") N_VIEWS=20; KD_W=100.0; KD_T=4.0; DIV_W=0.0 ;;
    
    # ------------------------------------------------------------------------
    # Oxford Flowers - 最佳配置 (Baseline)
    # ------------------------------------------------------------------------
    # HM: 82.92% | 目标: 84.27% | 差距: -1.35%
    # 状态: ⚠️ 接近目标，难以突破
    # 配置: 中等KD权重，标准温度，20视角
    "302") N_VIEWS=20; KD_W=0.5; KD_T=2.0; DIV_W=0.0 ;;
    
    # ------------------------------------------------------------------------
    # Food101 - 最佳配置
    # ------------------------------------------------------------------------
    # HM: 90.83% | 目标: 91.11% (AnchorOPT) | 差距: -0.28%
    # 状态: ⚠️ 非常接近目标
    # 配置: 高KD权重，低温度，20视角
    "401") N_VIEWS=20; KD_W=100.0; KD_T=1.5; DIV_W=0.0 ;;
    
    # ------------------------------------------------------------------------
    # FGVC Aircraft - 最佳配置
    # ------------------------------------------------------------------------
    # HM: 34.25% | Base: 42.22% | Novel: 28.56%
    # 状态: ✅ 已达标
    # 配置: 低KD权重，标准温度，20视角
    "501") N_VIEWS=20; KD_W=2.0; KD_T=2.0; DIV_W=0.0 ;;
    
    # ------------------------------------------------------------------------
    # SUN397 - 最佳配置
    # ------------------------------------------------------------------------
    # HM: 79.03% | AnchorOPT: 78.71% | 提升: +0.32%
    # 状态: ✅ 超越AnchorOPT
    # 配置: 中等KD权重，低温度，20视角
    "601") N_VIEWS=20; KD_W=0.5; KD_T=1.2; DIV_W=0.0 ;;
    
    # ------------------------------------------------------------------------
    # DTD - 最佳配置 (D14)
    # ------------------------------------------------------------------------
    # HM: 65.22% | Base: 72.57% | Novel: 59.22%
    # 目标: 65.93% (AnchorOPT) | 差距: -0.71%
    # 状态: ⚠️ 接近目标，参数已优化至局部最优
    # 配置: 低KD权重，标准温度，25视角（增加视角数提升性能）
    "701") N_VIEWS=25; KD_W=2.0; KD_T=2.0; DIV_W=0.0 ;;
    
    # ------------------------------------------------------------------------
    # EuroSAT - 最佳配置 (E10)
    # ------------------------------------------------------------------------
    # HM: 75.09% | Base: 89.37% | Novel: 64.74%
    # 目标: 75.30% (AnchorOPT) | 差距: -0.21%
    # 状态: ⚠️ 非常接近目标，参数对KD_W极其敏感
    # 配置: 极低KD权重(0.3)，标准温度，20视角
    # 注意: KD_W偏离0.3仅±0.05就会导致性能下降3-4%
    "801") N_VIEWS=20; KD_W=0.3; KD_T=2.0; DIV_W=0.0 ;;
    
    # ------------------------------------------------------------------------
    # UCF101 - 最佳配置 (U12)
    # ------------------------------------------------------------------------
    # HM: 76.50% | Base: 74.82% | Novel: 78.26%
    # 目标: 79.50% (AnchorOPT) | 差距: -3.00%
    # 状态: ❌ 差距较大，需要新策略
    # 配置: 高KD权重(60.0)，高温度(5.0)，25视角
    # 注意: 增加KD权重会降低Novel准确率
    "901") N_VIEWS=25; KD_W=60.0; KD_T=5.0; DIV_W=0.0 ;;
    
    *) echo "Unknown exp: $EXP"; exit 1 ;;
esac

# ============================================================================
# 开始训练
# ============================================================================
echo "============================================================================"
echo "TextConsistency V2 - 最佳配置训练"
echo "============================================================================"
echo "数据集: $DATASET"
echo "实验ID: EXP$EXP"
echo "配置参数:"
echo "  - N_VIEWS (视角数): $N_VIEWS"
echo "  - KD_W (知识蒸馏权重): $KD_W"
echo "  - KD_T (蒸馏温度): $KD_T"
echo "  - DIV_W (多样性权重): $DIV_W"
echo "  - N_CTX (上下文长度): $NCTX"
echo "  - MAX_EPOCH (训练轮数): $EPO"
echo "============================================================================"
echo ""

EXP_DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/EXP${EXP}_V${N_VIEWS}_KD${KD_W}

echo "输出目录: ${EXP_DIR}"
echo ""

for SEED in 1 2 3; do
  DIR=${EXP_DIR}/seed${SEED}
  
  echo "============================================================================"
  echo "训练 Seed ${SEED}/3"
  echo "============================================================================"
  
  CUDA_VISIBLE_DEVICES=0 python train.py --root ${DATA} --seed ${SEED} \
    --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml --output-dir ${DIR} \
    TRAINER.TEXTCONSISTENCY.N_CTX ${NCTX} \
    TRAINER.TEXTCONSISTENCY.N_VIEWS ${N_VIEWS} \
    TRAINER.TEXTCONSISTENCY.KD_WEIGHT ${KD_W} \
    TRAINER.TEXTCONSISTENCY.KD_TEMP ${KD_T} \
    TRAINER.TEXTCONSISTENCY.DIVERSITY_WEIGHT ${DIV_W} \
    TRAINER.TEXTCONSISTENCY.PREC fp32 \
    DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base OPTIM.MAX_EPOCH ${EPO}

  echo ""
  echo "测试 Base Classes (Seed ${SEED})..."
  CUDA_VISIBLE_DEVICES=0 python train.py --root ${DATA} --seed ${SEED} \
    --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml --output-dir ${DIR}/evaluation \
    --model-dir ${DIR} --eval-only \
    TRAINER.TEXTCONSISTENCY.N_CTX ${NCTX} \
    TRAINER.TEXTCONSISTENCY.N_VIEWS ${N_VIEWS} \
    TRAINER.TEXTCONSISTENCY.KD_WEIGHT ${KD_W} \
    TRAINER.TEXTCONSISTENCY.KD_TEMP ${KD_T} \
    TRAINER.TEXTCONSISTENCY.DIVERSITY_WEIGHT ${DIV_W} \
    TRAINER.TEXTCONSISTENCY.PREC fp32 \
    DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES base

  echo ""
  echo "测试 Novel Classes (Seed ${SEED})..."
  CUDA_VISIBLE_DEVICES=0 python train.py --root ${DATA} --seed ${SEED} \
    --trainer ${TRAINER} --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml --output-dir ${DIR}/evaluation \
    --model-dir ${DIR} --eval-only \
    TRAINER.TEXTCONSISTENCY.N_CTX ${NCTX} \
    TRAINER.TEXTCONSISTENCY.N_VIEWS ${N_VIEWS} \
    TRAINER.TEXTCONSISTENCY.KD_WEIGHT ${KD_W} \
    TRAINER.TEXTCONSISTENCY.KD_TEMP ${KD_T} \
    TRAINER.TEXTCONSISTENCY.DIVERSITY_WEIGHT ${DIV_W} \
    TRAINER.TEXTCONSISTENCY.PREC fp32 \
    DATASET.NUM_SHOTS ${SHOTS} DATASET.SUBSAMPLE_CLASSES new
  
  echo ""
done

# ============================================================================
# 计算平均分数并生成报告
# ============================================================================
echo "============================================================================"
echo "计算三个Seed的平均分数..."
echo "============================================================================"

BASE_SCORES=""
NOVEL_SCORES=""
for SEED in 1 2 3; do
  BASE_FILE=$(ls ${EXP_DIR}/seed${SEED}/evaluation/log.txt 2>/dev/null)
  NOVEL_FILE=$(ls ${EXP_DIR}/seed${SEED}/evaluation/log.txt-* 2>/dev/null | head -1)
  BASE_ACC=$(grep "accuracy" "$BASE_FILE" 2>/dev/null | head -1 | grep -oP '\d+\.\d+')
  NOVEL_ACC=$(grep "accuracy" "$NOVEL_FILE" 2>/dev/null | head -1 | grep -oP '\d+\.\d+')
  echo "Seed${SEED}: Base=${BASE_ACC}%, Novel=${NOVEL_ACC}%"
  BASE_SCORES="${BASE_SCORES} ${BASE_ACC}"
  NOVEL_SCORES="${NOVEL_SCORES} ${NOVEL_ACC}"
done

AVG_BASE=$(echo $BASE_SCORES | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; printf "%.2f", sum/NF}')
AVG_NOVEL=$(echo $NOVEL_SCORES | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; printf "%.2f", sum/NF}')
HM=$(echo "$AVG_BASE $AVG_NOVEL" | awk '{printf "%.2f", 2*$1*$2/($1+$2)}')

echo ""
echo "============================================================================"
echo "最终结果"
echo "============================================================================"
echo "数据集: ${DATASET}"
echo "实验ID: EXP${EXP}"
echo "配置: N_VIEWS=${N_VIEWS}, KD_W=${KD_W}, KD_T=${KD_T}, DIV_W=${DIV_W}"
echo "----------------------------------------------------------------------------"
echo "Avg Base:  ${AVG_BASE}%"
echo "Avg Novel: ${AVG_NOVEL}%"
echo "HM:        ${HM}%"
echo "============================================================================"

# 保存结果摘要
echo "========================================" > ${EXP_DIR}/summary.txt
echo "EXP${EXP} on ${DATASET}" >> ${EXP_DIR}/summary.txt
echo "Config: N_VIEWS=${N_VIEWS}, KD_W=${KD_W}, KD_T=${KD_T}, DIV_W=${DIV_W}" >> ${EXP_DIR}/summary.txt
echo "----------------------------------------" >> ${EXP_DIR}/summary.txt
echo "Avg Base:  ${AVG_BASE}%" >> ${EXP_DIR}/summary.txt
echo "Avg Novel: ${AVG_NOVEL}%" >> ${EXP_DIR}/summary.txt
echo "HM:        ${HM}%" >> ${EXP_DIR}/summary.txt
echo "========================================" >> ${EXP_DIR}/summary.txt

echo ""
echo "结果已保存至: ${EXP_DIR}/summary.txt"
echo ""
echo "============================================================================"
echo "训练完成！"
echo "============================================================================"
