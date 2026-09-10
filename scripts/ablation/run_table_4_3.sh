#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-${DATA:-/path/to/data}}
GPU=${GPU:-0}
DATASET=${1:-caltech101}
GROUP=${2:-all}
SEEDS=${SEEDS:-1 2 3}

TRAINER=CoOp_TextConsistency_Ablation
CFG=vit_b16
SHOTS=16
PREC=fp32
OUT_ROOT=output/ablation/table_4_3
GROUPS="baseline single_view_kd no_kd v2_no_fg full"
PAPER_DATASETS="caltech101 ucf101 fgvc_aircraft"

case "${DATASET}" in
  caltech101)
    NCTX=4
    EPOCHS=100
    N_VIEWS=20
    KD_W=60.0
    KD_T=3.7
    CONTRAST_W=0.0
    DIV_W=0.0
    N_GROUPS=4
    ;;
  ucf101)
    NCTX=8
    EPOCHS=100
    N_VIEWS=55
    KD_W=0.4
    KD_T=5.6
    CONTRAST_W=0.1
    DIV_W=0.05
    N_GROUPS=5
    ;;
  fgvc_aircraft)
    NCTX=4
    EPOCHS=100
    N_VIEWS=22
    KD_W=1.8
    KD_T=2.0
    CONTRAST_W=0.1
    DIV_W=0.05
    N_GROUPS=4
    ;;
  *)
    echo "Unknown dataset: ${DATASET}"
    echo "Supported paper datasets: ${PAPER_DATASETS}"
    exit 1
    ;;
esac

if [ "${GROUP}" = "all" ]; then
  SELECTED_GROUPS="${GROUPS}"
else
  case " ${GROUPS} " in
    *" ${GROUP} "*) SELECTED_GROUPS="${GROUP}" ;;
    *)
      echo "Unknown group: ${GROUP}"
      echo "Supported groups: ${GROUPS}"
      exit 1
      ;;
  esac
fi

run_one() {
  local group_name=$1
  local seed=$2
  local train_dir="${OUT_ROOT}/${DATASET}/${group_name}/seed${seed}"

  echo "================================================================"
  echo "Dataset=${DATASET} Group=${group_name} Seed=${seed}"
  echo "N_CTX=${NCTX} N_VIEWS=${N_VIEWS} KD_W=${KD_W} KD_T=${KD_T}"
  echo "================================================================"

  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root "${DATA_ROOT}" \
    --seed "${seed}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/CoOp/${CFG}.yaml" \
    --output-dir "${train_dir}" \
    TRAINER.TEXTCONSISTENCY.ABLATION_GROUP "${group_name}" \
    TRAINER.TEXTCONSISTENCY.N_CTX "${NCTX}" \
    TRAINER.TEXTCONSISTENCY.N_VIEWS "${N_VIEWS}" \
    TRAINER.TEXTCONSISTENCY.KD_WEIGHT "${KD_W}" \
    TRAINER.TEXTCONSISTENCY.KD_TEMP "${KD_T}" \
    TRAINER.TEXTCONSISTENCY.ADAPTIVE_CE_WEIGHT 1.0 \
    TRAINER.TEXTCONSISTENCY.CONTRASTIVE_WEIGHT "${CONTRAST_W}" \
    TRAINER.TEXTCONSISTENCY.DIVERSITY_WEIGHT "${DIV_W}" \
    TRAINER.TEXTCONSISTENCY.N_GROUPS "${N_GROUPS}" \
    TRAINER.TEXTCONSISTENCY.PREC "${PREC}" \
    DATASET.NUM_SHOTS "${SHOTS}" \
    DATASET.SUBSAMPLE_CLASSES base \
    OPTIM.MAX_EPOCH "${EPOCHS}"

  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root "${DATA_ROOT}" \
    --seed "${seed}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/CoOp/${CFG}.yaml" \
    --output-dir "${train_dir}/eval_base" \
    --model-dir "${train_dir}" \
    --eval-only \
    TRAINER.TEXTCONSISTENCY.ABLATION_GROUP "${group_name}" \
    TRAINER.TEXTCONSISTENCY.N_CTX "${NCTX}" \
    TRAINER.TEXTCONSISTENCY.N_VIEWS "${N_VIEWS}" \
    TRAINER.TEXTCONSISTENCY.KD_WEIGHT "${KD_W}" \
    TRAINER.TEXTCONSISTENCY.KD_TEMP "${KD_T}" \
    TRAINER.TEXTCONSISTENCY.ADAPTIVE_CE_WEIGHT 1.0 \
    TRAINER.TEXTCONSISTENCY.CONTRASTIVE_WEIGHT "${CONTRAST_W}" \
    TRAINER.TEXTCONSISTENCY.DIVERSITY_WEIGHT "${DIV_W}" \
    TRAINER.TEXTCONSISTENCY.N_GROUPS "${N_GROUPS}" \
    TRAINER.TEXTCONSISTENCY.PREC "${PREC}" \
    DATASET.NUM_SHOTS "${SHOTS}" \
    DATASET.SUBSAMPLE_CLASSES base

  CUDA_VISIBLE_DEVICES=${GPU} python train.py \
    --root "${DATA_ROOT}" \
    --seed "${seed}" \
    --trainer "${TRAINER}" \
    --dataset-config-file "configs/datasets/${DATASET}.yaml" \
    --config-file "configs/trainers/CoOp/${CFG}.yaml" \
    --output-dir "${train_dir}/eval_new" \
    --model-dir "${train_dir}" \
    --eval-only \
    TRAINER.TEXTCONSISTENCY.ABLATION_GROUP "${group_name}" \
    TRAINER.TEXTCONSISTENCY.N_CTX "${NCTX}" \
    TRAINER.TEXTCONSISTENCY.N_VIEWS "${N_VIEWS}" \
    TRAINER.TEXTCONSISTENCY.KD_WEIGHT "${KD_W}" \
    TRAINER.TEXTCONSISTENCY.KD_TEMP "${KD_T}" \
    TRAINER.TEXTCONSISTENCY.ADAPTIVE_CE_WEIGHT 1.0 \
    TRAINER.TEXTCONSISTENCY.CONTRASTIVE_WEIGHT "${CONTRAST_W}" \
    TRAINER.TEXTCONSISTENCY.DIVERSITY_WEIGHT "${DIV_W}" \
    TRAINER.TEXTCONSISTENCY.N_GROUPS "${N_GROUPS}" \
    TRAINER.TEXTCONSISTENCY.PREC "${PREC}" \
    DATASET.NUM_SHOTS "${SHOTS}" \
    DATASET.SUBSAMPLE_CLASSES new
}

for group_name in ${SELECTED_GROUPS}; do
  for seed in ${SEEDS}; do
    run_one "${group_name}" "${seed}"
  done
done

python scripts/ablation/summarize_table_4_3.py --root "${OUT_ROOT}"
