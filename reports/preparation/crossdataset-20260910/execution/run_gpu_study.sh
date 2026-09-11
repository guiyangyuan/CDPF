#!/usr/bin/env bash
# Prepared only. Run after the user enables a GPU on this same instance.
# A failed or existing study is preserved; the controller refuses to overwrite it.
set -euo pipefail
TC_ROOT=/root/autodl-tmp/textconsistency
TC_OPS="$TC_ROOT/releases/crossdataset-ops-20260910-f20795c"
TC_SOURCE="$TC_ROOT/releases/structure-20260910-v1"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
cd "$TC_OPS"
exec "$TC_ROOT/.venv/bin/python" -B "$TC_OPS/scripts/repro/cross_dataset_study.py" \
  --source-root "$TC_SOURCE" \
  --data-root "$TC_ROOT/data" \
  --output-root "$TC_ROOT/runs/crossdataset-study-20260910-v1" \
  --preparation-dir "$TC_ROOT/setup-logs/crossdataset-prep-20260910/prepared-v1" \
  --clip-path "$TC_ROOT/assets/ViT-B-16.pt" \
  --execute
