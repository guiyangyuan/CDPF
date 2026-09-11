#!/usr/bin/env bash
# Prepared only. Enable a GPU on this same instance before running.
# Existing or failed studies are preserved; no automatic retry or overwrite.
set -euo pipefail
TC_ROOT=/root/autodl-tmp/textconsistency
TC_OPS="$TC_ROOT/releases/attribution-ops-20260911-778d9d0"
TC_PREP="$TC_ROOT/setup-logs/attribution-prep-20260911-v1/prepared-778d9d0"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
cd "$TC_OPS"
exec "$TC_ROOT/.venv/bin/python" -B "$TC_OPS/scripts/repro/attribution_study.py" \
  --source-root "$TC_ROOT/releases/structure-20260910-v1" \
  --data-root "$TC_ROOT/data" \
  --output-root "$TC_ROOT/runs/attribution-study-20260911-v1" \
  --input-receipt "$TC_ROOT/reused-inputs/attribution-20260911/input-receipt.json" \
  --preparation-dir "$TC_PREP" \
  --clip-path "$TC_ROOT/assets/ViT-B-16.pt" \
  --execute
