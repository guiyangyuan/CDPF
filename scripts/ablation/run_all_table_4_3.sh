#!/usr/bin/env bash
set -euo pipefail

DATASETS="caltech101 ucf101 fgvc_aircraft"
GROUP=${GROUP:-all}

for dataset in ${DATASETS}; do
  bash scripts/ablation/run_table_4_3.sh "${dataset}" "${GROUP}"
done
