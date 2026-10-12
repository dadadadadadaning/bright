#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/shared-nvme/BRIGHT-master/bda_benchmark
DATA_ROOT=${DATA_ROOT:-$ROOT/dataset}
VAL_LIST=${VAL_LIST:-$DATA_ROOT/splitname/standard_ML/val_set.txt}
CKPT=${1:?Usage: bash run/run_B1_analysis.sh /path/to/B0B/best_miou3.pth}
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 python -u script/standard_ML/analyze_B1_instance_pooling.py \
  --checkpoint "$CKPT" --dataset_path "$DATA_ROOT" --data_list_path "$VAL_LIST" \
  --split val --output_dir /root/shared-nvme/BRIGHT-eval-results/B1_instance_analysis
