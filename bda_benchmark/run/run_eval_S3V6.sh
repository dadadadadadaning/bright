#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
CKPT="${1:?Usage: bash run_eval_S3V6.sh /path/to/checkpoint.pth}"
CKPT_DIR="$(dirname "${CKPT}")"
OUTPUT_JSON="${2:-${CKPT_DIR}/eval_S3V6_full.json}"
LOG_FILE="${OUTPUT_JSON%.json}.log"

cd "${PROJECT_ROOT}"
export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0 python \
  script/standard_ML/eval_S3V6_EventBalanced.py \
  --checkpoint "${CKPT}" \
  --dataset_path "${PROJECT_ROOT}/dataset" \
  --val_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt" \
  --test_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/test_set.txt" \
  --split both \
  --eval_batch_size 1 \
  --num_workers 4 \
  --device cuda:0 \
  --amp none \
  --lambda_sweep "0,0.02,0.05,0.08,0.10" \
  --output_json "${OUTPUT_JSON}" \
  2>&1 | tee "${LOG_FILE}"
