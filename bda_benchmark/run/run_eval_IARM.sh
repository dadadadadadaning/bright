#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
CKPT_DIR="${1:?Usage: run_eval_IARM.sh CKPT_DIR [CHECKPOINT_NAME]}"
CHECKPOINT_NAME="${2:-best_val_miou.pth}"
CKPT="${CKPT_DIR}/${CHECKPOINT_NAME}"
CONFIG="${CKPT_DIR}/config.json"
OUTPUT="${CKPT_DIR}/eval_${CHECKPOINT_NAME%.pth}.json"
cd "${PROJECT_ROOT}"
export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0 python script/standard_ML/eval_DamageFormer_IARM.py \
  --checkpoint "${CKPT}" --config_json "${CONFIG}" \
  --dataset_path "${PROJECT_ROOT}/dataset" \
  --val_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt" \
  --test_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/test_set.txt" \
  --split both --eval_batch_size 1 --num_workers 4 --device cuda:0 \
  --output_json "${OUTPUT}" 2>&1 | tee "${OUTPUT%.json}.log"
