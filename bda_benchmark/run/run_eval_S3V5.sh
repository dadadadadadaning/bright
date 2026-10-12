#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
CKPT_DIR="/root/shared-nvme/BRIGHT-checkpoints/S3V5/BRIGHT/S3V5_K133_HardGatedLogit_2x4090_EBS16_50k_20260714_111137_20260714_111145"
CKPT="${CKPT_DIR}/best_model.pth"
OUT_JSON="${CKPT_DIR}/eval_S3V5_active_vs_main_full.json"
LOG_FILE="${CKPT_DIR}/eval_S3V5_active_vs_main_full.log"

cd "${PROJECT_ROOT}"
export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0 python \
  script/standard_ML/eval_S3V5_K3Gated.py \
  --checkpoint "${CKPT}" \
  --dataset_path "${PROJECT_ROOT}/dataset" \
  --val_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt" \
  --test_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/test_set.txt" \
  --split both \
  --eval_batch_size 1 \
  --num_workers 4 \
  --device cuda:0 \
  --amp none \
  --lambda_sweep "0,0.02,0.05,0.08" \
  --output_json "${OUT_JSON}" \
  2>&1 | tee "${LOG_FILE}"
