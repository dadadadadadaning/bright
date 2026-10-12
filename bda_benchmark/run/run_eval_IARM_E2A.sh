#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run/run_eval_IARM_E2A.sh <experiment_dir> [split]"
  exit 2
fi

PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
EXPERIMENT_DIR="$1"
SPLIT="${2:-both}"
GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

CHECKPOINT="${EXPERIMENT_DIR}/best_val_exist_f1.pth"
CONFIG="${EXPERIMENT_DIR}/config.json"
OUTPUT="${EXPERIMENT_DIR}/eval_best_val_exist_f1_${SPLIT}.json"
LOG="${EXPERIMENT_DIR}/eval_best_val_exist_f1_${SPLIT}.log"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: missing ${CHECKPOINT}"
  exit 3
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CUDA_VISIBLE_DEVICES="${GPU_ID}" python \
  script/standard_ML/eval_DamageFormer_IARM_E2A.py \
  --checkpoint "${CHECKPOINT}" \
  --config_json "${CONFIG}" \
  --dataset_path "${PROJECT_ROOT}/dataset" \
  --val_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt" \
  --test_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/test_set.txt" \
  --split "${SPLIT}" \
  --eval_batch_size 1 \
  --num_workers "${NUM_WORKERS}" \
  --device cuda:0 \
  --output_json "${OUTPUT}" \
  2>&1 | tee "${LOG}"
