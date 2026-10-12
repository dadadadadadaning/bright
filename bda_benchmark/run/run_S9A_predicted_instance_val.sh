#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
CKPT="/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth"
OUT_DIR="$(dirname "${CKPT}")/S9A_PredictedInstance"

mkdir -p "${OUT_DIR}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${PROJECT_ROOT}:${PROJECT_ROOT}/script/standard_ML:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" python \
  script/standard_ML/eval_S9A_predicted_instance.py \
  --checkpoint "${CKPT}" \
  --output_json "${OUT_DIR}/s9a_predicted_instance_val.json" \
  --device cuda:0 \
  --num_workers "${NUM_WORKERS:-8}" \
  --min_instance_area 32 \
  --matching greedy \
  --match_iou 0.5 \
  --merge_split_overlap 0.10 \
  --expected_parent_miou 69.76121287610887 \
  2>&1 | tee "${OUT_DIR}/s9a_predicted_instance_val.log"
