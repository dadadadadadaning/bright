#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
CKPT="/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth"
HEAD="$(dirname "${CKPT}")/S9I_CenterBoundary/best_val_instance_f1.pth"
S9A="$(dirname "${CKPT}")/S9A_PredictedInstance/s9a_predicted_instance_val.json"
OUT="$(dirname "${CKPT}")/S9I_CenterBoundary"
cd "${ROOT}"
export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${ROOT}:${ROOT}/script/standard_ML:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" python \
  script/standard_ML/eval_S9I_center_boundary.py \
  --s8_checkpoint "${CKPT}" \
  --geometry_checkpoint "${HEAD}" \
  --s9a_json "${S9A}" \
  --output_json "${OUT}/s9i_final_val.json" \
  --device cuda:0 \
  --num_workers "${NUM_WORKERS:-8}" \
  2>&1 | tee "${OUT}/s9i_final_val.log"
