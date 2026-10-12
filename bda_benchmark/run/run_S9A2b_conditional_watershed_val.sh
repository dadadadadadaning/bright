#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
CKPT="/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth"
S9A="$(dirname "${CKPT}")/S9A_PredictedInstance/s9a_predicted_instance_val.json"
S9A2="$(dirname "${CKPT}")/S9A2_ProbabilitySeed_Watershed/s9a2_decision.json"
OUT="$(dirname "${CKPT}")/S9A2b_ConditionalWatershed"
mkdir -p "${OUT}"
cd "${ROOT}"
export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${ROOT}:${ROOT}/script/standard_ML:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" python \
  script/standard_ML/eval_S9A2b_conditional_watershed.py \
  --checkpoint "${CKPT}" \
  --s9a_json "${S9A}" \
  --s9a2_json "${S9A2}" \
  --output_json "${OUT}/s9a2b_decision.json" \
  --device cuda:0 \
  --num_workers "${NUM_WORKERS:-8}" \
  2>&1 | tee "${OUT}/s9a2b_validation.log"
