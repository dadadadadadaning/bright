#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
BASE="/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16"
S8_CKPT="${BASE}/best_val_miou.pth"
S9IB_JSON="${BASE}/S9Ib_ConservativeCenter/S9IB_FINAL_VALIDATION_LOCKED.json"

cd "${ROOT}"
export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${ROOT}:${ROOT}/script/standard_ML:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" \
python script/standard_ML/smoke_S9Ic_G1.py \
  --checkpoint "${S8_CKPT}" \
  --s9ib_json "${S9IB_JSON}" \
  --device cuda:0 \
  --num_workers "${NUM_WORKERS:-2}" \
  --amp bf16
