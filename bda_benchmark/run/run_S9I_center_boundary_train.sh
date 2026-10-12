#!/usr/bin/env bash
set -euo pipefail
ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
CKPT="/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth"
A2B="$(dirname "${CKPT}")/S9A2b_ConditionalWatershed/s9a2b_decision.json"
OUT="$(dirname "${CKPT}")/S9I_CenterBoundary"
mkdir -p "${OUT}"
cd "${ROOT}"
export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${ROOT}:${ROOT}/script/standard_ML:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MAX_JOBS=1
export MALLOC_ARENA_MAX=2
CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" python \
  script/standard_ML/train_S9I_center_boundary.py \
  --checkpoint "${CKPT}" \
  --a2b_json "${A2B}" \
  --output_dir "${OUT}" \
  --device cuda:0 \
  --train_batch_size "${MICRO_BATCH:-4}" \
  --accumulation_steps "${ACCUM_STEPS:-4}" \
  --optimizer_steps "${OPTIMIZER_STEPS:-3000}" \
  --max_iters "${MAX_ITERS:-48000}" \
  --crop_size 640 \
  --eval_every 1000 \
  --log_every 20 \
  --num_workers "${NUM_WORKERS:-8}" \
  --learning_rate 1e-3 \
  --weight_decay 1e-4 \
  --hidden_channels 64 \
  --amp bf16 \
  --seed 3407 \
  2>&1 | tee "${OUT}/s9i_training.log"
