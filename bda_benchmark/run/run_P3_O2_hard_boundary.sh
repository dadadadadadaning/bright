#!/usr/bin/env bash
set -euo pipefail

# Final bounded diagnostic: exact Train-only D->I regions versus risky Intact.
# The cache supplies region identity and frozen P3 outputs only; feature keys,
# neighbours, Memory voting and Test are forbidden.

MODE="${1:-screen}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
P3_CHECKPOINT="${P3_CHECKPOINT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
TRAIN_CACHE="${TRAIN_CACHE:-/root/shared-nvme/BRIGHT-eval-results/CMR_REPLAY_R1/CMR_POOLING/CMR_P0_train_pooling_seed3407.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/P3_O2_HARD_BOUNDARY}"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

preflight() {
  python -m py_compile \
    script/standard_ML/p3_direct_occurrence_runtime.py \
    script/standard_ML/train_P3_O1_direct_occurrence.py \
    script/standard_ML/train_P3_O2_hard_boundary.py
  python -m unittest tests/test_p3_o2_hard_boundary.py -v
  echo "P3-O2 hard boundary preflight passed."
}

run_screen() {
  test -s "${P3_CHECKPOINT}"
  test -s "${TRAIN_CACHE}"
  local output_dir="${OUTPUT_DIR:-${OUTPUT_ROOT}/P3_O2_screen500_seed${SEED}}"
  mkdir -p "${output_dir}"

  python -u script/standard_ML/train_P3_O2_hard_boundary.py \
    --train_dataset_path "${DATASET_ROOT}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --val_dataset_path "${DATASET_ROOT}" \
    --val_data_list_path "${VAL_LIST}" \
    --train_cache "${TRAIN_CACHE}" \
    --p3_checkpoint "${P3_CHECKPOINT}" \
    --output_dir "${output_dir}" \
    --device cuda:0 \
    --crop_size 640 \
    --train_batch_size "${TRAIN_BATCH_SIZE:-2}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --optimizer_steps 500 \
    --eval_every 250 \
    --log_every 20 \
    --seed "${SEED}" \
    --learning_rate "${LEARNING_RATE:-0.000001}" \
    --weight_decay "${WEIGHT_DECAY:-0.005}" \
    --hard_weight "${HARD_WEIGHT:-0.05}" \
    --protect_weight "${PROTECT_WEIGHT:-0.5}" \
    --required_margin 0.0 \
    --cap_per_event 64 \
    --minimum_damaged_gain_pp "${MINIMUM_DAMAGED_GAIN_PP:-0.30}" \
    --maximum_guard_drop_pp "${MAXIMUM_GUARD_DROP_PP:-0.20}" \
    --minimum_nonnegative_events "${MINIMUM_NONNEGATIVE_EVENTS:-10}"
}

case "${MODE}" in
  preflight) preflight ;;
  screen) run_screen ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|screen" >&2
    exit 2
    ;;
esac

