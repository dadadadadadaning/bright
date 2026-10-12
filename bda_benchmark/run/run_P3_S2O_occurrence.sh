#!/usr/bin/env bash
set -euo pipefail

# P3-S2O: frozen-P3 Stage2 local Occurrence branch.
# This isolated screen never loads Memory and has no Test entry point.

MODE="${1:-screen}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
P3_CHECKPOINT="${P3_CHECKPOINT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/P3_S2O_OCCURRENCE}"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

preflight() {
  python -m py_compile \
    model/DamageFormer_P3_S2Occurrence.py \
    script/standard_ML/p3_s2_occurrence_runtime.py \
    script/standard_ML/train_P3_S2O_occurrence.py
  python -m unittest tests/test_p3_s2_occurrence.py -v
  echo "P3-S2O Stage2 Occurrence preflight passed."
}

screen() {
  test -s "${P3_CHECKPOINT}"
  local output_dir="${OUTPUT_DIR:-${OUTPUT_ROOT}/P3_S2O_screen500_seed${SEED}}"
  mkdir -p "${output_dir}"
  python -u script/standard_ML/train_P3_S2O_occurrence.py \
    --train_dataset_path "${DATASET_ROOT}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --val_dataset_path "${DATASET_ROOT}" \
    --val_data_list_path "${VAL_LIST}" \
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
    --learning_rate "${LEARNING_RATE:-0.0001}" \
    --weight_decay "${WEIGHT_DECAY:-0.0001}" \
    --hard_fraction "${HARD_FRACTION:-0.25}" \
    --minimum_hard_pixels "${MINIMUM_HARD_PIXELS:-64}" \
    --required_margin "${REQUIRED_MARGIN:-0.0}" \
    --protection_weight "${PROTECTION_WEIGHT:-0.10}" \
    --maximum_delta "${MAXIMUM_DELTA:-2.0}" \
    --minimum_damaged_gain_pp "${MINIMUM_DAMAGED_GAIN_PP:-0.30}" \
    --maximum_guard_drop_pp "${MAXIMUM_GUARD_DROP_PP:-0.20}" \
    --minimum_nonnegative_events "${MINIMUM_NONNEGATIVE_EVENTS:-10}"
}

case "${MODE}" in
  preflight) preflight ;;
  screen) screen ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|screen" >&2
    exit 2
    ;;
esac
