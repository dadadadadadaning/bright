#!/usr/bin/env bash
set -euo pipefail

# Isolated P3-HR0/HR1/HR2/HR3-O Occurrence--Severity experiments.
# P3 is frozen; Memory and Test are deliberately unavailable.

MODE="${1:-preflight}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
P3_CHECKPOINT="${P3_CHECKPOINT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/P3_HIERARCHICAL_RESIDUAL}"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

preflight() {
  python -m py_compile \
    model/DamageFormer_P3_HierarchicalResidual.py \
    script/standard_ML/p3_hierarchical_residual_runtime.py \
    script/standard_ML/train_P3_HierarchicalResidual.py
  python -m unittest tests/test_p3_hierarchical_residual.py -v
  echo "P3-HR hierarchical residual preflight passed."
}

run_variant() {
  local variant="$1"
  local steps="$2"
  test -s "${P3_CHECKPOINT}"
  local default_name="P3_${variant^^}_screen${steps}_seed${SEED}"
  if [[ "${variant}" == "hr0" ]]; then
    default_name="P3_HR0_exact_guard_seed${SEED}"
  fi
  local protection_default="0.10"
  local severity_default="0.5"
  local four_class_default="0.25"
  local repairable_min_default="-2.0"
  if [[ "${variant}" == "hr3o" ]]; then
    protection_default="1.0"
    severity_default="0.0"
    four_class_default="0.0"
  fi
  if [[ "${variant}" == "hr1o" ]]; then
    severity_default="0.0"
  fi
  if [[ "${variant}" == "hr1s" ]]; then
    occurrence_default="0.0"
  else
    occurrence_default="1.0"
  fi
  local output_dir="${OUTPUT_DIR:-${OUTPUT_ROOT}/${default_name}}"
  mkdir -p "${output_dir}"
  python -u script/standard_ML/train_P3_HierarchicalResidual.py \
    --variant "${variant}" \
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
    --optimizer_steps "${steps}" \
    --eval_every 250 \
    --log_every "${LOG_EVERY:-20}" \
    --seed "${SEED}" \
    --learning_rate "${LEARNING_RATE:-0.0001}" \
    --weight_decay "${WEIGHT_DECAY:-0.0001}" \
    --hard_fraction "${HARD_FRACTION:-0.25}" \
    --minimum_hard_pixels "${MINIMUM_HARD_PIXELS:-64}" \
    --minimum_component_area "${MINIMUM_COMPONENT_AREA:-16}" \
    --repairable_min_margin "${REPAIRABLE_MIN_MARGIN:-${repairable_min_default}}" \
    --repairable_max_margin "${REPAIRABLE_MAX_MARGIN:-0.0}" \
    --required_margin "${REQUIRED_MARGIN:-0.0}" \
    --four_class_weight "${FOUR_CLASS_WEIGHT:-${four_class_default}}" \
    --occurrence_weight "${OCCURRENCE_WEIGHT:-${occurrence_default}}" \
    --severity_weight "${SEVERITY_WEIGHT:-${severity_default}}" \
    --protection_weight "${PROTECTION_WEIGHT:-${protection_default}}" \
    --maximum_occurrence_delta "${MAXIMUM_OCCURRENCE_DELTA:-2.0}" \
    --maximum_severity_delta "${MAXIMUM_SEVERITY_DELTA:-2.0}" \
    --minimum_damaged_gain_pp "${MINIMUM_DAMAGED_GAIN_PP:-0.30}" \
    --minimum_gate_miss_gain_pp "${MINIMUM_GATE_MISS_GAIN_PP:-0.50}" \
    --minimum_affected_recall_gain_pp "${MINIMUM_AFFECTED_RECALL_GAIN_PP:-0.50}" \
    --maximum_guard_drop_pp "${MAXIMUM_GUARD_DROP_PP:-0.20}" \
    --minimum_nonnegative_events "${MINIMUM_NONNEGATIVE_EVENTS:-10}"
}

case "${MODE}" in
  preflight) preflight ;;
  guard) run_variant hr0 0 ;;
  hr1) run_variant hr1 500 ;;
  hr1o) run_variant hr1o 500 ;;
  hr1s) run_variant hr1s 500 ;;
  hr2) run_variant hr2 500 ;;
  hr3o) run_variant hr3o 500 ;;
  auto_h1)
    preflight
    run_variant hr0 0
    guard_summary="${OUTPUT_ROOT}/P3_HR0_exact_guard_seed${SEED}/training_summary.json"
    python - "${guard_summary}" <<'PY'
import json
import sys

row = json.load(open(sys.argv[1], encoding="utf-8"))
assert row["decision"] == "ADMIT_TO_P3_HR1_500_STEP_SCREEN", row
assert row["exact_online_guard"] is True, row
assert row["test_accessed"] is False, row
print("H0 exact P3 guard passed; starting isolated H1.")
PY
    run_variant hr1 500
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|guard|hr1|hr1o|hr1s|hr2|hr3o|auto_h1" >&2
    exit 2
    ;;
esac
