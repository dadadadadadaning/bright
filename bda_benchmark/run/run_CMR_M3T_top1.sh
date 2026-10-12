#!/usr/bin/env bash
set -euo pipefail

# CMR-M3T keeps the admitted Top-20 pool and M3C conditional heads fixed.
# The only model-training change is Listwise -> class-balanced hard Top-1 loss.
MODE="${1:-train}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

SEED="${SEED:-3407}"
RESULT_ROOT="${RESULT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_M3_PAIRWISE}"
CKPT_ROOT="${CKPT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/CMR_M3T_TOP1}"
TRAIN_CANDIDATES="${TRAIN_CANDIDATES:-${RESULT_ROOT}/CMR_M3_train_top20_seed${SEED}.pth}"
VAL_CANDIDATES="${VAL_CANDIDATES:-${RESULT_ROOT}/CMR_M3_val_top20_seed${SEED}.pth}"
ORACLE_RESULT="${ORACLE_RESULT:-${RESULT_ROOT}/CMR_M3O_oracle_seed${SEED}.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${CKPT_ROOT}/CMR_M3T_top20_epochs${EPOCHS:-6}_seed${SEED}}"

preflight() {
  python -m py_compile \
    model/cmr/pairwise.py \
    script/standard_ML/cmr_pairwise_utils.py \
    script/standard_ML/train_CMR_M3C_conditional.py
  python -m unittest \
    tests.test_cmr_m3_pairwise \
    tests.test_cmr_m3c_conditional \
    tests.test_cmr_m3t_top1 -v
  echo "CMR-M3T class-balanced hard Top-1 preflight passed."
}

train_top1() {
  test -s "${ORACLE_RESULT}" || { echo "Missing admitted Oracle result" >&2; exit 3; }
  test -s "${TRAIN_CANDIDATES}" || { echo "Missing Train candidates" >&2; exit 3; }
  test -s "${VAL_CANDIDATES}" || { echo "Missing Validation candidates" >&2; exit 3; }
  mkdir -p "${OUTPUT_DIR}"
  python -u script/standard_ML/train_CMR_M3C_conditional.py \
    --train_candidates "${TRAIN_CANDIDATES}" \
    --val_candidates "${VAL_CANDIDATES}" \
    --oracle_result "${ORACLE_RESULT}" \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda:0 \
    --epochs "${EPOCHS:-6}" \
    --steps_per_epoch "${STEPS_PER_EPOCH:-500}" \
    --events_per_class "${EVENTS_PER_CLASS:-4}" \
    --samples_per_event "${SAMPLES_PER_EVENT:-8}" \
    --hidden_dim "${HIDDEN_DIM:-128}" \
    --maximum_residual "${MAXIMUM_RESIDUAL:-0.05}" \
    --temperature "${TEMPERATURE:-0.10}" \
    --loss_protocol class_balanced_top1 \
    --top1_margin "${TOP1_MARGIN:-0.0}" \
    --learning_rate "${LEARNING_RATE:-0.001}" \
    --weight_decay "${WEIGHT_DECAY:-0.0001}" \
    --minimum_damaged_gain "${MINIMUM_DAMAGED_GAIN:-0.01}" \
    --maximum_destroyed_drop "${MAXIMUM_DESTROYED_DROP:-0.01}" \
    --minimum_damaged_events "${MINIMUM_DAMAGED_EVENTS:-10}" \
    --seed "${SEED}"
}

case "${MODE}" in
  preflight) preflight ;;
  train) train_top1 ;;
  all)
    preflight
    train_top1
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|train|all" >&2
    exit 2
    ;;
esac

