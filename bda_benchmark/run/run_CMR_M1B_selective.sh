#!/usr/bin/env bash
set -euo pipefail

# CMR-M1B is isolated from P3, old CMR experiments and Test.  Its only
# controlled change is the selective positive policy for Damaged samples.
MODE="${1:-screen}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

SEED="${SEED:-3407}"
P0_ROOT="${P0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_POOLING}"
TRAIN_CACHE="${TRAIN_CACHE:-${P0_ROOT}/CMR_P0_train_pooling_seed${SEED}.pth}"
VAL_CACHE="${VAL_CACHE:-${P0_ROOT}/CMR_P0_val_pooling_seed${SEED}.pth}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/CMR_M1B_SELECTIVE}"

preflight() {
  python -m py_compile \
    model/cmr/metric.py \
    script/standard_ML/cmr_metric_utils.py \
    script/standard_ML/train_CMR_M1B_selective.py
  python -m unittest \
    tests.test_cmr_pooling \
    tests.test_cmr_m1_metric \
    tests.test_cmr_m1b_selective -v
  echo "CMR-M1B selective-Damaged preflight passed."
}

ensure_caches() {
  if [[ ! -s "${TRAIN_CACHE}" || ! -s "${VAL_CACHE}" ]]; then
    bash run/run_CMR_pooling_audit.sh build
  fi
  test -s "${TRAIN_CACHE}" || { echo "Missing Train cache: ${TRAIN_CACHE}" >&2; exit 3; }
  test -s "${VAL_CACHE}" || { echo "Missing Validation cache: ${VAL_CACHE}" >&2; exit 3; }
}

train_selective() {
  ensure_caches
  local epochs="${EPOCHS:-30}"
  local steps="${STEPS_PER_EPOCH:-500}"
  local topk="${DAMAGED_POSITIVE_TOPK:-2}"
  local output="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/CMR_M1B_mean_top${topk}_epochs${epochs}_seed${SEED}}"
  mkdir -p "${output}"
  python -u script/standard_ML/train_CMR_M1B_selective.py \
    --train_cache "${TRAIN_CACHE}" \
    --val_cache "${VAL_CACHE}" \
    --output_dir "${output}" \
    --device cuda:0 \
    --epochs "${epochs}" \
    --steps_per_epoch "${steps}" \
    --eval_every "${EVAL_EVERY:-2}" \
    --log_every "${LOG_EVERY:-50}" \
    --hidden_dim "${HIDDEN_DIM:-256}" \
    --events_per_class "${EVENTS_PER_CLASS:-4}" \
    --samples_per_event "${SAMPLES_PER_EVENT:-2}" \
    --damaged_positive_topk "${topk}" \
    --temperature "${TEMPERATURE:-0.10}" \
    --learning_rate "${LEARNING_RATE:-0.001}" \
    --weight_decay "${WEIGHT_DECAY:-0.0001}" \
    --query_chunk "${QUERY_CHUNK:-1024}" \
    --project_batch_size "${PROJECT_BATCH_SIZE:-4096}" \
    --ap_k "${AP_K:-10}" \
    --relation_pairs "${RELATION_PAIRS:-200000}" \
    --minimum_damaged_gain "${MINIMUM_DAMAGED_GAIN:-0.01}" \
    --maximum_destroyed_drop "${MAXIMUM_DESTROYED_DROP:-0.02}" \
    --seed "${SEED}"
}

case "${MODE}" in
  preflight) preflight ;;
  smoke)
    EPOCHS="${EPOCHS:-1}" \
    STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-5}" \
    EVAL_EVERY="${EVAL_EVERY:-1}" \
    train_selective
    ;;
  screen) train_selective ;;
  all)
    preflight
    train_selective
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|smoke|screen|all" >&2
    exit 2
    ;;
esac
