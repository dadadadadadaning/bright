#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

SEED="${SEED:-3407}"
EVAL_ROOT="${EVAL_ROOT:-/root/shared-nvme/BRIGHT-eval-results/RM_REGION_RETRIEVAL}"
MEMORY="${MEMORY:-${EVAL_ROOT}/RM3E_class_event_train_memory_seed${SEED}.pth}"
QUERIES="${QUERIES:-${EVAL_ROOT}/RM3E_val_queries_seed${SEED}.pth}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/RM5_CROSS_EVENT_METRIC}"

preflight() {
  python -m py_compile \
    model/RM5_CrossEventMetric.py \
    script/standard_ML/train_RM5_cross_event_metric.py
  python -m unittest \
    tests.test_rm_region_retrieval \
    tests.test_rm5_cross_event_metric -v
  echo "RM5 cross-event metric preflight passed."
}

ensure_caches() {
  if [[ ! -s "${MEMORY}" || ! -s "${QUERIES}" ]]; then
    bash run/run_RM_region_retrieval.sh rm3e_build
  fi
  test -s "${MEMORY}" || { echo "Missing RM5 train memory: ${MEMORY}" >&2; exit 3; }
  test -s "${QUERIES}" || { echo "Missing RM5 validation queries: ${QUERIES}" >&2; exit 3; }
}

train_metric() {
  local name="$1"
  local steps="$2"
  local eval_every="$3"
  local variant="$4"
  local preserve_weight="$5"
  local output="${CHECKPOINT_ROOT}/${name}"
  mkdir -p "${output}"
  python -u script/standard_ML/train_RM5_cross_event_metric.py \
    --memory "${MEMORY}" \
    --queries "${QUERIES}" \
    --output_dir "${output}" \
    --variant "${variant}" \
    --device cuda:0 \
    --optimizer_steps "${steps}" \
    --eval_every "${eval_every}" \
    --log_every "${LOG_EVERY:-50}" \
    --hidden_dim "${HIDDEN_DIM:-128}" \
    --events_per_class "${EVENTS_PER_CLASS:-4}" \
    --samples_per_event "${SAMPLES_PER_EVENT:-4}" \
    --temperature "${TEMPERATURE:-0.10}" \
    --preserve_weight "${preserve_weight}" \
    --learning_rate "${LEARNING_RATE:-0.001}" \
    --weight_decay "${WEIGHT_DECAY:-0.0001}" \
    --topk "${TOPK:-8}" \
    --beta "${BETA:-5.0}" \
    --max_neighbors_per_event "${MAX_NEIGHBORS_PER_EVENT:-1}" \
    --seed "${SEED}"
}

run_smoke() {
  ensure_caches
  train_metric "RM5_A_smoke20_seed${SEED}" "${SMOKE_STEPS:-20}" "${SMOKE_EVAL_EVERY:-20}" rm5a 0.0
}

run_screen() {
  ensure_caches
  train_metric \
    "RM5_A_screen${OPTIMIZER_STEPS:-1500}_seed${SEED}" \
    "${OPTIMIZER_STEPS:-1500}" \
    "${EVAL_EVERY:-500}" \
    rm5a 0.0
}

run_b_smoke() {
  ensure_caches
  train_metric \
    "RM5_B_smoke20_seed${SEED}" \
    "${SMOKE_STEPS:-20}" \
    "${SMOKE_EVAL_EVERY:-20}" \
    rm5b "${PRESERVE_WEIGHT:-5.0}"
}

run_b_screen() {
  ensure_caches
  train_metric \
    "RM5_B_screen${OPTIMIZER_STEPS:-1500}_seed${SEED}" \
    "${OPTIMIZER_STEPS:-1500}" \
    "${EVAL_EVERY:-500}" \
    rm5b "${PRESERVE_WEIGHT:-5.0}"
}

case "${MODE}" in
  preflight) preflight ;;
  smoke) run_smoke ;;
  screen) run_screen ;;
  b_smoke) run_b_smoke ;;
  b_screen) run_b_screen ;;
  b_all)
    preflight
    run_b_smoke
    run_b_screen
    ;;
  all)
    preflight
    run_smoke
    run_screen
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|smoke|screen|b_smoke|b_screen|b_all|all" >&2
    exit 2
    ;;
esac
