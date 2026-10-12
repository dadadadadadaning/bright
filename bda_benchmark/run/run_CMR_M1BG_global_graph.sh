#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-audit}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

SEED="${SEED:-3407}"
P0_ROOT="${P0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_POOLING}"
TRAIN_CACHE="${TRAIN_CACHE:-${P0_ROOT}/CMR_P0_train_pooling_seed${SEED}.pth}"
VAL_CACHE="${VAL_CACHE:-${P0_ROOT}/CMR_P0_val_pooling_seed${SEED}.pth}"
RESULT_ROOT="${RESULT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_M1BG_GLOBAL_GRAPH}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/CMR_M1BG_GLOBAL_GRAPH}"
GRAPH="${GRAPH:-${RESULT_ROOT}/CMR_M1BG_train_graph_top${GRAPH_TOPK:-20}_seed${SEED}.pth}"
AUDIT_JSON="${AUDIT_JSON:-${RESULT_ROOT}/CMR_M1BG_graph_audit_top${GRAPH_TOPK:-20}_seed${SEED}.json}"

preflight() {
  python -m py_compile \
    script/standard_ML/cmr_global_graph_utils.py \
    script/standard_ML/audit_CMR_M1BG_graph.py \
    script/standard_ML/train_CMR_M1BG_graph.py
  python -m unittest \
    tests.test_cmr_pooling \
    tests.test_cmr_m1_metric \
    tests.test_cmr_m1bg_global_graph -v
  echo "CMR-M1BG global-graph preflight passed."
}

ensure_caches() {
  if [[ ! -s "${TRAIN_CACHE}" || ! -s "${VAL_CACHE}" ]]; then
    bash run/run_CMR_pooling_audit.sh build
  fi
  test -s "${TRAIN_CACHE}" || { echo "Missing Train cache: ${TRAIN_CACHE}" >&2; exit 3; }
  test -s "${VAL_CACHE}" || { echo "Missing Validation cache: ${VAL_CACHE}" >&2; exit 3; }
}

audit_graph() {
  ensure_caches
  mkdir -p "${RESULT_ROOT}"
  python -u script/standard_ML/audit_CMR_M1BG_graph.py \
    --train_cache "${TRAIN_CACHE}" \
    --graph_output "${GRAPH}" \
    --audit_output "${AUDIT_JSON}" \
    --device cuda:0 \
    --topk "${GRAPH_TOPK:-20}" \
    --max_per_event "${MAX_NEIGHBORS_PER_EVENT:-2}" \
    --chunk_size "${GRAPH_CHUNK:-512}" \
    --seed "${SEED}"
}

require_passed_graph() {
  test -s "${GRAPH}" || { echo "Missing graph: ${GRAPH}" >&2; exit 4; }
  test -s "${AUDIT_JSON}" || { echo "Missing graph audit: ${AUDIT_JSON}" >&2; exit 4; }
  python -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("passed") is True else 5)' "${AUDIT_JSON}" || {
    echo "CMR-M1BG graph audit failed; training is blocked." >&2
    exit 5
  }
}

train_graph_metric() {
  ensure_caches
  require_passed_graph
  local epochs="${EPOCHS:-30}"
  local output="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/CMR_M1BG_top${GRAPH_TOPK:-20}_epochs${epochs}_seed${SEED}}"
  mkdir -p "${output}"
  python -u script/standard_ML/train_CMR_M1BG_graph.py \
    --train_cache "${TRAIN_CACHE}" \
    --val_cache "${VAL_CACHE}" \
    --graph "${GRAPH}" \
    --output_dir "${output}" \
    --device cuda:0 \
    --epochs "${epochs}" \
    --steps_per_epoch "${STEPS_PER_EPOCH:-500}" \
    --eval_every "${EVAL_EVERY:-2}" \
    --log_every "${LOG_EVERY:-50}" \
    --hidden_dim "${HIDDEN_DIM:-256}" \
    --anchors_per_batch "${ANCHORS_PER_BATCH:-24}" \
    --negatives_per_class "${NEGATIVES_PER_CLASS:-8}" \
    --negative_classes "${NEGATIVE_CLASSES:-0,2}" \
    --gate_protocol "${GATE_PROTOCOL:-m1bg}" \
    --preserve_weight "${PRESERVE_WEIGHT:-0}" \
    --preserve_samples_per_class "${PRESERVE_SAMPLES_PER_CLASS:-24}" \
    --temperature "${TEMPERATURE:-0.10}" \
    --learning_rate "${LEARNING_RATE:-0.001}" \
    --weight_decay "${WEIGHT_DECAY:-0.0001}" \
    --query_chunk "${QUERY_CHUNK:-1024}" \
    --project_batch_size "${PROJECT_BATCH_SIZE:-4096}" \
    --ap_k "${AP_K:-10}" \
    --relation_pairs "${RELATION_PAIRS:-200000}" \
    --minimum_damaged_gain "${MINIMUM_DAMAGED_GAIN:-0.01}" \
    --maximum_destroyed_drop "${MAXIMUM_DESTROYED_DROP:-0.02}" \
    --minimum_nonnegative_events "${MINIMUM_NONNEGATIVE_EVENTS:-10}" \
    --seed "${SEED}"
}

case "${MODE}" in
  preflight) preflight ;;
  audit) audit_graph ;;
  smoke)
    EPOCHS="${EPOCHS:-1}" STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-5}" EVAL_EVERY="${EVAL_EVERY:-1}" train_graph_metric
    ;;
  train) train_graph_metric ;;
  sequence)
    preflight
    audit_graph
    train_graph_metric
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|audit|smoke|train|sequence" >&2
    exit 2
    ;;
esac
