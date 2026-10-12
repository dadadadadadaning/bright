#!/usr/bin/env bash
set -euo pipefail

# Isolated CMR-M3 protocol:
#   M3O: frozen Raw Mean Top-20 oracle audit (no training)
#   M3P: fixed-budget bounded pairwise reranker (Validation once)
# P3, Raw Mean keys, Memory items and official Test remain untouched.
MODE="${1:-oracle}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

SEED="${SEED:-3407}"
P0_ROOT="${P0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_POOLING}"
TRAIN_CACHE="${TRAIN_CACHE:-${P0_ROOT}/CMR_P0_train_pooling_seed${SEED}.pth}"
VAL_CACHE="${VAL_CACHE:-${P0_ROOT}/CMR_P0_val_pooling_seed${SEED}.pth}"
RESULT_ROOT="${RESULT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_M3_PAIRWISE}"
CKPT_ROOT="${CKPT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/CMR_M3_PAIRWISE}"
TRAIN_CANDIDATES="${TRAIN_CANDIDATES:-${RESULT_ROOT}/CMR_M3_train_top20_seed${SEED}.pth}"
VAL_CANDIDATES="${VAL_CANDIDATES:-${RESULT_ROOT}/CMR_M3_val_top20_seed${SEED}.pth}"
ORACLE_RESULT="${ORACLE_RESULT:-${RESULT_ROOT}/CMR_M3O_oracle_seed${SEED}.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${CKPT_ROOT}/CMR_M3P_top20_epochs${EPOCHS:-6}_seed${SEED}}"

preflight() {
  python -m py_compile \
    model/cmr/pairwise.py \
    script/standard_ML/cmr_pairwise_utils.py \
    script/standard_ML/audit_CMR_M3_oracle.py \
    script/standard_ML/train_CMR_M3_pairwise.py
  python -m unittest tests.test_cmr_m3_pairwise -v
  echo "CMR-M3 oracle/pairwise preflight passed."
}

ensure_p0() {
  if [[ ! -s "${TRAIN_CACHE}" || ! -s "${VAL_CACHE}" ]]; then
    bash run/run_CMR_pooling_audit.sh build
  fi
  test -s "${TRAIN_CACHE}" || { echo "Missing Train cache: ${TRAIN_CACHE}" >&2; exit 3; }
  test -s "${VAL_CACHE}" || { echo "Missing Validation cache: ${VAL_CACHE}" >&2; exit 3; }
}

oracle_audit() {
  ensure_p0
  mkdir -p "${RESULT_ROOT}"
  python -u script/standard_ML/audit_CMR_M3_oracle.py \
    --train_cache "${TRAIN_CACHE}" \
    --val_cache "${VAL_CACHE}" \
    --train_candidates "${TRAIN_CANDIDATES}" \
    --val_candidates "${VAL_CANDIDATES}" \
    --output "${ORACLE_RESULT}" \
    --device cuda:0 \
    --coarse_k "${COARSE_K:-20}" \
    --query_chunk "${QUERY_CHUNK:-1024}" \
    --minimum_damaged_headroom "${MINIMUM_DAMAGED_HEADROOM:-0.05}" \
    --minimum_damaged_coverage "${MINIMUM_DAMAGED_COVERAGE:-0.95}"
}

train_pairwise() {
  test -s "${ORACLE_RESULT}" || oracle_audit
  test -s "${TRAIN_CANDIDATES}" || { echo "Missing Train candidates" >&2; exit 4; }
  test -s "${VAL_CANDIDATES}" || { echo "Missing Validation candidates" >&2; exit 4; }
  mkdir -p "${OUTPUT_DIR}"
  python -u script/standard_ML/train_CMR_M3_pairwise.py \
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
    --learning_rate "${LEARNING_RATE:-0.001}" \
    --weight_decay "${WEIGHT_DECAY:-0.0001}" \
    --seed "${SEED}"
}

case "${MODE}" in
  preflight) preflight ;;
  oracle) oracle_audit ;;
  train) train_pairwise ;;
  all)
    preflight
    oracle_audit
    train_pairwise
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|oracle|train|all" >&2
    exit 2
    ;;
esac

