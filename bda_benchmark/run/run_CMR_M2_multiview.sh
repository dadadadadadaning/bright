#!/usr/bin/env bash
set -euo pipefail

# Train-locked, parameter-free CMR-M2 audit.  "Train" means LOEO selection of
# the reranking view/lambda; no neural network, P3 weight, or Memory item is
# updated.  Validation is queried once with the locked configuration.
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
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_M2_MULTIVIEW}"
RESULT="${RESULT:-${OUTPUT_ROOT}/CMR_M2_train_locked_multiview_val_seed${SEED}.json}"

preflight() {
  python -m py_compile \
    script/standard_ML/cmr_metric_utils.py \
    script/standard_ML/cmr_multiview_utils.py \
    script/standard_ML/audit_CMR_M2_multiview.py
  python -m unittest \
    tests.test_cmr_m1_metric \
    tests.test_cmr_m2_multiview -v
  echo "CMR-M2 multiview preflight passed."
}

ensure_caches() {
  if [[ ! -s "${TRAIN_CACHE}" || ! -s "${VAL_CACHE}" ]]; then
    bash run/run_CMR_pooling_audit.sh build
  fi
  test -s "${TRAIN_CACHE}" || { echo "Missing P0 train cache: ${TRAIN_CACHE}" >&2; exit 3; }
  test -s "${VAL_CACHE}" || { echo "Missing P0 validation cache: ${VAL_CACHE}" >&2; exit 3; }
}

run_audit() {
  ensure_caches
  mkdir -p "$(dirname "${RESULT}")"
  python -u script/standard_ML/audit_CMR_M2_multiview.py \
    --train_cache "${TRAIN_CACHE}" \
    --val_cache "${VAL_CACHE}" \
    --output "${RESULT}" \
    --device cuda:0 \
    --views "${RERANK_VIEWS:-hard_critical,exist_embedding}" \
    --lambdas "${RERANK_LAMBDAS:-0.10,0.25,0.50,1.0}" \
    --coarse_k "${COARSE_K:-20}" \
    --query_chunk "${QUERY_CHUNK:-512}" \
    --ap_k "${AP_K:-10}" \
    --minimum_damaged_gain "${MINIMUM_DAMAGED_GAIN:-0.01}" \
    --relation_pairs "${RELATION_PAIRS:-50000}" \
    --seed "${SEED}"
}

case "${MODE}" in
  preflight) preflight ;;
  audit) run_audit ;;
  all)
    preflight
    run_audit
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|audit|all" >&2
    exit 2
    ;;
esac

