#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-audit}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SEED="${SEED:-3407}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/RM_REGION_RETRIEVAL}"
MEMORY="${MEMORY:-${OUTPUT_ROOT}/RM3E_class_event_train_memory_seed${SEED}.pth}"
QUERIES="${QUERIES:-${OUTPUT_ROOT}/RM3E_val_queries_seed${SEED}.pth}"
RESULT="${RESULT:-${OUTPUT_ROOT}/RM7_dual_key_reject_val_seed${SEED}.json}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

preflight() {
  python -m py_compile \
    script/standard_ML/rm_region_retrieval_runtime.py \
    script/standard_ML/audit_RM7_dual_key_reject.py
  python -m unittest tests/test_rm_region_retrieval.py -v
  echo "RM7 dual-key reject preflight passed."
}

audit() {
  test -s "${MEMORY}"
  test -s "${QUERIES}"
  python -u script/standard_ML/audit_RM7_dual_key_reject.py \
    --memory "${MEMORY}" \
    --queries "${QUERIES}" \
    --output "${RESULT}" \
    --device cuda:0 \
    --topk "${TOPK:-8}" \
    --beta "${BETA:-5.0}" \
    --max_neighbors_per_event "${MAX_NEIGHBORS_PER_EVENT:-1}"
}

case "${MODE}" in
  preflight) preflight ;;
  audit) audit ;;
  *) echo "Unknown mode '${MODE}'; choose preflight|audit" >&2; exit 2 ;;
esac
