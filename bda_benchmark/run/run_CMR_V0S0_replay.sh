#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-support}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
P0_ROOT="${P0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_POOLING}"
M3_ROOT="${M3_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_M3_PAIRWISE}"
RESULT_ROOT="${RESULT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_V0_REPLAY}"

preflight() {
  python -m py_compile script/standard_ML/cmr_v0_replay_runtime.py script/standard_ML/audit_CMR_V0S0_replay.py
  python -m unittest tests.test_cmr_v0s0_replay -v
  echo "CMR-V0S0 replay preflight passed."
}

run_role() {
  local role="$1" p0 candidates
  if [[ "${role}" == "validation_query" ]]; then
    p0="${VAL_P0:-${P0_ROOT}/CMR_P0_val_pooling_seed${SEED}.pth}"
    candidates="${VAL_CANDIDATES:-${M3_ROOT}/CMR_M3_val_top20_seed${SEED}.pth}"
  else
    p0="${TRAIN_P0:-${P0_ROOT}/CMR_P0_train_pooling_seed${SEED}.pth}"
    candidates="${TRAIN_CANDIDATES:-${M3_ROOT}/CMR_M3_train_top20_seed${SEED}.pth}"
  fi
  mkdir -p "${RESULT_ROOT}"
  python -u script/standard_ML/audit_CMR_V0S0_replay.py \
    --role "${role}" --dataset_path "${DATASET_ROOT}" --p0_cache "${p0}" \
    --candidate_cache "${candidates}" --p3_checkpoint "${P3_CKPT}" \
    --output "${RESULT_ROOT}/CMR_V0S0_${role}_seed${SEED}.json" \
    --device cuda:0 --num_workers "${NUM_WORKERS:-8}" \
    --sample_count "${SAMPLE_COUNT:-0}" --seed "${SEED}"
}

case "${MODE}" in
  preflight) preflight ;;
  support|train_query|validation_query) run_role "${MODE}" ;;
  all) preflight; run_role support; run_role train_query; run_role validation_query ;;
  *) echo "Unknown mode '${MODE}'; choose preflight|support|train_query|validation_query|all" >&2; exit 2 ;;
esac

