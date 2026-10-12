#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-audit}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
P0_ROOT="${P0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_POOLING}"
TRAIN_P0="${TRAIN_P0:-${P0_ROOT}/CMR_P0_train_pooling_seed${SEED}.pth}"
VAL_P0="${VAL_P0:-${P0_ROOT}/CMR_P0_val_pooling_seed${SEED}.pth}"
M3_ROOT="${M3_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_M3_PAIRWISE}"
TRAIN_CANDIDATES="${TRAIN_CANDIDATES:-${M3_ROOT}/CMR_M3_train_top20_seed${SEED}.pth}"
VAL_CANDIDATES="${VAL_CANDIDATES:-${M3_ROOT}/CMR_M3_val_top20_seed${SEED}.pth}"
RESULT_ROOT="${RESULT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_V0_LOCAL_VERIFY}"
SUPPORT="${SUPPORT:-${RESULT_ROOT}/CMR_V0_train_damaged_support_tokens_seed${SEED}.pth}"
TRAIN_QUERY="${TRAIN_QUERY:-${RESULT_ROOT}/CMR_V0_train_proposal_tokens_seed${SEED}.pth}"
VAL_QUERY="${VAL_QUERY:-${RESULT_ROOT}/CMR_V0_val_proposal_tokens_seed${SEED}.pth}"
RESULT="${RESULT:-${RESULT_ROOT}/CMR_V0_local_verifier_val_seed${SEED}.json}"

preflight() {
  python -c 'import imageio.v2' || {
    echo "Missing Python dependency: imageio (required by the existing P4 dataset loader)." >&2
    echo "Install the repository requirement with: python -m pip install imageio==2.34.0" >&2
    exit 3
  }
  python -m py_compile model/cmr/local_verifier.py script/standard_ML/build_CMR_V0_token_cache.py script/standard_ML/audit_CMR_V0_local_verifier.py
  python -m unittest tests.test_cmr_v0_local_verifier tests.test_im1_matching_reliability -v
  echo "CMR-V0 local verifier preflight passed."
}

require_inputs() {
  for path in "${P3_CKPT}" "${TRAIN_P0}" "${VAL_P0}" "${TRAIN_CANDIDATES}" "${VAL_CANDIDATES}"; do
    test -s "${path}" || { echo "Missing V0 input: ${path}" >&2; exit 3; }
  done
}

build_one() {
  local role="$1" p0="$2" candidates="$3" output="$4"
  if [[ -s "${output}" && "${FORCE_CACHE:-0}" != "1" ]]; then
    echo "Cache already exists: ${output}"
    return
  fi
  python -u script/standard_ML/build_CMR_V0_token_cache.py \
    --role "${role}" --dataset_path "${DATASET_ROOT}" --p0_cache "${p0}" \
    --candidate_cache "${candidates}" --p3_checkpoint "${P3_CKPT}" \
    --output "${output}" --device cuda:0 --num_workers "${NUM_WORKERS:-8}"
}

build() {
  require_inputs
  mkdir -p "${RESULT_ROOT}"
  build_one support "${TRAIN_P0}" "${TRAIN_CANDIDATES}" "${SUPPORT}"
  build_one train_query "${TRAIN_P0}" "${TRAIN_CANDIDATES}" "${TRAIN_QUERY}"
  build_one validation_query "${VAL_P0}" "${VAL_CANDIDATES}" "${VAL_QUERY}"
}

audit() {
  build
  python -u script/standard_ML/audit_CMR_V0_local_verifier.py \
    --support "${SUPPORT}" --train_queries "${TRAIN_QUERY}" \
    --validation_queries "${VAL_QUERY}" --output "${RESULT}"
}

case "${MODE}" in
  preflight) preflight ;;
  build) build ;;
  audit) audit ;;
  all) preflight; audit ;;
  *) echo "Unknown mode '${MODE}'; choose preflight|build|audit|all" >&2; exit 2 ;;
esac
