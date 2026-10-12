#!/usr/bin/env bash
set -euo pipefail

# CMR-V1 isolated runner.
# It only READS existing P3/P0/M3 artifacts and WRITES CMR_V1_ISOLATED outputs.
MODE="${1:-audit_complement}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -n "${CMR_V1_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CMR_V1_ENV}"
fi

SEED="${SEED:-3407}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
P0_ROOT="${P0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_POOLING}"
M3_ROOT="${M3_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_M3_PAIRWISE}"
RESULT_ROOT="${RESULT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_V1_ISOLATED}"

TRAIN_P0="${TRAIN_P0:-${P0_ROOT}/CMR_P0_train_pooling_seed${SEED}.pth}"
VAL_P0="${VAL_P0:-${P0_ROOT}/CMR_P0_val_pooling_seed${SEED}.pth}"
TRAIN_CANDIDATES="${TRAIN_CANDIDATES:-${M3_ROOT}/CMR_M3_train_top20_seed${SEED}.pth}"
VAL_CANDIDATES="${VAL_CANDIDATES:-${M3_ROOT}/CMR_M3_val_top20_seed${SEED}.pth}"

COMPLEMENT="${COMPLEMENT:-${RESULT_ROOT}/CMR_V1_ISO_A_top20_complement_seed${SEED}.json}"
SUPPORT="${SUPPORT:-${RESULT_ROOT}/CMR_V1_ISO_train_support_tokens_seed${SEED}.pth}"
TRAIN_QUERY="${TRAIN_QUERY:-${RESULT_ROOT}/CMR_V1_ISO_train_query_tokens_seed${SEED}.pth}"
VAL_QUERY="${VAL_QUERY:-${RESULT_ROOT}/CMR_V1_ISO_val_query_tokens_seed${SEED}.pth}"
SUPPORT_REGION="${SUPPORT_REGION:-${RESULT_ROOT}/CMR_V1_REGION_train_support_seed${SEED}.pth}"
TRAIN_QUERY_REGION="${TRAIN_QUERY_REGION:-${RESULT_ROOT}/CMR_V1_REGION_train_query_seed${SEED}.pth}"
VAL_QUERY_REGION="${VAL_QUERY_REGION:-${RESULT_ROOT}/CMR_V1_REGION_val_query_seed${SEED}.pth}"
METHOD="${METHOD:-emd}"
SET_RESULT_OVERRIDE="${SET_RESULT:-}"

preflight() {
  python -m py_compile \
    model/cmr_v1/set_matcher.py \
    model/cmr_v1/replay.py \
    model/cmr_v1/region_cache.py \
    script/standard_ML/cmr_v1/audit_top20_complement.py \
    script/standard_ML/cmr_v1/build_region_cache.py \
    script/standard_ML/cmr_v1/build_token_cache.py \
    script/standard_ML/cmr_v1/audit_set_verifier.py \
    script/standard_ML/cmr_v1/summarize_matchers.py
  python -m unittest \
    tests.cmr_v1.test_set_matcher tests.cmr_v1.test_replay \
    tests.cmr_v1.test_region_cache tests.cmr_v1.test_complement_audit -v
  echo "CMR-V1 isolated preflight passed."
}

require_audit_inputs() {
  for path in "${TRAIN_P0}" "${VAL_P0}" "${TRAIN_CANDIDATES}" "${VAL_CANDIDATES}"; do
    test -s "${path}" || { echo "Missing read-only upstream input: ${path}" >&2; exit 3; }
  done
}

require_build_inputs() {
  require_audit_inputs
  test -s "${P3_CKPT}" || { echo "Missing frozen P3 checkpoint: ${P3_CKPT}" >&2; exit 3; }
}

audit_complement() {
  require_audit_inputs
  mkdir -p "${RESULT_ROOT}"
  python -u script/standard_ML/cmr_v1/audit_top20_complement.py \
    --train_p0 "${TRAIN_P0}" --val_p0 "${VAL_P0}" \
    --train_candidates "${TRAIN_CANDIDATES}" --val_candidates "${VAL_CANDIDATES}" \
    --output "${COMPLEMENT}"
}

build_region_one() {
  local role="$1" p0="$2" candidates="$3" output="$4"
  if [[ -s "${output}" && "${FORCE_REGION_CACHE:-0}" != "1" ]]; then
    echo "Frozen region cache already exists: ${output}"
    return
  fi
  python -u script/standard_ML/cmr_v1/build_region_cache.py \
    --role "${role}" --dataset_path "${DATASET_ROOT}" --p0_cache "${p0}" \
    --candidate_cache "${candidates}" --p3_checkpoint "${P3_CKPT}" \
    --output "${output}" --device cuda:0 --num_workers "${NUM_WORKERS:-8}" \
    --warn_verified_fraction "${WARN_VERIFIED_FRACTION:-0.02}" \
    --max_verified_fraction "${MAX_VERIFIED_FRACTION:-0.05}" \
    --max_unresolved_fraction "${MAX_UNRESOLVED_FRACTION:-0.005}"
}

build_regions() {
  require_build_inputs
  mkdir -p "${RESULT_ROOT}"
  build_region_one support "${TRAIN_P0}" "${TRAIN_CANDIDATES}" "${SUPPORT_REGION}"
  build_region_one train_query "${TRAIN_P0}" "${TRAIN_CANDIDATES}" "${TRAIN_QUERY_REGION}"
  build_region_one validation_query "${VAL_P0}" "${VAL_CANDIDATES}" "${VAL_QUERY_REGION}"
}

build_one() {
  local role="$1" p0="$2" candidates="$3" region="$4" output="$5"
  if [[ -s "${output}" && "${FORCE_CACHE:-0}" != "1" ]]; then
    echo "Isolated cache already exists: ${output}"
    return
  fi
  python -u script/standard_ML/cmr_v1/build_token_cache.py \
    --role "${role}" --dataset_path "${DATASET_ROOT}" --p0_cache "${p0}" \
    --candidate_cache "${candidates}" --p3_checkpoint "${P3_CKPT}" \
    --region_cache "${region}" --output "${output}" \
    --device cuda:0 --num_workers "${NUM_WORKERS:-8}" \
    --storage_layout "${TOKEN_STORAGE_LAYOUT:-packed_list}"
}

build_tokens() {
  build_regions
  mkdir -p "${RESULT_ROOT}"
  build_one support "${TRAIN_P0}" "${TRAIN_CANDIDATES}" "${SUPPORT_REGION}" "${SUPPORT}"
  build_one train_query "${TRAIN_P0}" "${TRAIN_CANDIDATES}" "${TRAIN_QUERY_REGION}" "${TRAIN_QUERY}"
  build_one validation_query "${VAL_P0}" "${VAL_CANDIDATES}" "${VAL_QUERY_REGION}" "${VAL_QUERY}"
}

audit_set() {
  build_tokens
  local set_result="${SET_RESULT_OVERRIDE:-${RESULT_ROOT}/CMR_V1_ISO_B_${METHOD}_val_seed${SEED}.json}"
  python -u script/standard_ML/cmr_v1/audit_set_verifier.py \
    --support "${SUPPORT}" --train_queries "${TRAIN_QUERY}" \
    --validation_queries "${VAL_QUERY}" --output "${set_result}" \
    --method "${METHOD}" --device cuda:0 \
    --class_temperature "${CLASS_TEMPERATURE:-0.10}" \
    --neighbor_k "${NEIGHBOR_K:-3}" \
    --epsilon "${EMD_EPSILON:-0.07}" --iterations "${EMD_ITERATIONS:-40}" \
    --ridge "${FRN_RIDGE:-0.01}"
}

audit_all_matchers() {
  build_tokens
  local original_method="${METHOD}"
  for METHOD in chamfer dn4 emd frn; do
    SET_RESULT_OVERRIDE=""
    audit_set
  done
  METHOD="${original_method}"
  python -u script/standard_ML/cmr_v1/summarize_matchers.py \
    --inputs \
      "${RESULT_ROOT}/CMR_V1_ISO_B_chamfer_val_seed${SEED}.json" \
      "${RESULT_ROOT}/CMR_V1_ISO_B_dn4_val_seed${SEED}.json" \
      "${RESULT_ROOT}/CMR_V1_ISO_B_emd_val_seed${SEED}.json" \
      "${RESULT_ROOT}/CMR_V1_ISO_B_frn_val_seed${SEED}.json" \
    --output "${RESULT_ROOT}/CMR_V1_ISO_B_matcher_comparison_seed${SEED}.json"
}

case "${MODE}" in
  preflight) preflight ;;
  audit_complement) audit_complement ;;
  build_regions) build_regions ;;
  build) build_tokens ;;
  audit_set) audit_set ;;
  audit_all) audit_all_matchers ;;
  all) preflight; audit_complement; audit_all_matchers ;;
  *) echo "Unknown mode '${MODE}'; choose preflight|audit_complement|build_regions|build|audit_set|audit_all|all" >&2; exit 2 ;;
esac
