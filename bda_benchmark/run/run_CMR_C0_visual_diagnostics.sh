#!/usr/bin/env bash
set -euo pipefail

# Read-only diagnosis of the frozen CMR-C0 Validation artifacts.
MODE="${1:-preflight}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ -n "${CMR_C0_VIS_ENV:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CMR_C0_VIS_ENV}"
fi

SEED="${SEED:-3407}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
C0_ROOT="${C0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_C0_CRITICAL_ISOLATED}"
V1_REGION_ROOT="${V1_REGION_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_V1_ISOLATED}"
VIS_ROOT="${VIS_ROOT:-${C0_ROOT}/visual_diagnostics_seed${SEED}}"

SUPPORT="${SUPPORT:-${C0_ROOT}/CMR_C0_train_support_cases_seed${SEED}.pth}"
VAL_QUERY="${VAL_QUERY:-${C0_ROOT}/CMR_C0_val_query_cases_seed${SEED}.pth}"
AUDIT_RESULT="${AUDIT_RESULT:-${C0_ROOT}/CMR_C0_critical_audit_seed${SEED}.json}"
SUPPORT_REGION="${SUPPORT_REGION:-${V1_REGION_ROOT}/CMR_V1_REGION_train_support_seed${SEED}.pth}"
VAL_QUERY_REGION="${VAL_QUERY_REGION:-${V1_REGION_ROOT}/CMR_V1_REGION_val_query_seed${SEED}.pth}"
SCORE_CACHE="${SCORE_CACHE:-${VIS_ROOT}/CMR_C0_visual_scores_seed${SEED}.json}"
OBSERVATION_CSV="${OBSERVATION_CSV:-${VIS_ROOT}/CMR_C0_VIS_manual_observations.csv}"
OBSERVATION_SUMMARY="${OBSERVATION_SUMMARY:-${VIS_ROOT}/CMR_C0_VIS_manual_summary.json}"

preflight() {
  python -c "import imageio, matplotlib, numpy, torch" || {
    echo "Missing visual dependency in the active environment." >&2
    echo "Activate bright5090, then install imageio and matplotlib in that environment." >&2
    exit 4
  }
  python -m py_compile \
    model/cmr_c0/visual_diagnostics.py \
    script/standard_ML/cmr_c0/visualize_critical_regions.py \
    script/standard_ML/cmr_c0/summarize_visual_observations.py
  python -m unittest tests.cmr_c0.test_visual_diagnostics -v
  echo "CMR-C0 visual-diagnostic preflight passed."
}

require_inputs() {
  for path in \
    "${SUPPORT}" "${VAL_QUERY}" "${AUDIT_RESULT}" \
    "${SUPPORT_REGION}" "${VAL_QUERY_REGION}"; do
    test -s "${path}" || { echo "Missing frozen diagnostic input: ${path}" >&2; exit 3; }
  done
  test -d "${DATASET_ROOT}/pre-event" || {
    echo "Missing pre-event image directory: ${DATASET_ROOT}/pre-event" >&2; exit 3;
  }
  test -d "${DATASET_ROOT}/post-event" || {
    echo "Missing post-event image directory: ${DATASET_ROOT}/post-event" >&2; exit 3;
  }
}

visualize() {
  require_inputs
  mkdir -p "${VIS_ROOT}"
  extra_args=()
  [[ "${FORCE_VIS_SCORES:-0}" == "1" ]] && extra_args+=(--force_scores)
  [[ "${SAVE_VIS_PDF:-0}" == "0" ]] && extra_args+=(--no_pdf)
  python -u script/standard_ML/cmr_c0/visualize_critical_regions.py \
    --support "${SUPPORT}" \
    --validation_queries "${VAL_QUERY}" \
    --support_region "${SUPPORT_REGION}" \
    --validation_region "${VAL_QUERY_REGION}" \
    --audit_json "${AUDIT_RESULT}" \
    --dataset_root "${DATASET_ROOT}" \
    --output_dir "${VIS_ROOT}" \
    --score_cache "${SCORE_CACHE}" \
    --device cuda:0 \
    --neighbor_k "${NEIGHBOR_K:-3}" \
    --class_temperature "${CLASS_TEMPERATURE:-0.10}" \
    --per_event "${VIS_PER_EVENT:-2}" \
    --maximum_per_group "${VIS_MAXIMUM_PER_GROUP:-24}" \
    "${extra_args[@]}"
}

summarize() {
  test -s "${OBSERVATION_CSV}" || {
    echo "Missing manual observation CSV: ${OBSERVATION_CSV}" >&2; exit 3;
  }
  python -u script/standard_ML/cmr_c0/summarize_visual_observations.py \
    --input "${OBSERVATION_CSV}" --output "${OBSERVATION_SUMMARY}"
}

case "${MODE}" in
  preflight) preflight ;;
  visualize) visualize ;;
  summarize) summarize ;;
  all) preflight; visualize ;;
  *) echo "Unknown mode '${MODE}'; choose preflight|visualize|summarize|all" >&2; exit 2 ;;
esac
