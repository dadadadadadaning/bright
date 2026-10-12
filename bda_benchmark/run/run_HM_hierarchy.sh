#!/usr/bin/env bash
set -euo pipefail

# Isolated hierarchical-memory entry point.  O-K0 trains no new network: it
# consumes the fixed Step-500 S2O branch and audits its building representation.

MODE="${1:-ok0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

SEED="${SEED:-3407}"
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
P3_CHECKPOINT="${P3_CHECKPOINT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
S2O_CHECKPOINT="${S2O_CHECKPOINT:-/root/shared-nvme/BRIGHT-checkpoints/P3_S2O_OCCURRENCE/P3_S2O_screen500_seed${SEED}/latest_resume.pth}"
P0_ROOT="${P0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_REPLAY_R1/CMR_POOLING}"
TRAIN_P0="${TRAIN_P0:-${P0_ROOT}/CMR_P0_train_pooling_seed${SEED}.pth}"
VAL_P0="${VAL_P0:-${P0_ROOT}/CMR_P0_val_pooling_seed${SEED}.pth}"
RESULT_ROOT="${RESULT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/HIERARCHICAL_MEMORY/OK0}"
TRAIN_KEY_CACHE="${TRAIN_KEY_CACHE:-${RESULT_ROOT}/HM_OK0_train_keys_seed${SEED}.pth}"
VAL_KEY_CACHE="${VAL_KEY_CACHE:-${RESULT_ROOT}/HM_OK0_validation_keys_seed${SEED}.pth}"
OK0_RESULT="${OK0_RESULT:-${RESULT_ROOT}/HM_OK0_train_locked_val_seed${SEED}.json}"

preflight() {
  python -m py_compile \
    model/hmemory/occurrence_key.py \
    script/standard_ML/hm_occurrence_runtime.py \
    script/standard_ML/build_HM_occurrence_cache.py \
    script/standard_ML/audit_HM_OK0_key.py
  python -m unittest tests.test_hm_occurrence_key -v
  echo "HM O-K0 hierarchical-memory preflight passed."
}

require_inputs() {
  for path in "${P3_CHECKPOINT}" "${S2O_CHECKPOINT}" "${TRAIN_P0}" "${VAL_P0}"; do
    test -s "${path}" || { echo "Missing O-K0 input: ${path}" >&2; exit 3; }
  done
  mkdir -p "${RESULT_ROOT}"
}

cache_train() {
  require_inputs
  python -u script/standard_ML/build_HM_occurrence_cache.py \
    --role train \
    --dataset_path "${DATASET_ROOT}" \
    --p0_cache "${TRAIN_P0}" \
    --p3_checkpoint "${P3_CHECKPOINT}" \
    --s2o_checkpoint "${S2O_CHECKPOINT}" \
    --output "${TRAIN_KEY_CACHE}" \
    --device cuda:0 \
    --num_workers "${NUM_WORKERS:-8}" \
    --expected_s2o_step "${EXPECTED_S2O_STEP:-500}" \
    --temperatures "${OCCURRENCE_TEMPERATURES:-0.25,0.5,1.0}"
}

cache_validation() {
  require_inputs
  python -u script/standard_ML/build_HM_occurrence_cache.py \
    --role validation \
    --dataset_path "${DATASET_ROOT}" \
    --p0_cache "${VAL_P0}" \
    --p3_checkpoint "${P3_CHECKPOINT}" \
    --s2o_checkpoint "${S2O_CHECKPOINT}" \
    --output "${VAL_KEY_CACHE}" \
    --device cuda:0 \
    --num_workers "${NUM_WORKERS:-8}" \
    --expected_s2o_step "${EXPECTED_S2O_STEP:-500}" \
    --temperatures "${OCCURRENCE_TEMPERATURES:-0.25,0.5,1.0}"
}

audit() {
  test -s "${TRAIN_KEY_CACHE}" || { echo "Missing Train O-K0 cache: ${TRAIN_KEY_CACHE}" >&2; exit 4; }
  test -s "${VAL_KEY_CACHE}" || { echo "Missing Validation O-K0 cache: ${VAL_KEY_CACHE}" >&2; exit 4; }
  python -u script/standard_ML/audit_HM_OK0_key.py \
    --train_key_cache "${TRAIN_KEY_CACHE}" \
    --validation_key_cache "${VAL_KEY_CACHE}" \
    --train_p0_cache "${TRAIN_P0}" \
    --validation_p0_cache "${VAL_P0}" \
    --output "${OK0_RESULT}" \
    --seed "${SEED}" \
    --cap_per_event_class "${CAP_PER_EVENT_CLASS:-500}" \
    --event_probe_cap "${EVENT_PROBE_CAP:-1000}" \
    --relation_pairs "${RELATION_PAIRS:-60000}" \
    --minimum_global_auc "${MINIMUM_GLOBAL_AUC:-0.70}" \
    --minimum_event_auc "${MINIMUM_EVENT_AUC:-0.60}" \
    --minimum_consistent_events "${MINIMUM_CONSISTENT_EVENTS:-10}" \
    --minimum_auc_gain "${MINIMUM_AUC_GAIN:-0.05}" \
    --maximum_event_probe_increase "${MAXIMUM_EVENT_PROBE_INCREASE:-0.02}"
}

ok0() {
  preflight
  cache_train
  cache_validation
  audit
}

case "${MODE}" in
  preflight) preflight ;;
  cache_train) cache_train ;;
  cache_validation) cache_validation ;;
  audit) audit ;;
  ok0|all) ok0 ;;
  *) echo "Unknown mode '${MODE}'; choose preflight|cache_train|cache_validation|audit|ok0" >&2; exit 2 ;;
esac
