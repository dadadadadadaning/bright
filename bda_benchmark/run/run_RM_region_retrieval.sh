#!/usr/bin/env bash
set -euo pipefail

# Validation-only RM0--RM6A controller.
# RM0: export a class-balanced train-only frozen-P3 region memory.
# RM1: parameter-free class-balanced visual kNN.
# RM2: add raw SAR mean/std and select its weight on validation only.
# RM3: same-event exclusion plus label-shuffle falsification controls.
# RM3E-A: class x event balanced train-only memory.
# RM3E-B: cap neighbours contributed by any one event.
# RM3E-C: replace absolute SAR mean/std with scene-relative statistics.
# RM4: audit whether confidence can separate useful Damaged rescues from
#      harmful Intact-to-Damaged proposals.  RM4 never changes P3 predictions.
# RM6-A: restore the natural train-region class prior after balanced retrieval;
#        tau is selected by train-memory leave-one-event-out only.

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
EXPECTED_P3_SHA256="${EXPECTED_P3_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/RM_REGION_RETRIEVAL}"
MEMORY="${MEMORY:-${OUTPUT_ROOT}/RM0_train_memory_seed${SEED}.pth}"
QUERIES="${QUERIES:-${OUTPUT_ROOT}/RM0_val_queries_seed${SEED}.pth}"
RM1_JSON="${RM1_JSON:-${OUTPUT_ROOT}/RM1_visual_knn_val_seed${SEED}.json}"
RM2_JSON="${RM2_JSON:-${OUTPUT_ROOT}/RM2_sar_stats_val_seed${SEED}.json}"
RM3_JSON="${RM3_JSON:-${OUTPUT_ROOT}/RM3_controls_val_seed${SEED}.json}"
RM3E_MEMORY="${RM3E_MEMORY:-${OUTPUT_ROOT}/RM3E_class_event_train_memory_seed${SEED}.pth}"
RM3E_QUERIES="${RM3E_QUERIES:-${OUTPUT_ROOT}/RM3E_val_queries_seed${SEED}.pth}"
RM3E_A_JSON="${RM3E_A_JSON:-${OUTPUT_ROOT}/RM3E_A_event_balanced_memory_val_seed${SEED}.json}"
RM3E_B_JSON="${RM3E_B_JSON:-${OUTPUT_ROOT}/RM3E_B_event_diverse_knn_val_seed${SEED}.json}"
RM3E_C_JSON="${RM3E_C_JSON:-${OUTPUT_ROOT}/RM3E_C_scene_relative_sar_val_seed${SEED}.json}"
RM4_JSON="${RM4_JSON:-${OUTPUT_ROOT}/RM4_selective_damaged_confidence_val_seed${SEED}.json}"
RM6A_JSON="${RM6A_JSON:-${OUTPUT_ROOT}/RM6A_train_prior_corrected_val_seed${SEED}.json}"
RM6P_JSON="${RM6P_JSON:-${OUTPUT_ROOT}/RM6P_p3_anchored_memory_val_seed${SEED}.json}"

check_protocol() {
  test -s "${P3_CKPT}" || { echo "Missing original P3 checkpoint: ${P3_CKPT}" >&2; exit 3; }
  for path in "${TRAIN_LIST}" "${VAL_LIST}"; do
    test -s "${path}" || { echo "Missing split list: ${path}" >&2; exit 3; }
  done
  local observed
  observed="$(sha256sum "${P3_CKPT}" | awk '{print $1}')"
  [[ "${observed}" == "${EXPECTED_P3_SHA256}" ]] || {
    echo "P3 SHA256 mismatch: ${observed}" >&2
    echo "Expected: ${EXPECTED_P3_SHA256}" >&2
    exit 4
  }
}

preflight() {
  python -m py_compile \
    script/standard_ML/rm_region_retrieval_runtime.py \
    script/standard_ML/build_RM0_region_cache.py \
    script/standard_ML/audit_RM_region_retrieval.py
  python -m unittest tests.test_rm_region_retrieval -v
  if [[ -f "${P3_CKPT}" ]]; then check_protocol; fi
  echo "RM0--RM6A preflight passed."
}

build_cache() {
  local role="$1"
  local list="$2"
  local output="$3"
  local balance_mode="${4:-class}"
  if [[ -s "${output}" && "${FORCE_CACHE:-0}" != "1" ]]; then
    echo "Cache already exists: ${output}"
    return 0
  fi
  mkdir -p "$(dirname "${output}")"
  python -u script/standard_ML/build_RM0_region_cache.py \
    --role "${role}" \
    --dataset_path "${DATASET_ROOT}" \
    --data_list_path "${list}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --val_data_list_path "${VAL_LIST}" \
    --p3_checkpoint "${P3_CKPT}" \
    --output "${output}" \
    --device cuda:0 \
    --num_workers "${NUM_WORKERS:-8}" \
    --min_area "${MIN_AREA:-32}" \
    --max_area "${MAX_AREA:-300000}" \
    --min_building_coverage "${MIN_BUILDING_COVERAGE:-0.50}" \
    --min_class_purity "${MIN_CLASS_PURITY:-0.75}" \
    --capacity_per_class "${CAPACITY_PER_CLASS:-2000}" \
    --balance_mode "${balance_mode}" \
    --capacity_per_event_class "${CAPACITY_PER_EVENT_CLASS:-200}" \
    --seed "${SEED}"
}

ensure_caches() {
  check_protocol
  build_cache memory "${TRAIN_LIST}" "${MEMORY}" class
  build_cache query "${VAL_LIST}" "${QUERIES}" class
}

ensure_rm3e_caches() {
  check_protocol
  build_cache memory "${TRAIN_LIST}" "${RM3E_MEMORY}" class_event
  build_cache query "${VAL_LIST}" "${RM3E_QUERIES}" class_event
}

require_passed_gate() {
  local path="$1"
  test -s "${path}" || { echo "Missing prerequisite result: ${path}" >&2; exit 5; }
  python -c 'import json,sys; r=json.load(open(sys.argv[1], encoding="utf-8")); assert r.get("passed") is True, f"gate failed: {sys.argv[1]}"' "${path}"
}

run_rm1() {
  ensure_caches
  python -u script/standard_ML/audit_RM_region_retrieval.py \
    --stage rm1 --memory "${MEMORY}" --queries "${QUERIES}" \
    --output "${RM1_JSON}" --device cuda:0 \
    --topk "${TOPK:-8}" --beta "${BETA:-5.0}" --seed "${SEED}"
}

run_rm2() {
  ensure_caches
  python -u script/standard_ML/audit_RM_region_retrieval.py \
    --stage rm2 --memory "${MEMORY}" --queries "${QUERIES}" \
    --output "${RM2_JSON}" --device cuda:0 \
    --topk "${TOPK:-8}" --beta "${BETA:-5.0}" \
    --raw_weights "${RAW_WEIGHTS:-0,0.25,0.5,1.0}" --seed "${SEED}"
}

run_rm3() {
  ensure_caches
  test -s "${RM2_JSON}" || run_rm2
  local selected
  selected="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_raw_weight"])' "${RM2_JSON}")"
  python -u script/standard_ML/audit_RM_region_retrieval.py \
    --stage rm3 --memory "${MEMORY}" --queries "${QUERIES}" \
    --output "${RM3_JSON}" --device cuda:0 \
    --topk "${TOPK:-8}" --beta "${BETA:-5.0}" \
    --selected_raw_weight "${selected}" \
    --shuffle_repeats "${SHUFFLE_REPEATS:-20}" --seed "${SEED}"
}

run_rm3e_a() {
  ensure_rm3e_caches
  python -u script/standard_ML/audit_RM_region_retrieval.py \
    --stage rm3e_a --memory "${RM3E_MEMORY}" --queries "${RM3E_QUERIES}" \
    --output "${RM3E_A_JSON}" --device cuda:0 \
    --topk "${TOPK:-8}" --beta "${BETA:-5.0}" \
    --selected_raw_weight "${RM3E_ABS_RAW_WEIGHT:-0.5}" --seed "${SEED}"
}

run_rm3e_b() {
  ensure_rm3e_caches
  if [[ "${REQUIRE_RM3E_GATES:-0}" == "1" ]]; then require_passed_gate "${RM3E_A_JSON}"; fi
  python -u script/standard_ML/audit_RM_region_retrieval.py \
    --stage rm3e_b --memory "${RM3E_MEMORY}" --queries "${RM3E_QUERIES}" \
    --output "${RM3E_B_JSON}" --device cuda:0 \
    --topk "${TOPK:-8}" --beta "${BETA:-5.0}" \
    --selected_raw_weight "${RM3E_ABS_RAW_WEIGHT:-0.5}" \
    --max_neighbors_per_event "${MAX_NEIGHBORS_PER_EVENT:-1}" --seed "${SEED}"
}

run_rm3e_c() {
  ensure_rm3e_caches
  if [[ "${REQUIRE_RM3E_GATES:-0}" == "1" ]]; then require_passed_gate "${RM3E_B_JSON}"; fi
  python -u script/standard_ML/audit_RM_region_retrieval.py \
    --stage rm3e_c --memory "${RM3E_MEMORY}" --queries "${RM3E_QUERIES}" \
    --output "${RM3E_C_JSON}" --device cuda:0 \
    --topk "${TOPK:-8}" --beta "${BETA:-5.0}" \
    --raw_weights "${RELATIVE_RAW_WEIGHTS:-0,0.25,0.5,1.0}" \
    --selected_raw_weight "${RM3E_ABS_RAW_WEIGHT:-0.5}" \
    --max_neighbors_per_event "${MAX_NEIGHBORS_PER_EVENT:-1}" \
    --shuffle_repeats "${SHUFFLE_REPEATS:-20}" --seed "${SEED}"
}

run_rm4() {
  ensure_rm3e_caches
  test -s "${RM3E_C_JSON}" || run_rm3e_c
  local selected
  selected="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_raw_weight"])' "${RM3E_C_JSON}")"
  python -u script/standard_ML/audit_RM_region_retrieval.py \
    --stage rm4 --memory "${RM3E_MEMORY}" --queries "${RM3E_QUERIES}" \
    --output "${RM4_JSON}" --device cuda:0 \
    --topk "${TOPK:-8}" --beta "${BETA:-5.0}" \
    --selected_raw_weight "${selected}" \
    --max_neighbors_per_event "${MAX_NEIGHBORS_PER_EVENT:-1}" --seed "${SEED}"
}

run_rm6a() {
  ensure_rm3e_caches
  python -u script/standard_ML/audit_RM_region_retrieval.py \
    --stage rm6a --memory "${RM3E_MEMORY}" --queries "${RM3E_QUERIES}" \
    --output "${RM6A_JSON}" --device cuda:0 \
    --topk "${TOPK:-8}" --beta "${BETA:-5.0}" \
    --selected_raw_weight "${RM6_RAW_WEIGHT:-1.0}" \
    --max_neighbors_per_event "${MAX_NEIGHBORS_PER_EVENT:-1}" \
    --prior_taus "${PRIOR_TAUS:-0,0.25,0.5,0.75,1.0}" \
    --prior_smoothing "${PRIOR_SMOOTHING:-1.0}" --seed "${SEED}"
}

run_rm6p() {
  ensure_rm3e_caches
  python -u script/standard_ML/audit_RM_region_retrieval.py \
    --stage rm6p --memory "${RM3E_MEMORY}" --queries "${RM3E_QUERIES}" \
    --output "${RM6P_JSON}" --device cuda:0 \
    --topk "${TOPK:-8}" --beta "${BETA:-5.0}" \
    --selected_raw_weight "${RM6_RAW_WEIGHT:-1.0}" \
    --max_neighbors_per_event "${MAX_NEIGHBORS_PER_EVENT:-1}" \
    --memory_alphas "${MEMORY_ALPHAS:-0,0.01,0.025,0.05,0.1,0.2,0.3,0.5}" \
    --seed "${SEED}"
}

mkdir -p "${OUTPUT_ROOT}"
case "${MODE}" in
  preflight) preflight ;;
  rm0_build) ensure_caches ;;
  rm1_val) run_rm1 ;;
  rm2_val) run_rm2 ;;
  rm3_controls) run_rm3 ;;
  rm3e_build) ensure_rm3e_caches ;;
  rm3e_a) run_rm3e_a ;;
  rm3e_b) run_rm3e_b ;;
  rm3e_c) run_rm3e_c ;;
  rm4_confidence) run_rm4 ;;
  rm6a_prior) run_rm6a ;;
  rm6p_anchor) run_rm6p ;;
  rm3e_all)
    preflight
    run_rm3e_a
    run_rm3e_b
    run_rm3e_c
    ;;
  all)
    preflight
    run_rm1
    run_rm2
    run_rm3
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|rm0_build|rm1_val|rm2_val|rm3_controls|rm3e_build|rm3e_a|rm3e_b|rm3e_c|rm4_confidence|rm6a_prior|rm6p_anchor|rm3e_all|all" >&2
    exit 2
    ;;
esac
