#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
VARIANT="${2:-m2_direct}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BDA_ROOT="$(cd "${HERE}/../.." && pwd)"
PROJECT_ROOT="$(cd "${BDA_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_PATH="${DATASET_PATH:-${BDA_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${BDA_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${BDA_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
PARENT_CKPT="${PARENT_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
V2_ROOT="${V2_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_SUPERPIXEL_EXEMPLAR_MEMORY_V2}"
METRIC_CKPT="${METRIC_CKPT:-${V2_ROOT}/metric_key_step5000.pth}"
MEMORY_PATH="${MEMORY_PATH:-${V2_ROOT}/train_superpixel_memory.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_RESIDUAL_UTILITY_VERIFIER_V1}"
RESULT_ROOT="${RESULT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_RESIDUAL_UTILITY_VERIFIER_V1}"
TRAIN_TARGETS="${TRAIN_TARGETS:-${OUTPUT_ROOT}/train_metric_targets_cross_event.pt}"
VAL_TARGETS="${VAL_TARGETS:-${OUTPUT_ROOT}/val450_metric_targets_readonly.pt}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"

checkpoint_for() { echo "${OUTPUT_ROOT}/${1}_logistic_seed${SEED:-3407}.pth"; }
result_for() { echo "${RESULT_ROOT}/${1}_val450.json"; }

preflight() {
  "${PYTHON_BIN}" -m py_compile \
    "${HERE}/core.py" "${HERE}/build_metric_targets.py" "${HERE}/train_verifier.py" \
    "${HERE}/evaluate_val450.py" "${HERE}/compare.py"
  "${PYTHON_BIN}" -m pytest "${HERE}/test_core.py" -q
  test -s "${PARENT_CKPT}" || { echo "Missing P3 checkpoint: ${PARENT_CKPT}" >&2; exit 3; }
  test -s "${METRIC_CKPT}" || { echo "Missing Metric checkpoint: ${METRIC_CKPT}" >&2; exit 3; }
  test -s "${MEMORY_PATH}" || { echo "Missing class Memory: ${MEMORY_PATH}" >&2; exit 3; }
  case "$(basename "${TRAIN_LIST}") $(basename "${VAL_LIST}")" in *test*) echo "Test access forbidden" >&2; exit 4;; esac
  echo "Residual Utility Verifier V1 preflight passed. Test access: FORBIDDEN."
}

build_targets() {
  local split="$1" list="$2" output="$3" expected="$4" expected_miou="${5:--1}"
  if [[ -s "${output}" && "${FORCE:-0}" != "1" ]]; then
    echo "Completed target cache exists; skipped: ${output}"
    return
  fi
  mkdir -p "$(dirname "${output}")"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_residual_utility_verifier_v1.build_metric_targets \
    --dataset_path "${DATASET_PATH}" --data_list "${list}" --split "${split}" \
    --parent_ckpt "${PARENT_CKPT}" --metric "${METRIC_CKPT}" --memory "${MEMORY_PATH}" \
    --output "${output}" --expected_images "${expected}" \
    --expected_parent_miou "${expected_miou}" --parent_tolerance_pp "${PARENT_TOLERANCE_PP:-0.02}" \
    --segments "${SEGMENTS:-400}" --compactness "${COMPACTNESS:-10}" \
    --alignment_radius "${ALIGNMENT_RADIUS:-1}" --topk "${TOPK:-8}" \
    --num_workers "${NUM_WORKERS:-8}" --device "${DEVICE:-cuda:0}"
}

train_variant() {
  local variant="$1" checkpoint
  checkpoint="$(checkpoint_for "${variant}")"
  test -s "${TRAIN_TARGETS}" || { echo "Run build_train first: ${TRAIN_TARGETS}" >&2; exit 3; }
  mkdir -p "${OUTPUT_ROOT}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_residual_utility_verifier_v1.train_verifier \
    --targets "${TRAIN_TARGETS}" --variant "${variant}" --output "${checkpoint}" \
    --steps "${VERIFIER_STEPS:-1500}" --folds "${FOLDS:-5}" --lr "${VERIFIER_LR:-0.03}" \
    --maximum_acceptance "${MAX_ACCEPTANCE:-0.70}" --conflict_margin "${CONFLICT_MARGIN:-0.10}" \
    --seed "${SEED:-3407}"
}

evaluate_variant() {
  local variant="$1" checkpoint result
  checkpoint="$(checkpoint_for "${variant}")"; result="$(result_for "${variant}")"
  test -s "${VAL_TARGETS}" || { echo "Run build_val first: ${VAL_TARGETS}" >&2; exit 3; }
  test -s "${checkpoint}" || { echo "Run train ${variant} first: ${checkpoint}" >&2; exit 3; }
  mkdir -p "${RESULT_ROOT}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_residual_utility_verifier_v1.evaluate_val450 \
    --targets "${VAL_TARGETS}" --checkpoint "${checkpoint}" --output_json "${result}"
}

case "${MODE}" in
  preflight) preflight ;;
  build_train) preflight; build_targets train "${TRAIN_LIST}" "${TRAIN_TARGETS}" "${EXPECTED_TRAIN_IMAGES:-2918}" -1 ;;
  build_val) preflight; build_targets validation "${VAL_LIST}" "${VAL_TARGETS}" "${EXPECTED_VAL_IMAGES:-450}" "${EXPECTED_PARENT_MIOU:-0.6050195963527841}" ;;
  train) preflight; train_variant "${VARIANT}" ;;
  evaluate) preflight; evaluate_variant "${VARIANT}" ;;
  stage_m2)
    preflight; build_targets train "${TRAIN_LIST}" "${TRAIN_TARGETS}" "${EXPECTED_TRAIN_IMAGES:-2918}" -1
    train_variant m2_direct
    build_targets validation "${VAL_LIST}" "${VAL_TARGETS}" "${EXPECTED_VAL_IMAGES:-450}" "${EXPECTED_PARENT_MIOU:-0.6050195963527841}"
    evaluate_variant m2_direct
    ;;
  stage_m3)
    preflight; train_variant m2_union_basic; train_variant m3_class_memory
    evaluate_variant m2_union_basic; evaluate_variant m3_class_memory
    mkdir -p "${RESULT_ROOT}"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_residual_utility_verifier_v1.compare \
      --m2 "$(result_for m2_union_basic)" --candidate "$(result_for m3_class_memory)" \
      --output "${RESULT_ROOT}/m3_vs_m2_union_admission.json"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_residual_utility_verifier_v1.compare \
      --m2 "$(result_for m2_direct)" --candidate "$(result_for m3_class_memory)" \
      --output "${RESULT_ROOT}/m3_vs_m2_direct_admission.json"
    ;;
  compare_m3)
    mkdir -p "${RESULT_ROOT}"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_residual_utility_verifier_v1.compare \
      --m2 "$(result_for m2_union_basic)" --candidate "$(result_for m3_class_memory)" \
      --output "${RESULT_ROOT}/m3_vs_m2_union_admission.json"
    ;;
  m4)
    echo "M4 is intentionally locked. Run stage_m3 first; implement residual Utility Memory only if M3 passes admission." >&2
    exit 5
    ;;
  *) echo "Choose preflight|build_train|build_val|train <variant>|evaluate <variant>|stage_m2|stage_m3|compare_m3|m4" >&2; exit 2 ;;
esac
