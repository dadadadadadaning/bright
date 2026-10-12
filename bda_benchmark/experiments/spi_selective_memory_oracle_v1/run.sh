#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BDA_ROOT="$(cd "${HERE}/../.." && pwd)"
PROJECT_ROOT="$(cd "${BDA_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_PATH="${DATASET_PATH:-${BDA_ROOT}/dataset}"
VAL_LIST="${VAL_LIST:-${BDA_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
PARENT_CKPT="${PARENT_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
V2_ROOT="${V2_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_SUPERPIXEL_EXEMPLAR_MEMORY_V2}"
METRIC_CKPT="${METRIC_CKPT:-${V2_ROOT}/metric_key_step5000.pth}"
MEMORY_PATH="${MEMORY_PATH:-${V2_ROOT}/train_superpixel_memory.pt}"
RESULT_ROOT="${RESULT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_SELECTIVE_MEMORY_ORACLE_V1}"
DEFAULT_SUFFIX="standard"
[[ "${MODE}" == "val_event" || "${MODE}" == "complementarity_event" ]] && DEFAULT_SUFFIX="exclude_event"
[[ "${MODE}" == complementarity* ]] && DEFAULT_SUFFIX="complementarity_${DEFAULT_SUFFIX}"
OUTPUT_JSON="${OUTPUT_JSON:-${RESULT_ROOT}/val450_occurrence_oracle_${DEFAULT_SUFFIX}.json}"
TARGET_CACHE="${TARGET_CACHE:-${RESULT_ROOT}/val450_metric_targets_${DEFAULT_SUFFIX}.pt}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"

preflight() {
  "${PYTHON_BIN}" -m py_compile "${HERE}/oracle_core.py" "${HERE}/run_oracle.py" "${HERE}/summarize_complementarity.py"
  "${PYTHON_BIN}" -m pytest "${HERE}/test_oracle_core.py" -q
  "${PYTHON_BIN}" -m pytest "${BDA_ROOT}/experiments/spi_superpixel_exemplar_memory_v2/test_core.py" -q
  test -s "${PARENT_CKPT}" || { echo "Missing P3 checkpoint: ${PARENT_CKPT}" >&2; exit 3; }
  test -s "${METRIC_CKPT}" || { echo "Missing metric checkpoint: ${METRIC_CKPT}" >&2; exit 3; }
  test -s "${MEMORY_PATH}" || { echo "Missing Memory bank: ${MEMORY_PATH}" >&2; exit 3; }
  case "$(basename "${VAL_LIST}")" in *test*) echo "Test access forbidden" >&2; exit 4;; esac
  echo "Selective Memory Oracle V1 preflight passed. Test access: FORBIDDEN."
}

case "${MODE}" in
  preflight) preflight ;;
  val|val_event|complementarity|complementarity_event)
    preflight
    mkdir -p "${RESULT_ROOT}"
    extra=()
    [[ "${MODE}" == "val_event" || "${MODE}" == "complementarity_event" ]] && extra+=(--exclude_same_event)
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_selective_memory_oracle_v1.run_oracle \
      --dataset_path "${DATASET_PATH}" --val_list "${VAL_LIST}" \
      --parent_ckpt "${PARENT_CKPT}" --metric "${METRIC_CKPT}" --memory "${MEMORY_PATH}" \
      --output_json "${OUTPUT_JSON}" --target_cache "${TARGET_CACHE}" \
      --expected_images "${EXPECTED_IMAGES:-450}" \
      --expected_parent_miou "${EXPECTED_PARENT_MIOU:--1}" \
      --parent_tolerance_pp "${PARENT_TOLERANCE_PP:-0.02}" \
      --segments "${SEGMENTS:-400}" --compactness "${COMPACTNESS:-10}" \
      --alignment_radius "${ALIGNMENT_RADIUS:-1}" --topk "${TOPK:-8}" \
      --num_workers "${NUM_WORKERS:-8}" --device "${DEVICE:-cuda:0}" "${extra[@]}"
    ;;
  summarize)
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_selective_memory_oracle_v1.summarize_complementarity \
      "${RESULT_ROOT}/val450_occurrence_oracle_complementarity_standard.json" \
      "${RESULT_ROOT}/val450_occurrence_oracle_complementarity_exclude_event.json"
    ;;
  *) echo "Choose preflight|val|val_event|complementarity|complementarity_event|summarize" >&2; exit 2 ;;
esac
