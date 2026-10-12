#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BDA_ROOT="$(cd "${HERE}/../.." && pwd)"
PROJECT_ROOT="$(cd "${BDA_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_PATH="${DATASET_PATH:-${BDA_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${BDA_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${BDA_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
PARENT_CKPT="${PARENT_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_SUPERPIXEL_EXEMPLAR_MEMORY_V2}"
RESULT_ROOT="${RESULT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_SUPERPIXEL_EXEMPLAR_MEMORY_V2}"
TRAIN_RECORDS="${TRAIN_RECORDS:-${OUTPUT_ROOT}/train_slic_records.pt}"
VAL_RECORDS="${VAL_RECORDS:-${OUTPUT_ROOT}/val_slic_records.pt}"
METRIC_CKPT="${METRIC_CKPT:-${OUTPUT_ROOT}/metric_key_step5000.pth}"
MEMORY_PATH="${MEMORY_PATH:-${OUTPUT_ROOT}/train_superpixel_memory.pt}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"

preflight() {
  "${PYTHON_BIN}" -m py_compile \
    "${HERE}/core.py" "${HERE}/build_records.py" "${HERE}/train_metric.py" \
    "${HERE}/build_memory.py" "${HERE}/audit_retrieval.py"
  "${PYTHON_BIN}" -m pytest "${HERE}/test_core.py" -q
  "${PYTHON_BIN}" - <<'PY'
import skimage
print("scikit-image", skimage.__version__)
PY
  test -s "${PARENT_CKPT}" || { echo "Missing P3 checkpoint: ${PARENT_CKPT}" >&2; exit 3; }
  echo "SPI Superpixel Exemplar Memory V2 preflight passed. Test access: FORBIDDEN."
}

build_records() {
  local list="$1" output="$2" mode="${3:-slic}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.build_records \
    --dataset_path "${DATASET_PATH}" --data_list "${list}" --parent_ckpt "${PARENT_CKPT}" \
    --output "${output}" --region_mode "${mode}" \
    --segments "${SEGMENTS:-400}" --compactness "${COMPACTNESS:-10}" \
    --alignment_radius "${ALIGNMENT_RADIUS:-1}" --purity "${PURITY:-0.80}" \
    --batch_size "${BATCH_SIZE:-1}" --num_workers "${NUM_WORKERS:-8}" --device "${DEVICE:-cuda:0}"
}

case "${MODE}" in
  preflight) preflight ;;
  build_train_records) preflight; build_records "${TRAIN_LIST}" "${TRAIN_RECORDS}" "${REGION_MODE:-slic}" ;;
  train_metric)
    preflight; test -s "${TRAIN_RECORDS}"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.train_metric \
      --records "${TRAIN_RECORDS}" --output "${METRIC_CKPT}" --steps "${STEPS:-5000}" \
      --batch_size "${METRIC_BATCH_SIZE:-96}" --device "${DEVICE:-cuda:0}"
    ;;
  build_memory)
    preflight; test -s "${TRAIN_RECORDS}"; test -s "${METRIC_CKPT}"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.build_memory \
      --records "${TRAIN_RECORDS}" --metric "${METRIC_CKPT}" --output "${MEMORY_PATH}" \
      --capacity_per_class "${MEMORY_PER_CLASS:-4096}" --event_cap "${EVENT_CAP:-256}" --device "${DEVICE:-cuda:0}"
    ;;
  audit_train_image|audit_train_event)
    preflight; mkdir -p "${RESULT_ROOT}"
    extra=(--exclude_same_image)
    [[ "${MODE}" == "audit_train_event" ]] && extra+=(--exclude_same_event)
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.audit_retrieval \
      --query_records "${TRAIN_RECORDS}" --metric "${METRIC_CKPT}" --memory "${MEMORY_PATH}" \
      --output_json "${RESULT_ROOT}/${MODE}.json" --topk "${TOPK:-8}" \
      --seed "${SEED:-3407}" --shuffle_repeats "${SHUFFLE_REPEATS:-10}" \
      --device "${DEVICE:-cuda:0}" "${extra[@]}"
    ;;
  build_val_records) preflight; build_records "${VAL_LIST}" "${VAL_RECORDS}" "${REGION_MODE:-slic}" ;;
  audit_val|audit_val_event)
    preflight; mkdir -p "${RESULT_ROOT}"
    extra=(--exclude_same_image)
    [[ "${MODE}" == "audit_val_event" ]] && extra+=(--exclude_same_event)
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.audit_retrieval \
      --query_records "${VAL_RECORDS}" --metric "${METRIC_CKPT}" --memory "${MEMORY_PATH}" \
      --output_json "${RESULT_ROOT}/${MODE}.json" --topk "${TOPK:-8}" \
      --seed "${SEED:-3407}" --shuffle_repeats "${SHUFFLE_REPEATS:-10}" \
      --device "${DEVICE:-cuda:0}" "${extra[@]}"
    ;;
  stage1)
    preflight; build_records "${TRAIN_LIST}" "${TRAIN_RECORDS}" "${REGION_MODE:-slic}"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.train_metric \
      --records "${TRAIN_RECORDS}" --output "${METRIC_CKPT}" --steps "${STEPS:-5000}" --device "${DEVICE:-cuda:0}"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.build_memory \
      --records "${TRAIN_RECORDS}" --metric "${METRIC_CKPT}" --output "${MEMORY_PATH}" --device "${DEVICE:-cuda:0}"
    mkdir -p "${RESULT_ROOT}"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_superpixel_exemplar_memory_v2.audit_retrieval \
      --query_records "${TRAIN_RECORDS}" --metric "${METRIC_CKPT}" --memory "${MEMORY_PATH}" \
      --output_json "${RESULT_ROOT}/audit_train_event.json" --exclude_same_image --exclude_same_event \
      --seed "${SEED:-3407}" --shuffle_repeats "${SHUFFLE_REPEATS:-10}" --device "${DEVICE:-cuda:0}"
    ;;
  *) echo "Choose preflight|build_train_records|train_metric|build_memory|audit_train_image|audit_train_event|build_val_records|audit_val|audit_val_event|stage1" >&2; exit 2 ;;
esac
