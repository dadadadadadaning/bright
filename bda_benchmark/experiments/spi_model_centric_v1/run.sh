#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
VARIANT="${VARIANT:-h1_hierarchy}"
SEED="${SEED:-7341}"
STEPS="${STEPS:-500}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${BENCHMARK_ROOT}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${BENCHMARK_ROOT}:${PYTHONPATH:-}"

P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
DATASET_ROOT="${DATASET_ROOT:-${BENCHMARK_ROOT}/dataset}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_MODEL_CENTRIC_V1}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_MODEL_CENTRIC_V1}"
RUN_DIR="${OUTPUT_ROOT}/${VARIANT}_step${STEPS}_seed${SEED}"
CHECKPOINT="${RUN_DIR}/plugin_last.pth"

preflight() {
  "${PYTHON_BIN}" -m py_compile \
    "${BENCHMARK_ROOT}/model/model_centric_plugins.py" \
    "${BENCHMARK_ROOT}/model/DamageFormer_ModelCentric_V1.py" \
    "${SCRIPT_DIR}/train.py" "${SCRIPT_DIR}/evaluate.py"
  "${PYTHON_BIN}" -m pytest -q "${BENCHMARK_ROOT}/model/test_model_centric_plugins.py"
  echo "Model-centric V1 preflight passed"
}

inputs() {
  for path in "${P3_CKPT}" "${DATASET_ROOT}" "${SPLIT_ROOT}/meta_train.txt"; do
    test -e "${path}" || { echo "Missing input: ${path}" >&2; exit 1; }
  done
}

train() {
  inputs
  mkdir -p "${RUN_DIR}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_model_centric_v1.train \
    --variant "${VARIANT}" --base-checkpoint "${P3_CKPT}" \
    --dataset-path "${DATASET_ROOT}" --train-list "${SPLIT_ROOT}/meta_train.txt" \
    --output-dir "${RUN_DIR}" --device "${DEVICE:-cuda:0}" \
    --seed "${SEED}" --crop-size "${CROP_SIZE:-640}" \
    --batch-size "${BATCH_SIZE:-2}" --accumulation-steps "${ACCUMULATION_STEPS:-8}" \
    --optimizer-steps "${STEPS}" --num-workers "${NUM_WORKERS:-4}" \
    --learning-rate "${LEARNING_RATE:-1e-4}" \
    --lambda-hierarchy "${LAMBDA_HIERARCHY:-0.25}" \
    --lambda-anchor "${LAMBDA_ANCHOR:-0.05}" \
    --log-every "${LOG_EVERY:-20}" --save-every "${SAVE_EVERY:-100}"
}

evaluate() {
  test -s "${CHECKPOINT}" || { echo "Run training first: missing ${CHECKPOINT}" >&2; exit 1; }
  test -s "${SPLIT_ROOT}/meta_calibration.txt" || { echo "Missing meta_calibration split" >&2; exit 1; }
  mkdir -p "${RESULT_ROOT}"
  local expected
  expected="$(wc -l < "${SPLIT_ROOT}/meta_calibration.txt" | tr -d ' ')"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_model_centric_v1.evaluate \
    --checkpoint "${CHECKPOINT}" --dataset-path "${DATASET_ROOT}" \
    --list-path "${SPLIT_ROOT}/meta_calibration.txt" \
    --output "${RESULT_ROOT}/${VARIANT}_step${STEPS}_seed${SEED}.json" \
    --device "${DEVICE:-cuda:0}" --num-workers "${NUM_WORKERS:-2}" \
    --expected-images "${expected}"
}

echo "Model-centric V1 Python: ${PYTHON_BIN} | variant=${VARIANT}"
case "${MODE}" in
  preflight) preflight ;;
  smoke)
    preflight
    STEPS=2 RUN_DIR="${OUTPUT_ROOT}/${VARIANT}_smoke_seed${SEED}" train
    echo "Real-data smoke passed. Meta-calibration/confirm/Val450/test were not accessed."
    ;;
  pilot) preflight; train ;;
  evaluate) preflight; evaluate ;;
  auto) preflight; train; evaluate ;;
  *) echo "usage: $0 {preflight|smoke|pilot|evaluate|auto}" >&2; exit 2 ;;
esac
