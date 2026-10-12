#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
VARIANT="${VARIANT:-a5_confident_local}"
STEPS="${STEPS:-500}"
SEED="${SEED:-7341}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${BENCHMARK_ROOT}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${BENCHMARK_ROOT}:${PYTHONPATH:-}"

P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
DATASET_ROOT="${DATASET_ROOT:-${BENCHMARK_ROOT}/dataset}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_SAFE_AFFECTED_MODEL_V2}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_SAFE_AFFECTED_MODEL_V2}"
RUN_DIR="${OUTPUT_ROOT}/${VARIANT}_step${STEPS}_seed${SEED}"
CHECKPOINT="${RUN_DIR}/safe_ao_last.pth"

preflight() {
  "${PYTHON_BIN}" -m py_compile \
    "${BENCHMARK_ROOT}/model/safe_affected_plugins.py" \
    "${BENCHMARK_ROOT}/model/DamageFormer_SafeAO_V2.py" \
    "${SCRIPT_DIR}/train.py" "${SCRIPT_DIR}/evaluate.py"
  "${PYTHON_BIN}" -m pytest -q "${BENCHMARK_ROOT}/model/test_safe_affected_plugins.py"
  echo "Safe AO V2 preflight passed"
}

inputs() {
  for path in "${P3_CKPT}" "${DATASET_ROOT}" \
    "${SPLIT_ROOT}/meta_train.txt" "${SPLIT_ROOT}/meta_calibration.txt"; do
    test -e "${path}" || { echo "Missing input: ${path}" >&2; exit 1; }
  done
}

train_model() {
  inputs
  mkdir -p "${RUN_DIR}"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_safe_affected_model_v2.train \
    --variant "${VARIANT}" --base-checkpoint "${P3_CKPT}" \
    --dataset-path "${DATASET_ROOT}" --train-list "${SPLIT_ROOT}/meta_train.txt" \
    --output-dir "${RUN_DIR}" --device "${DEVICE:-cuda:0}" --seed "${SEED}" \
    --crop-size "${CROP_SIZE:-640}" --batch-size "${BATCH_SIZE:-2}" \
    --accumulation-steps "${ACCUMULATION_STEPS:-8}" --optimizer-steps "${STEPS}" \
    --num-workers "${NUM_WORKERS:-4}" --learning-rate "${LEARNING_RATE:-1e-4}" \
    --lambda-occurrence "${LAMBDA_OCCURRENCE:-0.50}" \
    --lambda-anchor "${LAMBDA_ANCHOR:-0.05}" \
    --log-every "${LOG_EVERY:-20}" --save-every "${SAVE_EVERY:-100}"
}

evaluate_shift() {
  local dx="$1" dy="$2" suffix="$3"
  test -s "${CHECKPOINT}" || { echo "Missing trained checkpoint: ${CHECKPOINT}" >&2; exit 1; }
  mkdir -p "${RESULT_ROOT}"
  local expected
  expected="$(wc -l < "${SPLIT_ROOT}/meta_calibration.txt" | tr -d ' ')"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_safe_affected_model_v2.evaluate \
    --checkpoint "${CHECKPOINT}" --dataset-path "${DATASET_ROOT}" \
    --list-path "${SPLIT_ROOT}/meta_calibration.txt" \
    --output "${RESULT_ROOT}/${VARIANT}_step${STEPS}_seed${SEED}${suffix}.json" \
    --device "${DEVICE:-cuda:0}" --num-workers "${EVAL_WORKERS:-2}" \
    --expected-images "${expected}" --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-2000}" \
    --shift-x "${dx}" --shift-y "${dy}" --seed "${SEED}"
}

echo "Safe AO V2 Python: ${PYTHON_BIN} | variant=${VARIANT}"
case "${MODE}" in
  preflight) preflight ;;
  smoke)
    preflight
    STEPS=2 RUN_DIR="${OUTPUT_ROOT}/${VARIANT}_smoke_seed${SEED}" train_model
    echo "Safe AO V2 smoke passed. Meta-calibration/confirm/Val450/test were not accessed."
    ;;
  pilot) preflight; train_model ;;
  evaluate) preflight; inputs; evaluate_shift 0 0 "" ;;
  robustness)
    preflight; inputs
    evaluate_shift 1 0 "_shift_xp1"
    evaluate_shift -1 0 "_shift_xm1"
    evaluate_shift 2 0 "_shift_xp2"
    evaluate_shift -2 0 "_shift_xm2"
    ;;
  auto) preflight; train_model; evaluate_shift 0 0 "" ;;
  *) echo "usage: $0 {preflight|smoke|pilot|evaluate|robustness|auto}" >&2; exit 2 ;;
esac
