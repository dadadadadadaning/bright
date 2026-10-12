#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${BENCHMARK_ROOT}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${BENCHMARK_ROOT}:${PYTHONPATH:-}"
SEED="${SEED:-7341}"; STEPS="${STEPS:-500}"; THRESHOLD="${THRESHOLD:-0.50}"
P3_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
DATASET_ROOT="${DATASET_ROOT:-${BENCHMARK_ROOT}/dataset}"
SPLIT_ROOT="${SPLIT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/meta_split_seed${SEED}}"
OLD_VARIANT="${OLD_VARIANT:-a4_local_match}"
OLD_CKPT="${OLD_CKPT:-/root/data/BRIGHT-checkpoints/SPI_SAFE_AFFECTED_MODEL_V2/${OLD_VARIANT}_step500_seed${SEED}/safe_ao_last.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_MARGIN_AWARE_AO_V21}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_MARGIN_AWARE_AO_V21}"
CHECKPOINT="${OUTPUT_ROOT}/seed${SEED}/margin_aware_step${STEPS}.pth"

preflight() {
  "${PYTHON_BIN}" -m py_compile "${BENCHMARK_ROOT}/model/margin_aware_affected.py" \
    "${BENCHMARK_ROOT}/model/DamageFormer_MarginAwareAO_V21.py" "${SCRIPT_DIR}/audit_candidate_scores.py" \
    "${SCRIPT_DIR}/train.py" "${SCRIPT_DIR}/evaluate.py"
  "${PYTHON_BIN}" -m pytest -q "${BENCHMARK_ROOT}/model/test_margin_aware_affected.py"
  echo "Margin-Aware AO V2.1 preflight passed"
}

case "${MODE}" in
  preflight) preflight ;;
  d0)
    preflight; test -s "${OLD_CKPT}" || { echo "Missing old checkpoint: ${OLD_CKPT}" >&2; exit 1; }
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_margin_aware_affected_v21.audit_candidate_scores \
      --checkpoint "${OLD_CKPT}" --dataset-path "${DATASET_ROOT}" --list-path "${SPLIT_ROOT}/meta_calibration.txt" \
      --output "${RESULT_ROOT}/candidate_score_d0_${OLD_VARIANT}_seed${SEED}.json" --device "${DEVICE:-cuda:0}" \
      --num-workers "${NUM_WORKERS:-1}" --maximum-per-image "${MAXIMUM_PER_IMAGE:-4096}"
    ;;
  smoke)
    preflight; mkdir -p "$(dirname "${CHECKPOINT}")"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_margin_aware_affected_v21.train \
      --base-checkpoint "${P3_CKPT}" --dataset-path "${DATASET_ROOT}" --train-list "${SPLIT_ROOT}/meta_train.txt" \
      --output "${OUTPUT_ROOT}/seed${SEED}/smoke.pth" --optimizer-steps 2 --batch-size 1 --accumulation-steps 1 \
      --num-workers 0 --device "${DEVICE:-cuda:0}" --seed "${SEED}"
    echo "Smoke passed. Meta-calibration/confirm/Val450/test were not accessed."
    ;;
  train)
    preflight; mkdir -p "$(dirname "${CHECKPOINT}")"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_margin_aware_affected_v21.train \
      --base-checkpoint "${P3_CKPT}" --dataset-path "${DATASET_ROOT}" --train-list "${SPLIT_ROOT}/meta_train.txt" \
      --output "${CHECKPOINT}" --optimizer-steps "${STEPS}" --batch-size "${BATCH_SIZE:-2}" \
      --accumulation-steps "${ACCUMULATION_STEPS:-8}" --num-workers "${NUM_WORKERS:-4}" \
      --learning-rate "${LEARNING_RATE:-1e-4}" --threshold "${THRESHOLD}" --device "${DEVICE:-cuda:0}" --seed "${SEED}"
    ;;
  evaluate)
    preflight; test -s "${CHECKPOINT}" || { echo "Missing checkpoint: ${CHECKPOINT}" >&2; exit 1; }; mkdir -p "${RESULT_ROOT}"
    expected="$(wc -l < "${SPLIT_ROOT}/meta_calibration.txt" | tr -d ' ')"
    "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_margin_aware_affected_v21.evaluate \
      --checkpoint "${CHECKPOINT}" --dataset-path "${DATASET_ROOT}" --list-path "${SPLIT_ROOT}/meta_calibration.txt" \
      --output "${RESULT_ROOT}/margin_aware_step${STEPS}_threshold${THRESHOLD}_seed${SEED}.json" \
      --threshold "${THRESHOLD}" --expected-images "${expected}" --device "${DEVICE:-cuda:0}" \
      --num-workers "${NUM_WORKERS:-1}" --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-2000}" --seed "${SEED}"
    ;;
  *) echo "usage: $0 {preflight|d0|smoke|train|evaluate}" >&2; exit 2 ;;
esac
