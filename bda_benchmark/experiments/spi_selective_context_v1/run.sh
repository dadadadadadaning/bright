#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|smoke|audit|test_locked) ;;
  *) echo "Choose preflight|smoke|audit|test_locked" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ "${MODE}" == "preflight" ]]; then
  bash -n "${SCRIPT_DIR}/run.sh"
  python -m py_compile \
    "${SCRIPT_DIR}/core.py" \
    "${SCRIPT_DIR}/audit.py" \
    "${SCRIPT_DIR}/evaluate_locked_test.py"
  python -m unittest discover -s "${SCRIPT_DIR}/tests" -p 'test_*.py' -v
  python -m unittest discover \
    -s "${PROJECT_ROOT}/tests/spi_region_feedback" -p 'test_*.py' -v
  echo "SPI selective-context V1 preflight passed"
  exit 0
fi

STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
SAR_STATS_JSON="${SAR_STATS_JSON:-${STORAGE_ROOT}/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
P3_CKPT="${P3_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_SELECTIVE_CONTEXT_VAL450}"
EXPECTED_SHA256="${EXPECTED_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"

if [[ "${MODE}" == "test_locked" ]]; then
  [[ "${ALLOW_TEST_ACCESS:-NO}" == "YES" ]] || {
    echo "Refusing Test access. Re-run with ALLOW_TEST_ACCESS=YES after freezing P3 + fixed S2." >&2
    exit 3
  }
  TEST_LIST="${TEST_LIST:-${DATASET_ROOT}/splitname/standard_ML/test_set.txt}"
  TEST_OUTPUT_ROOT="${TEST_OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_SELECTIVE_CONTEXT_TEST878_LOCKED_V1}"
  for required in "${DATASET_ROOT}" "${TEST_LIST}" "${SAR_STATS_JSON}" "${P3_CKPT}"; do
    test -e "${required}" || { echo "Missing input: ${required}" >&2; exit 1; }
  done
  observed="$(sha256sum "${P3_CKPT}" | awk '{print $1}')"
  [[ -z "${EXPECTED_SHA256}" || "${observed}" == "${EXPECTED_SHA256}" ]] || {
    echo "P3 identity mismatch: ${observed}" >&2
    exit 1
  }
  destination="${TEST_OUTPUT_ROOT}/P3_FIXED_S2_TEST878_LOCKED_seed3407.json"
  test ! -e "${destination}" || {
    echo "Locked Test result already exists; refusing a second run: ${destination}" >&2
    exit 4
  }
  mkdir -p "${TEST_OUTPUT_ROOT}"
  export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
  python -u -m bda_benchmark.experiments.spi_selective_context_v1.evaluate_locked_test \
    --checkpoint "${P3_CKPT}" \
    --dataset_path "${DATASET_ROOT}" \
    --test_list "${TEST_LIST}" \
    --sar_stats_json "${SAR_STATS_JSON}" \
    --output_json "${destination}" \
    --device cuda:0 \
    --num_workers "${NUM_WORKERS:-4}" \
    --expected_images 878 \
    --minimum_area 32 \
    --progress_every "${PROGRESS_EVERY:-20}" \
    --bootstrap_iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
    --seed 3407
  exit 0
fi

for required in "${DATASET_ROOT}" "${VAL_LIST}" "${SAR_STATS_JSON}" "${P3_CKPT}"; do
  test -e "${required}" || { echo "Missing input: ${required}" >&2; exit 1; }
done
observed="$(sha256sum "${P3_CKPT}" | awk '{print $1}')"
[[ -z "${EXPECTED_SHA256}" || "${observed}" == "${EXPECTED_SHA256}" ]] || {
  echo "P3 identity mismatch: ${observed}" >&2
  exit 1
}

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
mkdir -p "${OUTPUT_ROOT}"
max_images=0
tag=validation_full_v1
iterations="${BOOTSTRAP_ITERATIONS:-10000}"
if [[ "${MODE}" == "smoke" ]]; then
  max_images="${SMOKE_IMAGES:-2}"
  tag="smoke${max_images}_v1"
  iterations=10
fi

python -u -m bda_benchmark.experiments.spi_selective_context_v1.audit \
  --checkpoint "${P3_CKPT}" \
  --dataset_path "${DATASET_ROOT}" \
  --val_list "${VAL_LIST}" \
  --sar_stats_json "${SAR_STATS_JSON}" \
  --output_json "${OUTPUT_ROOT}/SPI_SELECTIVE_CONTEXT_${tag}_seed3407.json" \
  --device cuda:0 \
  --num_workers "${NUM_WORKERS:-4}" \
  --expected_images 450 \
  --max_images "${max_images}" \
  --minimum_area 32 \
  --progress_every "${PROGRESS_EVERY:-10}" \
  --profile_timing "${PROFILE_TIMING:-0}" \
  --bootstrap_iterations "${iterations}" \
  --seed 3407 \
  --expected_p3_miou 0.605020
