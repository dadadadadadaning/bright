#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${BENCHMARK_ROOT}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${BENCHMARK_ROOT}:${PYTHONPATH:-}"
SEED="${SEED:-7341}"
D1_ROOT="${D1_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed7341}"
V21_JSON="${V21_JSON:-/root/data/BRIGHT-eval-results/SPI_CLASS_SAFE_MACU_V21/cs_macu_v21_three_arm_loeo_seed7341.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_DIRECTION_IMPACT_UTILITY_V1/seed7341}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_DIRECTION_IMPACT_UTILITY_V1}"
REPLAY="${OUTPUT_ROOT}/b1_oof_replay.pt"
RESULT="${RESULT_ROOT}/direction_impact_e0_e1_seed${SEED}.json"
UTILITY_GROUPS="${UTILITY_GROUPS:-spatial,stage2_role,rgb_edge,sar_texture,structure_relation}"

echo "Direction-Impact Utility Python: ${PYTHON_BIN}"

preflight() {
  "${PYTHON_BIN}" -m pytest -q \
    "${SCRIPT_DIR}/test_core.py"
  "${PYTHON_BIN}" - <<'PY'
import torch
print("torch:", torch.__version__)
PY
  echo "Direction-Impact Utility preflight passed"
}

replay() {
  "${PYTHON_BIN}" -u -m \
    bda_benchmark.experiments.spi_direction_impact_utility_v1.replay_b1_oof \
    --train-cache "${D1_ROOT}/meta_train.pt" \
    --calibration-cache "${D1_ROOT}/meta_calibration.pt" \
    --v21-json "${V21_JSON}" \
    --output "${REPLAY}" \
    --groups "${UTILITY_GROUPS}" \
    --steps "${LOEO_STEPS:-1000}" \
    --batch-size "${BATCH_SIZE:-2048}" \
    --learning-rate "${LEARNING_RATE:-1e-3}" \
    --maximum-reject-fraction "${MAXIMUM_REJECT_FRACTION:-0.70}" \
    --seed "${SEED}"
}

audit() {
  test -s "${REPLAY}" || { echo "Run replay first: missing ${REPLAY}"; exit 1; }
  "${PYTHON_BIN}" -u -m \
    bda_benchmark.experiments.spi_direction_impact_utility_v1.audit \
    --train-cache "${D1_ROOT}/meta_train.pt" \
    --calibration-cache "${D1_ROOT}/meta_calibration.pt" \
    --replay "${REPLAY}" \
    --v21-json "${V21_JSON}" \
    --output "${RESULT}"
}

case "${MODE}" in
  preflight) preflight ;;
  replay) replay ;;
  audit) audit ;;
  auto) preflight; replay; audit ;;
  *) echo "usage: $0 {preflight|replay|audit|auto}"; exit 2 ;;
esac
