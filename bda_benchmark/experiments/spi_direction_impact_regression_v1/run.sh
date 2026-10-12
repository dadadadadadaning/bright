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
E1_JSON="${E1_JSON:-/root/data/BRIGHT-eval-results/SPI_DIRECTION_IMPACT_UTILITY_V1_RETRY1/direction_impact_e0_e1_seed7341.json}"
V21_JSON="${V21_JSON:-/root/data/BRIGHT-eval-results/SPI_CLASS_SAFE_MACU_V21/cs_macu_v21_three_arm_loeo_seed7341.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_DIRECTION_IMPACT_REGRESSION_E2_V1/seed7341}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_DIRECTION_IMPACT_REGRESSION_E2_V1}"
E2_GROUPS="${E2_GROUPS:-spatial,stage2_role,rgb_edge,sar_texture,structure_relation}"
RESULT="${RESULT_ROOT}/direction_impact_e2_loeo_seed${SEED}.json"
OOF="${OUTPUT_ROOT}/direction_impact_e2_oof.pt"

echo "Direction-Impact E2 Python: ${PYTHON_BIN}"

preflight() {
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  "${PYTHON_BIN}" - <<'PY'
import torch
print("torch:", torch.__version__)
PY
  echo "Direction-Impact E2 preflight passed"
}

loeo() {
  "${PYTHON_BIN}" -u -m \
    bda_benchmark.experiments.spi_direction_impact_regression_v1.loeo \
    --train-cache "${D1_ROOT}/meta_train.pt" \
    --calibration-cache "${D1_ROOT}/meta_calibration.pt" \
    --e1-json "${E1_JSON}" \
    --v21-json "${V21_JSON}" \
    --output "${RESULT}" \
    --oof-output "${OOF}" \
    --groups "${E2_GROUPS}" \
    --steps "${LOEO_STEPS:-1000}" \
    --batch-size "${BATCH_SIZE:-2048}" \
    --learning-rate "${LEARNING_RATE:-1e-3}" \
    --impact-cap "${IMPACT_CAP:-8.0}" \
    --maximum-reject-fraction "${MAXIMUM_REJECT_FRACTION:-0.10}" \
    --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
    --seed "${SEED}"
}

case "${MODE}" in
  preflight) preflight ;;
  loeo) loeo ;;
  auto) preflight; loeo ;;
  *) echo "usage: $0 {preflight|loeo|auto}"; exit 2 ;;
esac

