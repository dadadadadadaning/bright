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
E2_JSON="${E2_JSON:-/root/data/BRIGHT-eval-results/SPI_DIRECTION_IMPACT_REGRESSION_E2_V1/direction_impact_e2_loeo_seed7341.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CLUSTER_OUTCOME_UTILITY_E21_V1/seed7341}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_CLUSTER_OUTCOME_UTILITY_E21_V1}"
OUTCOME_GROUPS="${OUTCOME_GROUPS:-spatial,stage2_role,rgb_edge,sar_texture,structure_relation}"
RESULT="${RESULT_ROOT}/cluster_outcome_e21_loeo_seed${SEED}.json"
AUDIT="${RESULT_ROOT}/cluster_outcome_e21_oracle_seed${SEED}.json"
OOF="${OUTPUT_ROOT}/cluster_outcome_e21_oof.pt"

echo "Cluster Outcome Utility Python: ${PYTHON_BIN}"

preflight() {
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  "${PYTHON_BIN}" -m py_compile \
    "${SCRIPT_DIR}/core.py" "${SCRIPT_DIR}/audit_outcomes.py" "${SCRIPT_DIR}/loeo.py"
  echo "Cluster Outcome Utility preflight passed"
}

audit() {
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_cluster_outcome_utility_v1.audit_outcomes \
    --train-cache "${D1_ROOT}/meta_train.pt" \
    --calibration-cache "${D1_ROOT}/meta_calibration.pt" \
    --e2-json "${E2_JSON}" \
    --output "${AUDIT}" \
    --minimum-oracle-gain-pp "${MINIMUM_ORACLE_GAIN_PP:-0.30}" \
    --minimum-outcome-pixels "${MINIMUM_OUTCOME_PIXELS:-1000}" \
    --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
    --seed "${SEED}"
}

loeo() {
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_cluster_outcome_utility_v1.loeo \
    --train-cache "${D1_ROOT}/meta_train.pt" \
    --calibration-cache "${D1_ROOT}/meta_calibration.pt" \
    --e2-json "${E2_JSON}" \
    --audit-json "${AUDIT}" \
    --output "${RESULT}" \
    --oof-output "${OOF}" \
    --groups "${OUTCOME_GROUPS}" \
    --steps "${LOEO_STEPS:-1000}" \
    --batch-size "${BATCH_SIZE:-2048}" \
    --learning-rate "${LEARNING_RATE:-1e-3}" \
    --direction-shrinkage "${DIRECTION_SHRINKAGE:-1e-3}" \
    --maximum-reject-fraction "${MAXIMUM_REJECT_FRACTION:-0.10}" \
    --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
    --seed "${SEED}"
}

case "${MODE}" in
  preflight) preflight ;;
  audit) audit ;;
  loeo) loeo ;;
  auto)
    preflight
    audit
    AUDIT_DECISION="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${AUDIT}")"
    if [[ "${AUDIT_DECISION}" != "ADMIT_CLUSTER_OUTCOME_E21_LOEO" ]]; then
      echo "Scientific stop after local outcome audit: ${AUDIT_DECISION}"
      exit 0
    fi
    loeo
    ;;
  *) echo "usage: $0 {preflight|audit|loeo|auto}"; exit 2 ;;
esac
