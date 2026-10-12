#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "${BDA_ROOT}")"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_s2_evidence_ladder_v1"
RESULTS_ROOT="${RESULTS_ROOT:-/root/data/BRIGHT-eval-results/SPI_S2_EVIDENCE_LADDER_V1}"

case "${MODE}" in
  preflight)
    python -m py_compile \
      "${SCRIPT_DIR}/contracts.py" "${SCRIPT_DIR}/evidence.py" \
      "${SCRIPT_DIR}/memory_bank.py" "${SCRIPT_DIR}/partitions.py" \
      "${SCRIPT_DIR}/router.py" "${SCRIPT_DIR}/record_result.py" \
      "${SCRIPT_DIR}/advance.py" "${SCRIPT_DIR}/summarize_foundation.py" \
      "${SCRIPT_DIR}/summarize_utility_result.py"
    python -m unittest discover -s "${SCRIPT_DIR}/tests" -p 'test_*.py' -v
    bash -n "${SCRIPT_DIR}/run.sh"
    ;;
  status)
    python -m bda_benchmark.experiments.spi_s2_evidence_ladder_v1.advance \
      --results-root "${RESULTS_ROOT}"
    ;;
  *)
    echo "Choose preflight|status" >&2
    exit 2
    ;;
esac
