#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|audit) ;;
  *) echo "Choose preflight|audit" >&2; exit 2 ;;
esac

BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "${BDA_ROOT}")"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_pseudo_instance_memory_v1"
DEFAULT_PYTHON_BIN="/root/data/conda_envs/bright4090/bin/python"
if [[ -x "${DEFAULT_PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
"${PYTHON_BIN}" -c "import sys; print('PI-M0 Python:', sys.executable)"

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON_BIN}" -m py_compile "${SCRIPT_DIR}/core.py" "${SCRIPT_DIR}/audit.py"
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "PI-M0 preflight passed"
  exit 0
fi

SEED="${SEED:-7341}"
CACHE_ROOT="${CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_BUILDING_UTILITY_V1/cache_seed${SEED}}"
BU_RESULT="${BU_RESULT:-/root/data/BRIGHT-eval-results/SPI_BUILDING_UTILITY_V1/bu_e0_seed${SEED}.json}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_PSEUDO_INSTANCE_MEMORY_V1}"
MEMORY_ROOT="${MEMORY_ROOT:-/root/data/BRIGHT-checkpoints/SPI_PSEUDO_INSTANCE_MEMORY_V1}"

for required in "${CACHE_ROOT}/meta_train.pt" "${CACHE_ROOT}/meta_calibration.pt" "${BU_RESULT}"; do
  test -f "${required}" || { echo "Missing prerequisite: ${required}" >&2; exit 1; }
done
mkdir -p "${RESULT_ROOT}" "${MEMORY_ROOT}"

"${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_pseudo_instance_memory_v1.audit \
  --train-cache "${CACHE_ROOT}/meta_train.pt" \
  --calibration-cache "${CACHE_ROOT}/meta_calibration.pt" \
  --bu-e0-result "${BU_RESULT}" \
  --output "${RESULT_ROOT}/pi_m0_seed${SEED}.json" \
  --output-memory "${MEMORY_ROOT}/utility_memory_seed${SEED}.pt" \
  --seed "${SEED}" \
  --utility-margin "${UTILITY_MARGIN:-0.10}" \
  --capacity-per-event-label "${CAPACITY_PER_EVENT_LABEL:-256}" \
  --max-neighbors-per-event "${MAX_NEIGHBORS_PER_EVENT:-2}" \
  --ks "${KS:-1,3,5,8}" \
  --temperatures "${TEMPERATURES:-0.03,0.07,0.15,0.30}" \
  --chunk-size "${CHUNK_SIZE:-512}" \
  --permutations "${PERMUTATIONS:-1000}"

echo "PI-M0 audit complete. Val450 was not accessed."

