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
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_macu_failure_audit_d0_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN=python
echo "MACU Failure D0 Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON_BIN}" -m py_compile "${SCRIPT_DIR}/core.py" "${SCRIPT_DIR}/audit.py"
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "MACU Failure Audit D0 preflight passed"
  exit 0
fi

SEED="${SEED:-7341}"
D1_ROOT="${D1_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed${SEED}}"
MACU_ROOT="${MACU_ROOT:-/root/data/BRIGHT-checkpoints/SPI_METRIC_ALIGNED_CLUSTER_UTILITY_V1/seed${SEED}}"
MACU_RESULT_ROOT="${MACU_RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_METRIC_ALIGNED_CLUSTER_UTILITY_V1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-eval-results/SPI_MACU_FAILURE_AUDIT_D0_V1}"
TRAIN_CACHE="${D1_ROOT}/meta_train.pt"
CAL_CACHE="${D1_ROOT}/meta_calibration.pt"
CONFIRM_CACHE="${D1_ROOT}/meta_confirm.pt"
TRAIN_TARGETS="${MACU_ROOT}/targets/meta_train_targets.pt"
VERIFIER="${MACU_ROOT}/linear_step1000/verifier_frozen.pth"
CONFIRM_RESULT="${MACU_RESULT_ROOT}/macu_e1_confirm_seed${SEED}.json"
OUTPUT="${OUTPUT_ROOT}/macu_failure_d0_seed${SEED}.json"

for path in "${TRAIN_CACHE}" "${CAL_CACHE}" "${CONFIRM_CACHE}" \
  "${TRAIN_TARGETS}" "${VERIFIER}" "${CONFIRM_RESULT}"; do
  test -s "${path}" || { echo "Missing D0 input: ${path}" >&2; exit 1; }
done
test ! -e "${OUTPUT}" || { echo "Refusing repeated D0 audit: ${OUTPUT}" >&2; exit 1; }
mkdir -p "${OUTPUT_ROOT}"
"${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_macu_failure_audit_d0_v1.audit \
  --train-cache "${TRAIN_CACHE}" --calibration-cache "${CAL_CACHE}" \
  --confirm-cache "${CONFIRM_CACHE}" --train-targets "${TRAIN_TARGETS}" \
  --verifier-checkpoint "${VERIFIER}" --confirm-result "${CONFIRM_RESULT}" \
  --output "${OUTPUT}" --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" \
  --seed "${SEED}"
echo "MACU Failure Audit D0 complete. No image inference and no Val450/test access."
