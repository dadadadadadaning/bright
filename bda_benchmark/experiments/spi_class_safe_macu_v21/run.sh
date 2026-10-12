#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|loeo) ;;
  *) echo "Choose preflight|loeo" >&2; exit 2 ;;
esac

BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "${BDA_ROOT}")"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_class_safe_macu_v21"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN=python
echo "CS-MACU V2.1 Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON_BIN}" -m py_compile "${SCRIPT_DIR}/core.py" "${SCRIPT_DIR}/loeo.py"
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "CS-MACU V2.1 preflight passed"
  exit 0
fi

SEED="${SEED:-7341}"
D1_ROOT="${D1_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed${SEED}}"
D0_JSON="${D0_JSON:-/root/data/BRIGHT-eval-results/SPI_MACU_FAILURE_AUDIT_D0_V1/macu_failure_d0_seed${SEED}.json}"
V2_JSON="${V2_JSON:-/root/data/BRIGHT-eval-results/SPI_CLASS_SAFE_MACU_V2/cs_macu_v2_loeo_seed${SEED}.json}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_CLASS_SAFE_MACU_V21}"
TRAIN_CACHE="${D1_ROOT}/meta_train.pt"
CAL_CACHE="${D1_ROOT}/meta_calibration.pt"
OUTPUT="${RESULT_ROOT}/cs_macu_v21_three_arm_loeo_seed${SEED}.json"

for path in "${TRAIN_CACHE}" "${CAL_CACHE}" "${D0_JSON}" "${V2_JSON}"; do
  test -s "${path}" || { echo "Missing V2.1 input: ${path}" >&2; exit 1; }
done
test ! -e "${OUTPUT}" || { echo "Refusing repeated V2.1 LOEO: ${OUTPUT}" >&2; exit 1; }
mkdir -p "${RESULT_ROOT}"
"${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_class_safe_macu_v21.loeo \
  --train-cache "${TRAIN_CACHE}" --calibration-cache "${CAL_CACHE}" \
  --d0-json "${D0_JSON}" --v2-json "${V2_JSON}" --output "${OUTPUT}" \
  --groups "${FEATURE_GROUPS:-spatial,stage2_role,rgb_edge,sar_texture,structure_relation}" \
  --steps "${LOEO_STEPS:-1000}" --batch-size "${BATCH_SIZE:-2048}" \
  --learning-rate "${LEARNING_RATE:-1e-3}" \
  --maximum-reject-fraction "${MAXIMUM_REJECT_FRACTION:-0.70}" \
  --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" --seed "${SEED}"
echo "CS-MACU V2.1 three-arm LOEO complete. Meta-confirm/Val450/test were not accessed."
