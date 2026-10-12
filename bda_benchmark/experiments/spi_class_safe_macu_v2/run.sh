#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "${MODE}" in
  preflight|loeo|freeze|auto) ;;
  *) echo "Choose preflight|loeo|freeze|auto" >&2; exit 2 ;;
esac

BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "${BDA_ROOT}")"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_class_safe_macu_v2"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN=python
echo "CS-MACU V2 Python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON_BIN}" -m py_compile "${SCRIPT_DIR}/core.py" \
    "${SCRIPT_DIR}/loeo.py" "${SCRIPT_DIR}/freeze.py"
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "CS-MACU V2 preflight passed"
  exit 0
fi

SEED="${SEED:-7341}"
D1_ROOT="${D1_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed${SEED}}"
D0_JSON="${D0_JSON:-/root/data/BRIGHT-eval-results/SPI_MACU_FAILURE_AUDIT_D0_V1/macu_failure_d0_seed${SEED}.json}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_CLASS_SAFE_MACU_V2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CLASS_SAFE_MACU_V2/seed${SEED}}"
TRAIN_CACHE="${D1_ROOT}/meta_train.pt"
CAL_CACHE="${D1_ROOT}/meta_calibration.pt"
LOEO_JSON="${RESULT_ROOT}/cs_macu_v2_loeo_seed${SEED}.json"
FROZEN_DIR="${OUTPUT_ROOT}/linear_dual_step${LOEO_STEPS:-1000}"

for path in "${TRAIN_CACHE}" "${CAL_CACHE}" "${D0_JSON}"; do
  test -s "${path}" || { echo "Missing V2 input: ${path}" >&2; exit 1; }
done
"${PYTHON_BIN}" -c "import json; p=json.load(open('${D0_JSON}')); assert p.get('decision') == 'ADMIT_CS_MACU_V2_DEVELOPMENT_ONLY', p.get('decision'); print('Class-Safe Oracle development prerequisite passed')"
mkdir -p "${RESULT_ROOT}" "${OUTPUT_ROOT}"

run_loeo() {
  test ! -e "${LOEO_JSON}" || { echo "Refusing repeated LOEO: ${LOEO_JSON}" >&2; exit 1; }
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_class_safe_macu_v2.loeo \
    --train-cache "${TRAIN_CACHE}" --calibration-cache "${CAL_CACHE}" \
    --d0-json "${D0_JSON}" --output "${LOEO_JSON}" \
    --groups "${FEATURE_GROUPS:-spatial,stage2_role,rgb_edge,sar_texture,structure_relation}" \
    --steps "${LOEO_STEPS:-1000}" --batch-size "${BATCH_SIZE:-2048}" \
    --learning-rate "${LEARNING_RATE:-1e-3}" \
    --maximum-reject-fraction "${MAXIMUM_REJECT_FRACTION:-0.70}" \
    --minimum-gain-pp 0.30 --minimum-nonnegative-events 10 \
    --bootstrap-iterations "${BOOTSTRAP_ITERATIONS:-10000}" --seed "${SEED}"
}

run_freeze() {
  test -s "${LOEO_JSON}" || { echo "Run LOEO first: missing ${LOEO_JSON}" >&2; exit 1; }
  "${PYTHON_BIN}" -c "import json; p=json.load(open('${LOEO_JSON}')); assert p.get('decision') == 'FREEZE_CS_MACU_V2_FOR_VAL450', p.get('decision'); print('CS-MACU V2 LOEO prerequisite passed')"
  "${PYTHON_BIN}" -u -m bda_benchmark.experiments.spi_class_safe_macu_v2.freeze \
    --train-cache "${TRAIN_CACHE}" --calibration-cache "${CAL_CACHE}" \
    --loeo-json "${LOEO_JSON}" --d0-json "${D0_JSON}" \
    --output-dir "${FROZEN_DIR}" --steps "${LOEO_STEPS:-1000}" \
    --batch-size "${BATCH_SIZE:-2048}" --learning-rate "${LEARNING_RATE:-1e-3}" \
    --maximum-reject-fraction "${MAXIMUM_REJECT_FRACTION:-0.70}" --seed "${SEED}"
}

case "${MODE}" in
  loeo) run_loeo ;;
  freeze) run_freeze ;;
  auto)
    run_loeo
    decision="$(${PYTHON_BIN} -c "import json; print(json.load(open('${LOEO_JSON}'))['decision'])")"
    if [[ "${decision}" == "FREEZE_CS_MACU_V2_FOR_VAL450" ]]; then
      run_freeze
    else
      echo "Automatic stop after LOEO: ${decision}. Final V2 model was not frozen."
    fi
    ;;
esac
