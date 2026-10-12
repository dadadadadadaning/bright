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
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_event_robust_cluster_verifier_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || PYTHON_BIN=python

if [[ "${MODE}" == "preflight" ]]; then
  "${PYTHON_BIN}" -m py_compile \
    "${SCRIPT_DIR}/core.py" "${SCRIPT_DIR}/run_loeo.py"
  "${PYTHON_BIN}" -m pytest -q "${SCRIPT_DIR}/test_core.py"
  bash -n "${SCRIPT_DIR}/run.sh"
  echo "Event-robust cluster verifier preflight passed"
  exit 0
fi

SEED="${SEED:-7341}"
D1_ROOT="${D1_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed${SEED}}"
MACU_ROOT="${MACU_ROOT:-/root/data/BRIGHT-checkpoints/SPI_METRIC_ALIGNED_CLUSTER_UTILITY_V1/seed${SEED}/targets}"
OUTPUT="${OUTPUT:-/root/data/BRIGHT-eval-results/SPI_EVENT_ROBUST_CLUSTER_VERIFIER_V1/loeo_seed${SEED}.json}"

for path in \
  "${D1_ROOT}/meta_train.pt" "${D1_ROOT}/meta_calibration.pt" \
  "${MACU_ROOT}/meta_train_targets.pt" "${MACU_ROOT}/meta_calibration_targets.pt"; do
  test -s "${path}" || { echo "Missing input: ${path}" >&2; exit 1; }
done

test ! -e "${OUTPUT}" || {
  echo "Refusing to overwrite completed isolated LOEO result: ${OUTPUT}" >&2
  exit 1
}

"${PYTHON_BIN}" -u -m \
  bda_benchmark.experiments.spi_event_robust_cluster_verifier_v1.run_loeo \
  --train-cache "${D1_ROOT}/meta_train.pt" \
  --calibration-cache "${D1_ROOT}/meta_calibration.pt" \
  --train-targets "${MACU_ROOT}/meta_train_targets.pt" \
  --calibration-targets "${MACU_ROOT}/meta_calibration_targets.pt" \
  --output "${OUTPUT}" \
  --objectives "${OBJECTIVES:-erm,event_balanced,groupdro,vrex}" \
  --steps "${STEPS:-1000}" --batch-size "${BATCH_SIZE:-2048}" \
  --device "${DEVICE:-cpu}" --seed "${SEED}"
