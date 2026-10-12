#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: watch_then_run.sh PILOT_PID [-- NEXT_PROGRAM ARG ...]" >&2
  exit 2
fi

PILOT_PID="$1"
shift
if [[ ! "${PILOT_PID}" =~ ^[0-9]+$ ]]; then
  echo "PILOT_PID must be numeric, got: ${PILOT_PID}" >&2
  exit 2
fi
if [[ "${1:-}" == "--" ]]; then shift; fi

BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "${BDA_ROOT}")"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_SAR_ADAPTER_V1}"
PILOT_LOG="${PILOT_LOG:-/root/data/BRIGHT-train-logs/SPI_SAR_ADAPTER_V1_seed3407.log}"
STEPS="${PILOT_STEPS:-500}"
SEED="${SEED:-3407}"
POLL_SECONDS="${POLL_SECONDS:-30}"

echo "Watching SAR pilot pid=${PILOT_PID}; poll=${POLL_SECONDS}s"
while kill -0 "${PILOT_PID}" 2>/dev/null; do
  if [[ -f "${PILOT_LOG}" ]]; then
    last_line="$(tail -n 1 "${PILOT_LOG}" || true)"
    echo "[$(date '+%F %T')] ${last_line}"
  else
    echo "[$(date '+%F %T')] waiting for log: ${PILOT_LOG}"
  fi
  sleep "${POLL_SECONDS}"
done

echo "Pilot launcher pid=${PILOT_PID} has exited; validating artifacts."
if [[ ! -f "${PILOT_LOG}" ]]; then
  echo "Pilot log is missing: ${PILOT_LOG}" >&2
  exit 1
fi
if ! grep -Fq "SAR adapter run complete. Val450 was not accessed." "${PILOT_LOG}"; then
  echo "Pilot did not record its successful completion marker." >&2
  echo "Last 80 log lines:" >&2
  tail -n 80 "${PILOT_LOG}" >&2
  exit 1
fi

python -m bda_benchmark.experiments.spi_sar_adapter_v1.summarize_results \
  --result-root "${RESULT_ROOT}" --steps "${STEPS}" --seed "${SEED}"

if [[ "$#" -eq 0 ]]; then
  echo "No next program supplied; stopping safely after summary."
  exit 0
fi

echo "Starting explicitly supplied next program: $*"
"$@"
