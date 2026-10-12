#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]] || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: watch_then_run.sh AUDIT_PID" >&2
  exit 2
fi

AUDIT_PID="$1"
BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "${BDA_ROOT}")"
export PYTHONPATH="${PROJECT_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"
SCRIPT_DIR="${BDA_ROOT}/experiments/spi_s2_evidence_v2"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_S2_EVIDENCE_V2}"
SEED="${SEED:-3407}"
AUDIT_LOG="${AUDIT_LOG:-/root/data/BRIGHT-train-logs/SPI_S2_EVIDENCE_V2_AUDIT_seed${SEED}.log}"
ADMISSION_JSON="${RESULT_ROOT}/parameter_free_admission_seed${SEED}.json"
STEPS="${VERIFIER_STEPS:-500}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RUN_POLICY="${RUN_POLICY:-best}"
SELECTION_JSON="${RESULT_ROOT}/autorun_selection_seed${SEED}.json"
GROUPS_FILE="${RESULT_ROOT}/autorun_groups_seed${SEED}.txt"
SUMMARY_JSON="${RESULT_ROOT}/autorun_summary_step${STEPS}_seed${SEED}.json"

case "${RUN_POLICY}" in
  best|all_admitted) ;;
  *) echo "RUN_POLICY must be best or all_admitted" >&2; exit 2 ;;
esac

echo "Watching Evidence V2 audit pid=${AUDIT_PID}; policy=${RUN_POLICY}; poll=${POLL_SECONDS}s"
while kill -0 "${AUDIT_PID}" 2>/dev/null; do
  process_state="$(ps -p "${AUDIT_PID}" -o stat= 2>/dev/null || true)"
  process_command="$(ps -p "${AUDIT_PID}" -o args= 2>/dev/null || true)"
  if [[ "${process_state}" == Z* ]]; then
    break
  fi
  if [[ -n "${process_command}" ]] && [[ "${process_command}" != *"spi_s2_evidence_v2/run.sh audit"* ]]; then
    echo "PID ${AUDIT_PID} no longer belongs to the Evidence V2 audit; validating artifacts." >&2
    break
  fi
  if [[ -f "${AUDIT_LOG}" ]]; then
    echo "[$(date '+%F %T')] $(tail -n 1 "${AUDIT_LOG}" || true)"
  else
    echo "[$(date '+%F %T')] waiting for ${AUDIT_LOG}"
  fi
  sleep "${POLL_SECONDS}"
done

echo "Audit launcher exited; validating completion marker and admission artifact."
if [[ ! -f "${AUDIT_LOG}" ]]; then
  echo "Missing audit log: ${AUDIT_LOG}" >&2
  exit 1
fi
if ! grep -Fq "Parameter-free evidence audit complete. Val450 was not accessed." "${AUDIT_LOG}"; then
  echo "Audit did not finish successfully; no verifier will be started." >&2
  tail -n 100 "${AUDIT_LOG}" >&2
  exit 1
fi
if [[ ! -f "${ADMISSION_JSON}" ]]; then
  echo "Missing admission result: ${ADMISSION_JSON}" >&2
  exit 1
fi

python -u -m bda_benchmark.experiments.spi_s2_evidence_v2.select_admitted \
  --admission-json "${ADMISSION_JSON}" --policy "${RUN_POLICY}" \
  --output "${SELECTION_JSON}" --groups-output "${GROUPS_FILE}"

mapfile -t selected_groups < "${GROUPS_FILE}"
if [[ "${#selected_groups[@]}" -eq 0 ]]; then
  echo "No evidence group passed parameter-free admission; stopping safely."
  exit 0
fi

for evidence_group in "${selected_groups[@]}"; do
  echo "Starting isolated single-evidence verifier: ${evidence_group}"
  VERIFIER_STEPS="${STEPS}" \
  VERIFIER_BATCH_SIZE="${VERIFIER_BATCH_SIZE:-8192}" \
  VERIFIER_LR="${VERIFIER_LR:-1e-3}" \
  DEVICE="${DEVICE:-cuda:0}" \
  bash "${SCRIPT_DIR}/run.sh" "${evidence_group}"
done

python -u -m bda_benchmark.experiments.spi_s2_evidence_v2.summarize_autorun \
  --selection-json "${SELECTION_JSON}" --result-root "${RESULT_ROOT}" \
  --steps "${STEPS}" --seed "${SEED}" --output "${SUMMARY_JSON}"

echo "Evidence V2 autorun complete. Val450 was not accessed."
