#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pilot}"
case "${MODE}" in
  pilot|full) ;;
  *) echo "Mode must be pilot or full" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-3407}"
VARIANTS="${VARIANTS:-b1 b2}"
MAX_RETRIES="${MAX_RETRIES:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H6_SDCR}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_ROOT}/H6B_sequence_${MODE}_${STAMP}.tsv"
printf "variant\tstage\tstatus\tattempt\tlog\n" > "${SUMMARY}"
note() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
run_logged() { local log="$1"; shift; "$@" > "${log}" 2>&1; }

note "shared H6-B preflight starts"
PREFLIGHT_LOG="${LOG_ROOT}/H6B_preflight_${STAMP}.log"
run_logged "${PREFLIGHT_LOG}" bash run/run_H6_SDCR_5090.sh preflight || {
  note "preflight failed; see ${PREFLIGHT_LOG}"
  exit 1
}
note "shared H6-B preflight passed"

note "B0 exact P3 validation+test guard starts"
GUARD_LOG="${LOG_ROOT}/H6_B0_safe_guard_seed${SEED}_${STAMP}.log"
run_logged "${GUARD_LOG}" env \
  EXP_NAME="H6_B0_safe_guard_seed${SEED}_${STAMP}" \
  bash run/run_H6_SDCR_5090.sh safe_guard || {
    note "B0 guard failed; see ${GUARD_LOG}"
    exit 1
  }
note "B0 guard passed"

failures=0
for variant in ${VARIANTS}; do
  case "${variant}" in b1|b2) ;; *) note "invalid H6-B variant: ${variant}"; exit 2 ;; esac
  upper="${variant^^}"
  smoke_name="H6_${upper}_smoke20_seed${SEED}"
  smoke_log="${LOG_ROOT}/${smoke_name}.log"
  note "${variant}: smoke starts (test monitor limited to 2 batches)"
  if run_logged "${smoke_log}" env \
      VARIANT="${variant}" EXP_NAME="${smoke_name}" \
      TEST_MAX_BATCHES=2 \
      bash run/run_H6_SDCR_5090.sh smoke; then
    note "${variant}: smoke passed"
    printf "%s\tsmoke\tpassed\t1\t%s\n" "${variant}" "${smoke_log}" >> "${SUMMARY}"
  else
    note "${variant}: smoke failed; see ${smoke_log}"
    printf "%s\tsmoke\tfailed\t1\t%s\n" "${variant}" "${smoke_log}" >> "${SUMMARY}"
    failures=$((failures + 1))
    [[ "${CONTINUE_ON_FAILURE}" == 1 ]] && continue || exit 1
  fi

  if [[ "${MODE}" == pilot ]]; then
    target="${OPTIMIZER_STEPS:-3000}"
  else
    target="${OPTIMIZER_STEPS:-8000}"
  fi
  experiment="H6_${upper}_${MODE}${target}_seed${SEED}"
  completed=0
  for ((attempt=1; attempt<=MAX_RETRIES+1; attempt++)); do
    log="${LOG_ROOT}/${experiment}_attempt${attempt}.log"
    note "${variant}: ${MODE} attempt ${attempt} starts"
    if run_logged "${log}" env \
        VARIANT="${variant}" EXP_NAME="${experiment}" \
        OPTIMIZER_STEPS="${target}" TEST_MAX_BATCHES=0 \
        bash run/run_H6_SDCR_5090.sh "${MODE}"; then
      note "${variant}: ${MODE} completed"
      printf "%s\t%s\tcompleted\t%s\t%s\n" \
        "${variant}" "${MODE}" "${attempt}" "${log}" >> "${SUMMARY}"
      completed=1
      break
    fi
    note "${variant}: attempt ${attempt} failed; see ${log}"
    printf "%s\t%s\tfailed\t%s\t%s\n" \
      "${variant}" "${MODE}" "${attempt}" "${log}" >> "${SUMMARY}"
  done
  if [[ "${completed}" != 1 ]]; then
    failures=$((failures + 1))
    [[ "${CONTINUE_ON_FAILURE}" == 1 ]] || exit 1
  fi
done

note "H6-B sequence finished; summary=${SUMMARY}; failures=${failures}"
exit "${failures}"
