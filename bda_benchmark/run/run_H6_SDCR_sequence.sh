#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pilot}"
case "${MODE}" in pilot|full) ;; *) echo "Sequence mode must be pilot or full" >&2; exit 2;; esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-3407}"
VARIANTS="${VARIANTS:-a1 a2 a3}"
MAX_RETRIES="${MAX_RETRIES:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H6_SDCR}"
EXP_SUFFIX="${EXP_SUFFIX:-}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_ROOT}/sequence_${MODE}_${STAMP}.tsv"
printf "variant\tstage\tstatus\tattempt\tlog\n" > "${SUMMARY}"
note() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
run_logged() { local log="$1"; shift; "$@" > "${log}" 2>&1; }

note "shared H6 preflight starts"
PREFLIGHT_LOG="${LOG_ROOT}/shared_preflight_${STAMP}.log"
run_logged "${PREFLIGHT_LOG}" bash run/run_H6_SDCR_5090.sh preflight || {
  note "preflight failed; see ${PREFLIGHT_LOG}"; exit 1; }
note "shared H6 preflight passed"

note "A0 strict H4-A4 guard starts"
GUARD_LOG="${LOG_ROOT}/H6_A0_guard_seed${SEED}_${STAMP}.log"
run_logged "${GUARD_LOG}" env VARIANT=a0 EXP_NAME="H6_A0_guard_seed${SEED}_${STAMP}" \
  bash run/run_H6_SDCR_5090.sh guard || { note "A0 guard failed; see ${GUARD_LOG}"; exit 1; }
note "A0 guard passed"

note "A1 strict P3 RGB-anchor guard starts"
ANCHOR_LOG="${LOG_ROOT}/H6_A1_anchor_guard_seed${SEED}_${STAMP}.log"
run_logged "${ANCHOR_LOG}" env EXP_NAME="H6_A1_anchor_guard_seed${SEED}_${STAMP}" \
  bash run/run_H6_SDCR_5090.sh anchor_guard || { note "A1 anchor guard failed; see ${ANCHOR_LOG}"; exit 1; }
note "A1 anchor guard passed"

failures=0
for variant in ${VARIANTS}; do
  upper="${variant^^}"; suffix=""; [[ -n "${EXP_SUFFIX}" ]] && suffix="_${EXP_SUFFIX}"
  smoke_name="H6_${upper}_smoke20_seed${SEED}${suffix}"
  smoke_log="${LOG_ROOT}/${smoke_name}.log"
  note "${variant}: smoke starts"
  if run_logged "${smoke_log}" env VARIANT="${variant}" EXP_NAME="${smoke_name}" \
      bash run/run_H6_SDCR_5090.sh smoke; then
    note "${variant}: smoke passed"
    printf "%s\tsmoke\tpassed\t1\t%s\n" "${variant}" "${smoke_log}" >> "${SUMMARY}"
  else
    note "${variant}: smoke failed; see ${smoke_log}"
    printf "%s\tsmoke\tfailed\t1\t%s\n" "${variant}" "${smoke_log}" >> "${SUMMARY}"
    failures=$((failures + 1)); [[ "${CONTINUE_ON_FAILURE}" == 1 ]] && continue || exit 1
  fi
  [[ "${MODE}" == pilot ]] && target="${OPTIMIZER_STEPS:-3000}" || target="${OPTIMIZER_STEPS:-8000}"
  experiment="H6_${upper}_${MODE}${target}_seed${SEED}${suffix}"
  completed=0
  for ((attempt=1; attempt<=MAX_RETRIES+1; attempt++)); do
    log="${LOG_ROOT}/${experiment}_attempt${attempt}.log"
    note "${variant}: ${MODE} attempt ${attempt} starts"
    if run_logged "${log}" env VARIANT="${variant}" EXP_NAME="${experiment}" OPTIMIZER_STEPS="${target}" \
        bash run/run_H6_SDCR_5090.sh "${MODE}"; then
      note "${variant}: ${MODE} completed"
      printf "%s\t%s\tcompleted\t%s\t%s\n" "${variant}" "${MODE}" "${attempt}" "${log}" >> "${SUMMARY}"
      completed=1; break
    fi
    note "${variant}: attempt ${attempt} failed; see ${log}"
    printf "%s\t%s\tfailed\t%s\t%s\n" "${variant}" "${MODE}" "${attempt}" "${log}" >> "${SUMMARY}"
  done
  if [[ "${completed}" != 1 ]]; then failures=$((failures + 1)); [[ "${CONTINUE_ON_FAILURE}" == 1 ]] || exit 1; fi
done

COMPARISON_LOG="${LOG_ROOT}/H6_comparison_${STAMP}.log"
bash run/run_H6_SDCR_5090.sh summary > "${COMPARISON_LOG}" 2>&1 || true
note "sequence finished; summary=${SUMMARY}; failures=${failures}"
exit "${failures}"
