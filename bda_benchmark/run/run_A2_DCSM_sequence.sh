#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pilot}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

VARIANTS="${VARIANTS:-m1 m2 m3}"
MAX_RETRIES="${MAX_RETRIES:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/A2_DCSM}"
mkdir -p "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_ROOT}/sequence_${MODE}_${STAMP}.tsv"
printf 'variant\tpretrain\tsmoke\t%s\n' "${MODE}" > "${SUMMARY}"

log() { echo "[$(date '+%F %T')] $*"; }

log "shared A2-DCSM preflight starts"
bash run/run_A2_DCSM_5090.sh preflight
log "shared preflight passed"
log "M0 exact H6-A2 guard starts"
VARIANT=m0 bash run/run_A2_DCSM_5090.sh guard
log "M0 guard passed"

failures=0
for variant in ${VARIANTS}; do
  if [[ "${variant}" == m0 ]]; then continue; fi
  pretrain_status=failed; smoke_status=skipped; train_status=skipped
  log "${variant}: pretraining smoke starts"
  if VARIANT="${variant}" bash run/run_A2_DCSM_5090.sh pretrain_smoke \
      > "${LOG_ROOT}/A2_${variant^^}_pretrain_smoke.log" 2>&1; then
    log "${variant}: pretraining smoke passed"
  else
    log "${variant}: pretraining smoke failed"
    failures=$((failures + 1))
    printf '%s\t%s\t%s\t%s\n' "${variant}" failed skipped skipped >> "${SUMMARY}"
    [[ "${CONTINUE_ON_FAILURE}" == 1 ]] && continue || exit 1
  fi

  log "${variant}: pathway pretraining starts"
  if VARIANT="${variant}" bash run/run_A2_DCSM_5090.sh pretrain \
      > "${LOG_ROOT}/A2_${variant^^}_pretrain.log" 2>&1; then
    pretrain_status=passed
    log "${variant}: pathway pretraining passed"
  else
    log "${variant}: pathway pretraining failed"
    failures=$((failures + 1))
    printf '%s\t%s\t%s\t%s\n' "${variant}" failed skipped skipped >> "${SUMMARY}"
    [[ "${CONTINUE_ON_FAILURE}" == 1 ]] && continue || exit 1
  fi

  log "${variant}: supervised smoke starts"
  if VARIANT="${variant}" bash run/run_A2_DCSM_5090.sh smoke \
      > "${LOG_ROOT}/A2_${variant^^}_smoke.log" 2>&1; then
    smoke_status=passed
    log "${variant}: supervised smoke passed"
  else
    log "${variant}: supervised smoke failed"
    failures=$((failures + 1))
    printf '%s\t%s\t%s\t%s\n' "${variant}" "${pretrain_status}" failed skipped >> "${SUMMARY}"
    [[ "${CONTINUE_ON_FAILURE}" == 1 ]] && continue || exit 1
  fi

  attempt=0
  while [[ "${attempt}" -le "${MAX_RETRIES}" ]]; do
    attempt=$((attempt + 1))
    log "${variant}: ${MODE} attempt ${attempt} starts"
    if VARIANT="${variant}" bash run/run_A2_DCSM_5090.sh "${MODE}" \
        > "${LOG_ROOT}/A2_${variant^^}_${MODE}_attempt${attempt}.log" 2>&1; then
      train_status=passed
      log "${variant}: ${MODE} completed"
      break
    fi
    log "${variant}: ${MODE} attempt ${attempt} failed"
  done
  if [[ "${train_status}" != passed ]]; then failures=$((failures + 1)); fi
  printf '%s\t%s\t%s\t%s\n' \
    "${variant}" "${pretrain_status}" "${smoke_status}" "${train_status}" >> "${SUMMARY}"
  if [[ "${train_status}" != passed && "${CONTINUE_ON_FAILURE}" != 1 ]]; then exit 1; fi
done

log "A2-DCSM sequence finished; summary=${SUMMARY}; failures=${failures}"
[[ "${failures}" -eq 0 ]]
