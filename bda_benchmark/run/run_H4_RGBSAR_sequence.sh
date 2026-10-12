#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pilot}"
if [[ "${MODE}" != "pilot" && "${MODE}" != "full" ]]; then
  echo "Sequence mode must be pilot or full" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

VARIANTS="${VARIANTS:-a1 a2 a3 a4}"
SEED="${SEED:-3407}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H4_RGBSAR}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H4_RGBSAR}"
MAX_RETRIES="${MAX_RETRIES:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
mkdir -p "${LOG_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_ROOT}/sequence_${MODE}_${STAMP}.tsv"
printf "variant\tstage\tstatus\tlog\n" > "${SUMMARY}"

note() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_logged() {
  local log_path="$1"
  shift
  "$@" > "${log_path}" 2>&1
}

note "shared preflight starts"
PREFLIGHT_LOG="${LOG_ROOT}/H4_preflight_${STAMP}.log"
if run_logged "${PREFLIGHT_LOG}" bash run/run_H4_RGBSAR_5090.sh preflight; then
  note "shared preflight passed"
else
  note "shared preflight failed; see ${PREFLIGHT_LOG}"
  exit 1
fi

note "shared SAR stats and damage priors check starts"
STATS_LOG="${LOG_ROOT}/H4_stats_${STAMP}.log"
PRIORS_LOG="${LOG_ROOT}/H4_priors_${STAMP}.log"
run_logged "${STATS_LOG}" bash run/run_H4_RGBSAR_5090.sh stats
run_logged "${PRIORS_LOG}" bash run/run_H4_RGBSAR_5090.sh priors
note "shared statistics passed"

note "A0 strict H3 guard starts"
GUARD_LOG="${LOG_ROOT}/H4_A0_guard_seed${SEED}.log"
if run_logged "${GUARD_LOG}" env \
  VARIANT=a0 \
  EXP_NAME="H4_A0_guard_seed${SEED}" \
  bash run/run_H4_RGBSAR_5090.sh guard; then
  note "A0 guard passed"
else
  note "A0 guard failed; see ${GUARD_LOG}"
  exit 1
fi

if [[ " ${VARIANTS} " == *" a3 "* || " ${VARIANTS} " == *" a4 "* ]]; then
  note "shared train-only SAR-MAE pretraining starts/checks"
  SAR_LOG="${LOG_ROOT}/H4_SAR_MAE_seed${SEED}.log"
  if run_logged "${SAR_LOG}" bash run/run_H4_RGBSAR_5090.sh sar_pretrain; then
    note "shared SAR-MAE pretraining passed"
  else
    note "shared SAR-MAE pretraining failed; see ${SAR_LOG}"
    exit 1
  fi
fi

failures=0
for variant in ${VARIANTS}; do
  upper="${variant^^}"
  smoke_name="H4_${upper}_smoke20_seed${SEED}"
  smoke_log="${LOG_ROOT}/${smoke_name}.log"
  note "${variant}: smoke starts"
  if run_logged "${smoke_log}" env \
    VARIANT="${variant}" \
    EXP_NAME="${smoke_name}" \
    bash run/run_H4_RGBSAR_5090.sh smoke; then
    note "${variant}: smoke passed"
    printf "%s\tsmoke\tpassed\t%s\n" "${variant}" "${smoke_log}" >> "${SUMMARY}"
  else
    note "${variant}: smoke failed; see ${smoke_log}"
    printf "%s\tsmoke\tfailed\t%s\n" "${variant}" "${smoke_log}" >> "${SUMMARY}"
    failures=$((failures + 1))
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit 1
    fi
    continue
  fi

  if [[ "${MODE}" == "pilot" ]]; then
    target_steps="${PILOT_STEPS:-3000}"
  else
    target_steps="${FULL_STEPS:-8000}"
  fi
  exp_name="H4_${upper}_${MODE}${target_steps}_seed${SEED}"
  success=0
  for attempt in $(seq 1 $((MAX_RETRIES + 1))); do
    log_path="${LOG_ROOT}/${exp_name}_attempt${attempt}.log"
    note "${variant}: ${MODE} attempt ${attempt} starts"
    if run_logged "${log_path}" env \
      VARIANT="${variant}" \
      EXP_NAME="${exp_name}" \
      OPTIMIZER_STEPS="${target_steps}" \
      FORCE_RERUN="$([[ ${attempt} -gt 1 ]] && echo 1 || echo 0)" \
      bash run/run_H4_RGBSAR_5090.sh "${MODE}"; then
      note "${variant}: ${MODE} completed"
      printf "%s\t%s\tpassed\t%s\n" "${variant}" "${MODE}" "${log_path}" >> "${SUMMARY}"
      success=1
      break
    fi
    note "${variant}: ${MODE} attempt ${attempt} failed; see ${log_path}"
  done
  if [[ "${success}" != "1" ]]; then
    printf "%s\t%s\tfailed\t%s\n" "${variant}" "${MODE}" "${log_path}" >> "${SUMMARY}"
    failures=$((failures + 1))
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit 1
    fi
  fi
done

note "sequence finished; summary=${SUMMARY}; failures=${failures}"
COMPARISON_LOG="${LOG_ROOT}/H4_comparison_${STAMP}.log"
if run_logged "${COMPARISON_LOG}" bash run/run_H4_RGBSAR_5090.sh summary; then
  note "validation comparison saved; see ${COMPARISON_LOG}"
else
  note "validation comparison unavailable; see ${COMPARISON_LOG}"
fi
if [[ "${failures}" -gt 0 ]]; then
  exit 1
fi
