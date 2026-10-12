#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pilot}"
case "${MODE}" in
  pilot|full) ;;
  *)
    echo "Sequence mode must be pilot or full" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-3407}"
VARIANTS="${VARIANTS:-b1 c1}"
MAX_RETRIES="${MAX_RETRIES:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H5_BC}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H5_BC}"
EXP_SUFFIX="${EXP_SUFFIX:-}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_ROOT}/sequence_${MODE}_${STAMP}.tsv"
printf "variant\tstage\tstatus\tattempt\tlog\n" > "${SUMMARY}"

note() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_logged() {
  local log_path="$1"
  shift
  "$@" > "${log_path}" 2>&1
}

note "shared preflight starts"
PREFLIGHT_LOG="${LOG_ROOT}/shared_preflight_${STAMP}.log"
if run_logged "${PREFLIGHT_LOG}" bash run/run_H5_BC_5090.sh preflight; then
  note "shared preflight passed"
else
  note "shared preflight failed; see ${PREFLIGHT_LOG}"
  exit 1
fi

note "B0 strict H4-A4 guard starts"
GUARD_LOG="${LOG_ROOT}/H5_B0_guard_seed${SEED}_${STAMP}.log"
if run_logged "${GUARD_LOG}" env \
  VARIANT=b0 \
  EXP_NAME="H5_B0_guard_seed${SEED}_${STAMP}" \
  bash run/run_H5_BC_5090.sh guard; then
  note "B0 guard passed"
else
  note "B0 guard failed; see ${GUARD_LOG}"
  exit 1
fi

if [[ " ${VARIANTS} " == *" c1 "* ]]; then
  note "C1 joint-pretraining smoke starts"
  JOINT_SMOKE_LOG="${LOG_ROOT}/H5_C1_joint_smoke20_seed${SEED}_${STAMP}.log"
  if run_logged "${JOINT_SMOKE_LOG}" env \
    FORCE_RERUN=1 \
    bash run/run_H5_BC_5090.sh joint_smoke; then
    note "C1 joint-pretraining smoke passed"
  else
    note "C1 joint-pretraining smoke failed; see ${JOINT_SMOKE_LOG}"
    printf "c1\tjoint_smoke\tfailed\t1\t%s\n" "${JOINT_SMOKE_LOG}" >> "${SUMMARY}"
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit 1
    fi
    VARIANTS="${VARIANTS//c1/}"
  fi

  if [[ " ${VARIANTS} " == *" c1 "* ]]; then
    note "C1 joint pretraining starts"
    JOINT_LOG="${LOG_ROOT}/H5_C1_joint_pretrain_seed${SEED}_${STAMP}.log"
    if run_logged "${JOINT_LOG}" bash run/run_H5_BC_5090.sh joint_pretrain; then
      note "C1 joint pretraining passed"
      printf "c1\tjoint_pretrain\tcompleted\t1\t%s\n" "${JOINT_LOG}" >> "${SUMMARY}"
    else
      note "C1 joint pretraining failed; see ${JOINT_LOG}"
      printf "c1\tjoint_pretrain\tfailed\t1\t%s\n" "${JOINT_LOG}" >> "${SUMMARY}"
      if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
        exit 1
      fi
      VARIANTS="${VARIANTS//c1/}"
    fi
  fi
fi

failures=0
for variant in ${VARIANTS}; do
  upper="${variant^^}"
  suffix=""
  if [[ -n "${EXP_SUFFIX}" ]]; then
    suffix="_${EXP_SUFFIX}"
  fi

  smoke_name="H5_${upper}_smoke20_seed${SEED}${suffix}"
  smoke_log="${LOG_ROOT}/${smoke_name}.log"
  note "${variant}: supervised smoke starts"
  if run_logged "${smoke_log}" env \
    VARIANT="${variant}" \
    EXP_NAME="${smoke_name}" \
    bash run/run_H5_BC_5090.sh smoke; then
    note "${variant}: smoke passed"
    printf "%s\tsmoke\tpassed\t1\t%s\n" \
      "${variant}" "${smoke_log}" >> "${SUMMARY}"
  else
    note "${variant}: smoke failed; see ${smoke_log}"
    printf "%s\tsmoke\tfailed\t1\t%s\n" \
      "${variant}" "${smoke_log}" >> "${SUMMARY}"
    failures=$((failures + 1))
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit 1
    fi
    continue
  fi

  if [[ "${MODE}" == "pilot" ]]; then
    target_steps="${OPTIMIZER_STEPS:-3000}"
  else
    target_steps="${OPTIMIZER_STEPS:-8000}"
  fi
  experiment="H5_${upper}_${MODE}${target_steps}_seed${SEED}${suffix}"
  completed=0
  attempt=1
  while [[ "${attempt}" -le "$((MAX_RETRIES + 1))" ]]; do
    log_path="${LOG_ROOT}/${experiment}_attempt${attempt}.log"
    note "${variant}: ${MODE} attempt ${attempt} starts"
    if run_logged "${log_path}" env \
      VARIANT="${variant}" \
      EXP_NAME="${experiment}" \
      OPTIMIZER_STEPS="${target_steps}" \
      bash run/run_H5_BC_5090.sh "${MODE}"; then
      note "${variant}: ${MODE} completed"
      printf "%s\t%s\tcompleted\t%s\t%s\n" \
        "${variant}" "${MODE}" "${attempt}" "${log_path}" >> "${SUMMARY}"
      completed=1
      break
    fi
    note "${variant}: ${MODE} attempt ${attempt} failed; see ${log_path}"
    printf "%s\t%s\tfailed\t%s\t%s\n" \
      "${variant}" "${MODE}" "${attempt}" "${log_path}" >> "${SUMMARY}"
    attempt=$((attempt + 1))
  done
  if [[ "${completed}" != "1" ]]; then
    failures=$((failures + 1))
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit 1
    fi
  fi
done

COMPARISON_LOG="${LOG_ROOT}/H5_BC_comparison_${STAMP}.log"
if bash run/run_H5_BC_5090.sh summary > "${COMPARISON_LOG}" 2>&1; then
  note "validation comparison saved; see ${COMPARISON_LOG}"
else
  note "comparison failed; see ${COMPARISON_LOG}"
fi
note "sequence finished; summary=${SUMMARY}; failures=${failures}"
exit "${failures}"
