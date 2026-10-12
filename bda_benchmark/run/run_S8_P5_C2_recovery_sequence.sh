#!/usr/bin/env bash
set -uo pipefail

MODE="${1:-pilot}"
case "${MODE}" in
  pilot|full) ;;
  *)
    echo "Choose pilot|full" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VARIANTS="${VARIANTS:-r0 r1 r2 r3}"
MAX_RETRIES="${MAX_RETRIES:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/S8_P5_C2_Recovery}"
mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

timestamp="$(date +%Y%m%d_%H%M%S)"
summary="${LOG_ROOT}/sequence_${MODE}_${timestamp}.tsv"
printf "variant\tstage\tstatus\tattempt\tlog\n" > "${summary}"

stamp() {
  echo "[$(date '+%F %T')] $*"
}

run_logged() {
  local log_path="$1"
  shift
  "$@" > "${log_path}" 2>&1
}

stamp "shared preflight starts"
preflight_log="${LOG_ROOT}/preflight_${timestamp}.log"
if ! run_logged "${preflight_log}" \
    bash run/run_S8_P5_C2_recovery_5090.sh preflight; then
  stamp "shared preflight failed; see ${preflight_log}"
  exit 1
fi
stamp "shared preflight passed"

overall_status=0
for variant in ${VARIANTS}; do
  upper="${variant^^}"
  guard_name="S8_P5_C2_${upper}_guard_seed${SEED:-3407}"
  guard_log="${LOG_ROOT}/${guard_name}.log"
  stamp "${variant}: strict guard starts"
  if ! run_logged "${guard_log}" env \
      RECOVERY_VARIANT="${variant}" \
      EXP_NAME="${guard_name}" \
      bash run/run_S8_P5_C2_recovery_5090.sh guard; then
    stamp "${variant}: guard failed; see ${guard_log}"
    printf "%s\tguard\tfailed\t1\t%s\n" \
      "${variant}" "${guard_log}" >> "${summary}"
    overall_status=1
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit 1
    fi
    continue
  fi
  stamp "${variant}: guard passed"
  printf "%s\tguard\tpassed\t1\t%s\n" \
    "${variant}" "${guard_log}" >> "${summary}"

  smoke_name="S8_P5_C2_${upper}_smoke20_seed${SEED:-3407}"
  smoke_log="${LOG_ROOT}/${smoke_name}.log"
  stamp "${variant}: smoke starts"
  if ! run_logged "${smoke_log}" env \
      RECOVERY_VARIANT="${variant}" \
      EXP_NAME="${smoke_name}" \
      bash run/run_S8_P5_C2_recovery_5090.sh smoke; then
    stamp "${variant}: smoke failed; see ${smoke_log}"
    printf "%s\tsmoke\tfailed\t1\t%s\n" \
      "${variant}" "${smoke_log}" >> "${summary}"
    overall_status=1
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit 1
    fi
    continue
  fi
  stamp "${variant}: smoke passed"
  printf "%s\tsmoke\tpassed\t1\t%s\n" \
    "${variant}" "${smoke_log}" >> "${summary}"

  attempt=1
  completed=0
  while (( attempt <= MAX_RETRIES + 1 )); do
    tag="pilot6000"
    [[ "${MODE}" == "full" ]] && tag="full8000"
    exp_name="S8_P5_C2_${upper}_${tag}_seed${SEED:-3407}"
    run_log="${LOG_ROOT}/${exp_name}_attempt${attempt}.log"
    stamp "${variant}: ${MODE} attempt ${attempt} starts"
    if run_logged "${run_log}" env \
        RECOVERY_VARIANT="${variant}" \
        EXP_NAME="${exp_name}" \
        bash run/run_S8_P5_C2_recovery_5090.sh "${MODE}"; then
      stamp "${variant}: ${MODE} completed"
      printf "%s\t%s\tcompleted\t%s\t%s\n" \
        "${variant}" "${MODE}" "${attempt}" "${run_log}" >> "${summary}"
      completed=1
      break
    fi
    stamp "${variant}: ${MODE} attempt ${attempt} failed; see ${run_log}"
    printf "%s\t%s\tfailed\t%s\t%s\n" \
      "${variant}" "${MODE}" "${attempt}" "${run_log}" >> "${summary}"
    attempt=$((attempt + 1))
  done
  if [[ "${completed}" != "1" ]]; then
    overall_status=1
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit 1
    fi
  fi
done

stamp "sequence finished; summary=${summary}"
exit "${overall_status}"
