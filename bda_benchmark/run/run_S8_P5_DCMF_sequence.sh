#!/usr/bin/env bash
set -uo pipefail

# Fault-tolerant C0->C3 controller.  Every variant must pass smoke before the
# requested run.  A failed variant is recorded and does not hide later results.

MODE="${1:-pilot}"
case "${MODE}" in
  pilot|full|paper50k) ;;
  *)
    echo "Choose pilot|full|paper50k" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VARIANTS="${VARIANTS:-c0 c1 c2 c3}"
MAX_RETRIES="${MAX_RETRIES:-1}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/S8_P5_DCMF}"
mkdir -p "${LOG_ROOT}"

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

cd "${PROJECT_ROOT}"
stamp "shared preflight starts"
preflight_log="${LOG_ROOT}/preflight_${timestamp}.log"
if ! run_logged "${preflight_log}" bash run/run_S8_P5_DCMF_5090.sh preflight; then
  stamp "shared preflight failed; see ${preflight_log}"
  exit 1
fi
stamp "shared preflight passed"

stamp "shared stats check starts"
stats_log="${LOG_ROOT}/stats_${timestamp}.log"
if ! run_logged "${stats_log}" bash run/run_S8_P5_DCMF_5090.sh stats; then
  stamp "shared stats failed; see ${stats_log}"
  exit 1
fi
stamp "shared stats passed"

overall_status=0
for variant in ${VARIANTS}; do
  upper="${variant^^}"
  smoke_name="S8_P5_${upper}_smoke20_seed${SEED:-3407}"
  smoke_log="${LOG_ROOT}/${smoke_name}.log"
  stamp "${variant}: smoke starts"
  if ! run_logged "${smoke_log}" env \
      VARIANT="${variant}" \
      EXP_NAME="${smoke_name}" \
      bash run/run_S8_P5_DCMF_5090.sh smoke; then
    stamp "${variant}: smoke failed; see ${smoke_log}"
    printf "%s\tsmoke\tfailed\t1\t%s\n" "${variant}" "${smoke_log}" >> "${summary}"
    overall_status=1
    if [[ "${CONTINUE_ON_FAILURE}" != "1" ]]; then
      exit 1
    fi
    continue
  fi
  stamp "${variant}: smoke passed"
  printf "%s\tsmoke\tpassed\t1\t%s\n" "${variant}" "${smoke_log}" >> "${summary}"

  attempt=1
  completed=0
  while (( attempt <= MAX_RETRIES + 1 )); do
    case "${MODE}" in
      pilot) tag="pilot5000" ;;
      full) tag="full8000" ;;
      paper50k) tag="paper50000" ;;
    esac
    exp_name="S8_P5_${upper}_${tag}_seed${SEED:-3407}"
    run_log="${LOG_ROOT}/${exp_name}_attempt${attempt}.log"
    stamp "${variant}: ${MODE} attempt ${attempt} starts"
    if run_logged "${run_log}" env \
        VARIANT="${variant}" \
        EXP_NAME="${exp_name}" \
        bash run/run_S8_P5_DCMF_5090.sh "${MODE}"; then
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
