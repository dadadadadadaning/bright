#!/usr/bin/env bash
set -uo pipefail

MODE="${1:-pilot}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

VARIANTS="${VARIANTS:-h0 h1 h2 h3}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-1}"
MAX_RETRIES="${MAX_RETRIES:-1}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/HeterogeneousDamageFormer}"
mkdir -p "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${LOG_ROOT}/sequence_${MODE}_${STAMP}.tsv"
printf "variant\tstage\tstatus\tlog\n" > "${SUMMARY}"

report() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

report "shared preflight starts"
if ! bash run/run_HeterogeneousDamageFormer_5090.sh preflight; then
  report "shared preflight failed"
  exit 1
fi
report "shared preflight passed"

report "train-only SAR stats check starts"
if ! bash run/run_HeterogeneousDamageFormer_5090.sh stats; then
  report "stats failed"
  exit 1
fi
report "stats passed"

overall_status=0
for variant in ${VARIANTS}; do
  smoke_log="${LOG_ROOT}/HDF_${variant^^}_smoke20_seed3407.log"
  report "${variant}: smoke starts"
  if ! env VARIANT="${variant}" bash run/run_HeterogeneousDamageFormer_5090.sh smoke \
      > "${smoke_log}" 2>&1; then
    report "${variant}: smoke failed; see ${smoke_log}"
    printf "%s\tsmoke\tfailed\t%s\n" "${variant}" "${smoke_log}" >> "${SUMMARY}"
    overall_status=1
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] && continue
    exit 1
  fi
  report "${variant}: smoke passed"
  printf "%s\tsmoke\tpassed\t%s\n" "${variant}" "${smoke_log}" >> "${SUMMARY}"

  attempt=0
  completed=0
  while (( attempt <= MAX_RETRIES )); do
    attempt=$((attempt + 1))
    run_log="${LOG_ROOT}/HDF_${variant^^}_${MODE}_seed3407_attempt${attempt}.log"
    report "${variant}: ${MODE} attempt ${attempt} starts"
    if env VARIANT="${variant}" bash run/run_HeterogeneousDamageFormer_5090.sh "${MODE}" \
        > "${run_log}" 2>&1; then
      report "${variant}: ${MODE} completed"
      printf "%s\t%s\tcompleted\t%s\n" "${variant}" "${MODE}" "${run_log}" >> "${SUMMARY}"
      completed=1
      break
    fi
    report "${variant}: ${MODE} attempt ${attempt} failed; see ${run_log}"
  done
  if (( completed == 0 )); then
    printf "%s\t%s\tfailed\t%s\n" "${variant}" "${MODE}" "${run_log}" >> "${SUMMARY}"
    overall_status=1
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  fi
done

report "sequence finished; summary=${SUMMARY}"
exit "${overall_status}"
