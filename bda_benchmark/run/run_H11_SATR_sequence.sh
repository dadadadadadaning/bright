#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-screen}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# The naive T1 control is complete.  T1E is now the default controlled screen;
# T2E remains opt-in until T1E passes every validation gate.
VARIANTS="${VARIANTS:-t1e}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H11_SATR}"
mkdir -p "${LOG_ROOT}"

stamp="$(date +%Y%m%d_%H%M%S)"
summary="${LOG_ROOT}/H11_SATR_sequence_${MODE}_${stamp}.tsv"
printf 'variant\tmode\tstatus\tlog\n' > "${summary}"

echo "[$(date '+%F %T')] H11 preflight starts"
bash run/run_H11_SATR_5090.sh preflight
echo "[$(date '+%F %T')] H11 T0 exact guard starts"
VARIANT=t0 EXP_NAME=H11_T0_guard_seed${SEED:-3407} \
  bash run/run_H11_SATR_5090.sh guard

for variant in ${VARIANTS}; do
  log="${LOG_ROOT}/H11_${variant^^}_${MODE}_seed${SEED:-3407}.log"
  echo "[$(date '+%F %T')] ${variant}: smoke starts"
  if ! VARIANT="${variant}" EXP_NAME="H11_${variant^^}_smoke20_seed${SEED:-3407}" \
      bash run/run_H11_SATR_5090.sh smoke > "${log}.smoke" 2>&1; then
    printf '%s\t%s\tSMOKE_FAILED\t%s\n' "${variant}" "${MODE}" "${log}.smoke" >> "${summary}"
    continue
  fi
  echo "[$(date '+%F %T')] ${variant}: ${MODE} starts"
  if VARIANT="${variant}" bash run/run_H11_SATR_5090.sh "${MODE}" > "${log}" 2>&1; then
    printf '%s\t%s\tCOMPLETED\t%s\n' "${variant}" "${MODE}" "${log}" >> "${summary}"
  else
    printf '%s\t%s\tFAILED\t%s\n' "${variant}" "${MODE}" "${log}" >> "${summary}"
  fi
done
echo "[$(date '+%F %T')] H11 sequence finished; summary=${summary}"
