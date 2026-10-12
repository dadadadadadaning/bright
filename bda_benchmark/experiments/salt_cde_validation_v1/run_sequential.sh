#!/usr/bin/env bash
set -euo pipefail

# Sequential single-GPU launcher. Examples:
#   bash experiments/salt_cde_validation_v1/run_sequential.sh pilot
#   bash experiments/salt_cde_validation_v1/run_sequential.sh pilot d1 d2 d3
#   nohup bash experiments/salt_cde_validation_v1/run_sequential.sh short d2 d3 &

STAGE="${1:-pilot}"
shift || true

case "${STAGE}" in
  smoke|probe|pilot|short|long|final) ;;
  *) echo "stage must be smoke|probe|pilot|short|long|final" >&2; exit 2 ;;
esac

if (( $# > 0 )); then
  VARIANTS=("$@")
else
  VARIANTS=(d0 d0n d1 d2 d3)
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-/root/data/BRIGHT-train-logs/SALT_CDE_VALIDATION_V1}"
mkdir -p "${LOG_ROOT}"

case "${STAGE}" in
  smoke) TAG=smoke20 ;;
  probe) TAG=probe2000 ;;
  pilot) TAG=pilot8000 ;;
  short) TAG=short20000 ;;
  long) TAG=long35000 ;;
  final) TAG=final50000 ;;
esac

MASTER_LOG="${LOG_ROOT}/sequence_${TAG}_seed${SEED:-3407}.log"
touch "${MASTER_LOG}"

echo "Sequence stage=${STAGE} variants=${VARIANTS[*]}" | tee -a "${MASTER_LOG}"
echo "Started at $(date -Is)" | tee -a "${MASTER_LOG}"

for variant in "${VARIANTS[@]}"; do
  case "${variant}" in
    d0|d0n|d1|d2|d3|d2r|d3v2) ;;
    *) echo "Unknown variant: ${variant}" | tee -a "${MASTER_LOG}"; exit 2 ;;
  esac
  MODEL_LOG="${LOG_ROOT}/${variant}_${TAG}_seed${SEED:-3407}.log"
  echo "===== ${variant} ${STAGE} START $(date -Is) =====" | tee -a "${MASTER_LOG}"
  set +e
  bash "${SCRIPT_DIR}/run.sh" "${STAGE}" "${variant}" > "${MODEL_LOG}" 2>&1
  status=$?
  set -e
  if (( status != 0 )); then
    echo "===== ${variant} FAILED exit=${status} $(date -Is) =====" | tee -a "${MASTER_LOG}"
    echo "Last 40 lines: ${MODEL_LOG}" | tee -a "${MASTER_LOG}"
    tail -n 40 "${MODEL_LOG}" | tee -a "${MASTER_LOG}"
    exit "${status}"
  fi
  summary_line="$(grep -E 'Training complete|Completed experiment exists' "${MODEL_LOG}" | tail -n 1 || true)"
  echo "===== ${variant} COMPLETE $(date -Is) =====" | tee -a "${MASTER_LOG}"
  [[ -z "${summary_line}" ]] || echo "${summary_line}" | tee -a "${MASTER_LOG}"
done

echo "All requested experiments completed at $(date -Is)" | tee -a "${MASTER_LOG}"
