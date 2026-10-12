#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-probe}"
shift || true
if (( $# > 0 )); then
  VARIANTS=("$@")
else
  VARIANTS=(a0 a1 a2 a3)
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG_ROOT="${LOG_ROOT:-/root/data/BRIGHT-train-logs/P3_DAMAGEFORMER_ABLATION_V1}"
mkdir -p "${LOG_ROOT}"
MASTER="${LOG_ROOT}/sequence_${STAGE}_seed${SEED:-3407}.log"
echo "Stage=${STAGE}; variants=${VARIANTS[*]}; started=$(date -Is)" | tee -a "${MASTER}"
for variant in "${VARIANTS[@]}"; do
  case "${variant}" in a0|a1|a2|a3) ;; *) echo "unknown variant ${variant}" | tee -a "${MASTER}"; exit 2 ;; esac
  log="${LOG_ROOT}/${variant}_${STAGE}_seed${SEED:-3407}.log"
  echo "===== ${variant} START $(date -Is) =====" | tee -a "${MASTER}"
  if ! bash "${HERE}/run.sh" "${STAGE}" "${variant}" >"${log}" 2>&1; then
    echo "===== ${variant} FAILED; see ${log} =====" | tee -a "${MASTER}"
    tail -n 40 "${log}" | tee -a "${MASTER}"
    exit 1
  fi
  echo "===== ${variant} COMPLETE $(date -Is) =====" | tee -a "${MASTER}"
done
case "${STAGE}" in
  smoke) DEFAULT_TAG=smoke20 ;;
  probe) DEFAULT_TAG=probe5000 ;;
  confirm) DEFAULT_TAG=confirm20000 ;;
  long) DEFAULT_TAG=long35000 ;;
  full) DEFAULT_TAG=full50000 ;;
  *) echo "unsupported stage ${STAGE}" >&2; exit 2 ;;
esac
TRAIN_TAG="${TRAIN_TAG:-${DEFAULT_TAG}}" \
  bash "${HERE}/run.sh" summarize a0 | tee -a "${MASTER}"
echo "All complete $(date -Is)" | tee -a "${MASTER}"
