#!/usr/bin/env bash
set -euo pipefail
PHASE="${1:-probe}"
shift || true
VARIANTS=("${@:-a2_lca3 a3_conf3 a4_orient3 a5_conf5}")
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG_ROOT="${LOG_ROOT:-/root/data/BRIGHT-train-logs/DAMAGEFORMER_ROI_LCA_V1}"
mkdir -p "${LOG_ROOT}"
echo "Sequence phase=${PHASE} variants=${VARIANTS[*]} start=$(date -Iseconds)"
for variant in ${VARIANTS[*]}; do
  echo "===== ${variant} ${PHASE} START $(date -Iseconds) ====="
  bash "${HERE}/run.sh" "${PHASE}" "${variant}" > "${LOG_ROOT}/${variant}_${PHASE}.log" 2>&1
  echo "===== ${variant} ${PHASE} COMPLETE $(date -Iseconds) ====="
done
echo "ALL COMPLETE $(date -Iseconds)"
