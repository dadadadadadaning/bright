#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if (( $# )); then variants=("$@"); else variants=(w0_full_residual w1_id_calibration w2_id_logit w3_id_mass w5_logit4_kd); fi
for variant in "${variants[@]}"; do
  echo "===== ${variant} START $(date -Iseconds) ====="
  bash "${HERE}/run.sh" stage "${variant}"
  echo "===== ${variant} COMPLETE $(date -Iseconds) ====="
done
bash "${HERE}/run.sh" summarize
