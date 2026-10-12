#!/usr/bin/env bash
set -euo pipefail
PHASE="${1:-smoke}"
shift || true
VARIANTS=("$@")
if [[ ${#VARIANTS[@]} -eq 0 ]]; then
  VARIANTS=(c1_lca c2_slico c3_rgb_assoc c4_raw_sar_assoc c5_aligned_assoc c6_no_confidence c7_shuffled_sar c8_damage_hint c9_boundary)
fi
for variant in "${VARIANTS[@]}"; do
  echo "===== ${variant} ${PHASE} START $(date -Iseconds) ====="
  bash "$(dirname "$0")/run.sh" "${PHASE}" "${variant}"
  echo "===== ${variant} ${PHASE} COMPLETE $(date -Iseconds) ====="
done

