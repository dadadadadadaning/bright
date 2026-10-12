#!/usr/bin/env bash
set -euo pipefail
PHASE="${1:-probe}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# full_joint is intentionally first.
VARIANTS=(
  full_joint
  bir_on_frozen_c5
  bir_shuffle_on_frozen_c5
  rgb_only_building
  raw_sar_building
  no_conf_building
  shifted_sar_building
  wrong_building_joint
)

for variant in "${VARIANTS[@]}"; do
  echo "===== ${variant} ${PHASE} START $(date -Iseconds) ====="
  bash "${HERE}/run.sh" "${PHASE}" "${variant}"
  echo "===== ${variant} ${PHASE} COMPLETE $(date -Iseconds) ====="
done
