#!/usr/bin/env bash
set -euo pipefail

BASE="/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16"
S9I="${BASE}/S9I_CenterBoundary"

for required in \
  "${S9I}/best_val_instance_f1.pth" \
  "${S9I}/s9i_final_val.json" \
  "${S9I}/training_history.json"
do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: required S9-I file missing: ${required}" >&2
    exit 2
  fi
done

LOCKED_JSON="${S9I}/S9I_STEP2000_FINAL_VALIDATION_LOCKED.json"
SUMS="${S9I}/S9I_STEP2000_LOCKED_SHA256SUMS.txt"

if [[ -e "${LOCKED_JSON}" || -e "${SUMS}" ]]; then
  echo "ERROR: lock outputs already exist; refusing to overwrite." >&2
  echo "  ${LOCKED_JSON}" >&2
  echo "  ${SUMS}" >&2
  exit 3
fi

cp --no-clobber \
  "${S9I}/s9i_final_val.json" \
  "${LOCKED_JSON}"

sha256sum \
  "${S9I}/best_val_instance_f1.pth" \
  "${LOCKED_JSON}" \
  "${S9I}/training_history.json" \
  > "${SUMS}"

echo "S9-I result locked:"
cat "${SUMS}"
