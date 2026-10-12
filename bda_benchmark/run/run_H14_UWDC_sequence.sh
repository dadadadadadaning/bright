#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-screen}"
VARIANTS="${VARIANTS:-u1}"
SEED="${SEED:-3407}"
OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-1500}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H14_UWDC}"
mkdir -p "${LOG_ROOT}"

bash run/run_H14_UWDC_5090.sh preflight
bash run/run_H14_UWDC_5090.sh guard

failures=0
for variant in ${VARIANTS}; do
  upper="${variant^^}"
  echo "[$(date '+%F %T')] ${variant}: smoke starts"
  if ! env VARIANT="${variant}" EXP_NAME="H14_${upper}_smoke20_seed${SEED}" \
      bash run/run_H14_UWDC_5090.sh smoke >"${LOG_ROOT}/H14_${upper}_smoke20_seed${SEED}.log" 2>&1; then
    echo "[$(date '+%F %T')] ${variant}: smoke failed"
    failures=$((failures + 1)); continue
  fi
  echo "[$(date '+%F %T')] ${variant}: controlled screen starts"
  if ! env VARIANT="${variant}" OPTIMIZER_STEPS="${OPTIMIZER_STEPS}" \
      EXP_NAME="H14_${upper}_screen${OPTIMIZER_STEPS}_seed${SEED}" \
      bash run/run_H14_UWDC_5090.sh "${MODE}" \
      >"${LOG_ROOT}/H14_${upper}_screen${OPTIMIZER_STEPS}_seed${SEED}.log" 2>&1; then
    echo "[$(date '+%F %T')] ${variant}: screen failed"
    failures=$((failures + 1)); continue
  fi
  echo "[$(date '+%F %T')] ${variant}: screen completed"
done
echo "[$(date '+%F %T')] H14 sequence finished; failures=${failures}"
exit "${failures}"
