#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-screen}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

VARIANTS="${VARIANTS:-s1 s2}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H13_SPER}"
mkdir -p "${LOG_ROOT}"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] H13 preflight starts"
bash run/run_H13_SPER_5090.sh preflight
echo "[$(stamp)] H13 preflight passed"
echo "[$(stamp)] H13 validation-only partition audit starts"
bash run/run_H13_SPER_5090.sh audit_val
echo "[$(stamp)] H13 partition audit completed"
echo "[$(stamp)] S0 exact H6-A2 guard starts"
bash run/run_H13_SPER_5090.sh guard
echo "[$(stamp)] S0 guard passed"

failures=0
for variant in ${VARIANTS}; do
  upper="${variant^^}"
  smoke_log="${LOG_ROOT}/H13_${upper}_smoke20_seed${SEED:-3407}.log"
  train_log="${LOG_ROOT}/H13_${upper}_${MODE}${OPTIMIZER_STEPS:-1500}_seed${SEED:-3407}.log"
  echo "[$(stamp)] ${variant}: smoke starts"
  if ! env VARIANT="${variant}" EXP_NAME="H13_${upper}_smoke20_seed${SEED:-3407}" \
      bash run/run_H13_SPER_5090.sh smoke >"${smoke_log}" 2>&1; then
    echo "[$(stamp)] ${variant}: smoke failed; see ${smoke_log}"
    failures=$((failures + 1))
    continue
  fi
  echo "[$(stamp)] ${variant}: smoke passed"
  echo "[$(stamp)] ${variant}: controlled screen starts"
  if ! env VARIANT="${variant}" EXP_NAME="H13_${upper}_screen${OPTIMIZER_STEPS:-1500}_seed${SEED:-3407}" \
      bash run/run_H13_SPER_5090.sh "${MODE}" >"${train_log}" 2>&1; then
    echo "[$(stamp)] ${variant}: screen failed; see ${train_log}"
    failures=$((failures + 1))
    continue
  fi
  echo "[$(stamp)] ${variant}: screen completed"
done

echo "[$(stamp)] H13 sequence finished; failures=${failures}"
exit "${failures}"
