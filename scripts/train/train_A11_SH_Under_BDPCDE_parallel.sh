#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/shared-nvme/BRIGHT-master}"
GPU0="${1:-0}"
GPU1="${2:-1}"

cd "$ROOT"
mkdir -p logs/train_BDPCDE

bash scripts/train/train_A11_SH_Under_BDPCDE_B.sh "$GPU0" &
PID0=$!
bash scripts/train/train_A11_SH_Under_BDPCDE_BD.sh "$GPU1" &
PID1=$!

set +e
wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?
set -e

if [[ "$STATUS0" -ne 0 || "$STATUS1" -ne 0 ]]; then
  echo "One or more BDPCDE experiments failed: status0=$STATUS0 status1=$STATUS1"
  exit 1
fi
