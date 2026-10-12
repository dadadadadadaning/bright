#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/shared-nvme/BRIGHT-master}"
GPU0="${1:-0}"
GPU1="${2:-1}"
LOG_ROOT="$ROOT/logs/train_E1_priority3"

mkdir -p "$LOG_ROOT"
cd "$ROOT"

echo "Round 1: E1_noCons on GPU $GPU0, E1_strongerSampler on GPU $GPU1"
bash "$ROOT/scripts/train/train_E1_noCons_Sampler.sh" "$GPU0" \
  > "$LOG_ROOT/E1_noCons_Sampler_gpu0.log" 2>&1 &
PID0=$!

bash "$ROOT/scripts/train/train_E1_strongerSampler.sh" "$GPU1" \
  > "$LOG_ROOT/E1_strongerSampler_gpu1.log" 2>&1 &
PID1=$!

wait "$PID0"
wait "$PID1"

echo "Round 2: E1_noCons_strongerSampler on GPU $GPU0"
bash "$ROOT/scripts/train/train_E1_noCons_strongerSampler.sh" "$GPU0" \
  > "$LOG_ROOT/E1_noCons_strongerSampler_gpu0.log" 2>&1

echo "E1 priority3 training queue finished."
