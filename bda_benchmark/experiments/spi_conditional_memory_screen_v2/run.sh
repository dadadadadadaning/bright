#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "$MODE" in preflight|audit|auto) ;; *) echo "Choose preflight|audit|auto" >&2; exit 2;; esac
BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "$BDA_ROOT")"
export PYTHONPATH="$PROJECT_ROOT:$BDA_ROOT:${PYTHONPATH:-}"
HERE="$BDA_ROOT/experiments/spi_conditional_memory_screen_v2"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python

preflight() {
  "$PYTHON_BIN" -m py_compile "$HERE/core.py" "$HERE/audit.py" "$HERE/test_core.py"
  "$PYTHON_BIN" -m pytest -q "$HERE/test_core.py"
  bash -n "$HERE/run.sh"
  echo "Conditional Memory Screen V2 preflight passed"
}
[[ "$MODE" == preflight ]] && { preflight; exit 0; }

SEED="${SEED:-7341}"
CARD_ROOT="${CARD_ROOT:-/root/data/BRIGHT-checkpoints/SPI_INDEPENDENT_BUILDING_MEMORY_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_CONDITIONAL_MEMORY_SCREEN_V2}"
TRAIN="$CARD_ROOT/meta_train_cards.pt"
CAL="$CARD_ROOT/meta_calibration_cards.pt"
for path in "$TRAIN" "$CAL"; do test -s "$path" || { echo "Missing: $path" >&2; exit 1; }; done
mkdir -p "$RESULT_ROOT"

audit() {
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_conditional_memory_screen_v2.audit \
    --train-cards "$TRAIN" --calibration-cards "$CAL" \
    --output "$RESULT_ROOT/conditional_memory_screen_v2_seed${SEED}.json" \
    --device "${DEVICE:-cuda:0}" --hidden "${HIDDEN:-128}" --steps "${STEPS:-800}" \
    --batch-size "${BATCH_SIZE:-512}" --learning-rate "${LEARNING_RATE:-0.001}" \
    --capacity-per-event-class "${CAPACITY_PER_EVENT_CLASS:-256}" \
    --coarse-k "${COARSE_K:-64}" --topk "${TOPK:-8}" --beta "${BETA:-0.25}" \
    --progress-every "${PROGRESS_EVERY:-100}" --seed "$SEED"
}

case "$MODE" in audit) audit ;; auto) preflight; audit ;; esac

