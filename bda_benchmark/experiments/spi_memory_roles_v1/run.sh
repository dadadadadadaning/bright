#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "$MODE" in preflight|smoke|auto) ;; *) echo "Choose preflight|smoke|auto" >&2; exit 2;; esac
BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "$BDA_ROOT")"
export PYTHONPATH="$PROJECT_ROOT:$BDA_ROOT:${PYTHONPATH:-}"
HERE="$BDA_ROOT/experiments/spi_memory_roles_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
echo "Memory Roles Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"

preflight() {
  "$PYTHON_BIN" -m py_compile "$HERE/core.py" "$HERE/train_all.py" "$HERE/test_core.py"
  "$PYTHON_BIN" -m pytest -q "$HERE/test_core.py"
  bash -n "$HERE/run.sh"
  echo "Memory Roles V1 preflight passed"
}
[[ "$MODE" == preflight ]] && { preflight; exit 0; }

SEED="${SEED:-7341}"
CARD_ROOT="${CARD_ROOT:-/root/data/BRIGHT-checkpoints/SPI_COMPLETE_BUILDING_MEMORY_V1_SHARDED/seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_MEMORY_ROLES_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_MEMORY_ROLES_V1}"
TRAIN="$CARD_ROOT/meta_train_cards.pt"
CAL="$CARD_ROOT/meta_calibration_cards.pt"
test -s "$TRAIN" || { echo "Missing complete-building train cards: $TRAIN" >&2; exit 1; }
test -s "$CAL" || { echo "Missing complete-building calibration cards: $CAL" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT" "$RESULT_ROOT"

COMMON=(
  --train-cards "$TRAIN" --calibration-cards "$CAL" --output-root "$OUTPUT_ROOT" --result-root "$RESULT_ROOT"
  --hidden "${MEMORY_HIDDEN:-128}" --metric-steps "${METRIC_STEPS:-1000}"
  --prototype-steps "${PROTOTYPE_STEPS:-800}" --role-steps "${ROLE_STEPS:-500}"
  --router-steps "${ROUTER_STEPS:-500}" --sampler-steps "${SAMPLER_STEPS:-800}"
  --batch-size "${BATCH_SIZE:-512}" --role-batch-size "${ROLE_BATCH_SIZE:-256}"
  --learning-rate "${LEARNING_RATE:-0.001}" --topk 8 --maximum-per-event 2
  --maximum-per-event-class "${MAXIMUM_PER_EVENT_CLASS:-256}"
  --device "${DEVICE:-cuda:0}" --progress-every "${PROGRESS_EVERY:-100}" --seed "$SEED"
)

if [[ "$MODE" == smoke ]]; then
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_memory_roles_v1.train_all \
    "${COMMON[@]}" --metric-steps 2 --prototype-steps 2 --role-steps 2 \
    --router-steps 2 --sampler-steps 2 --max-train-rows "${SMOKE_TRAIN_ROWS:-2048}" \
    --max-calibration-rows "${SMOKE_CAL_ROWS:-512}"
  echo "Memory Roles V1 smoke complete"
else
  preflight
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_memory_roles_v1.train_all "${COMMON[@]}"
fi
