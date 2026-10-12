#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "$MODE" in preflight|smoke|audit|auto) ;; *) echo "Choose preflight|smoke|audit|auto" >&2; exit 2;; esac
BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "$BDA_ROOT")"
export PYTHONPATH="$PROJECT_ROOT:$BDA_ROOT:${PYTHONPATH:-}"
HERE="$BDA_ROOT/experiments/spi_memory_dependency_audit_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
echo "Memory Dependency Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"

preflight() {
  "$PYTHON_BIN" -m py_compile "$HERE/core.py" "$HERE/audit.py" "$HERE/test_core.py"
  "$PYTHON_BIN" -m pytest -q "$HERE/test_core.py"
  bash -n "$HERE/run.sh"
  echo "Memory Dependency Audit V1 preflight passed"
}
[[ "$MODE" == preflight ]] && { preflight; exit 0; }

SEED="${SEED:-7341}"
CARD_ROOT="${CARD_ROOT:-/root/data/BRIGHT-checkpoints/SPI_COMPLETE_BUILDING_MEMORY_V1_SHARDED/seed${SEED}}"
MEMORY_ROLES_ROOT="${MEMORY_ROLES_ROOT:-/root/data/BRIGHT-checkpoints/SPI_MEMORY_ROLES_V1/seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_MEMORY_DEPENDENCY_AUDIT_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_MEMORY_DEPENDENCY_AUDIT_V1}"
TRAIN="$CARD_ROOT/meta_train_cards.pt"
CAL="$CARD_ROOT/meta_calibration_cards.pt"
TEACHER="$MEMORY_ROLES_ROOT/r_m1_b2_learned_full.pth"
for path in "$TRAIN" "$CAL" "$TEACHER"; do
  test -s "$path" || { echo "Missing prerequisite: $path" >&2; exit 1; }
done
mkdir -p "$OUTPUT_ROOT" "$RESULT_ROOT"

COMMON=(
  --train-cards "$TRAIN" --calibration-cards "$CAL" --teacher-checkpoint "$TEACHER"
  --output-root "$OUTPUT_ROOT" --result-root "$RESULT_ROOT"
  --hidden "${MEMORY_HIDDEN:-128}" --role-steps "${ROLE_STEPS:-500}"
  --router-steps "${ROUTER_STEPS:-500}" --batch-size "${BATCH_SIZE:-512}"
  --role-batch-size "${ROLE_BATCH_SIZE:-256}" --learning-rate "${LEARNING_RATE:-0.001}"
  --topk 8 --maximum-per-event 2 --maximum-per-event-class "${MAXIMUM_PER_EVENT_CLASS:-256}"
  --target-route-fraction "${TARGET_ROUTE_FRACTION:-0.20}"
  --device "${DEVICE:-cuda:0}" --progress-every "${PROGRESS_EVERY:-100}" --seed "$SEED"
)

if [[ "$MODE" == smoke ]]; then
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_memory_dependency_audit_v1.audit \
    "${COMMON[@]}" --role-steps 2 --router-steps 2 \
    --max-train-rows "${SMOKE_TRAIN_ROWS:-2048}" \
    --max-calibration-rows "${SMOKE_CAL_ROWS:-512}"
  echo "Memory Dependency Audit V1 smoke complete"
else
  [[ "$MODE" == auto ]] && preflight
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_memory_dependency_audit_v1.audit "${COMMON[@]}"
fi
