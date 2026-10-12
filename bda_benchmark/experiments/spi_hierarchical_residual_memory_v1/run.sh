#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-preflight}"
case "$MODE" in preflight|train|audit|verify|auto) ;; *) echo "Choose preflight|train|audit|verify|auto" >&2; exit 2;; esac
BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"; PROJECT_ROOT="$(dirname "$BDA_ROOT")"
export PYTHONPATH="$PROJECT_ROOT:$BDA_ROOT:${PYTHONPATH:-}"
HERE="$BDA_ROOT/experiments/spi_hierarchical_residual_memory_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"; [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
echo "Hierarchical Residual Memory Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
preflight() {
  "$PYTHON_BIN" -m py_compile "$HERE/core.py" "$HERE/train_metric.py" "$HERE/audit.py" "$HERE/verify_results.py" "$HERE/test_core.py"
  "$PYTHON_BIN" -m pytest -q "$HERE/test_core.py"; bash -n "$HERE/run.sh"
  echo "Hierarchical Residual Memory preflight passed"
}
[[ "$MODE" == preflight ]] && { preflight; exit 0; }
SEED="${SEED:-7341}"
CARD_ROOT="${CARD_ROOT:-/root/data/BRIGHT-checkpoints/SPI_INDEPENDENT_BUILDING_MEMORY_V1/seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_HIERARCHICAL_RESIDUAL_MEMORY_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_HIERARCHICAL_RESIDUAL_MEMORY_V1}"
TRAIN="$CARD_ROOT/meta_train_cards.pt"; CAL="$CARD_ROOT/meta_calibration_cards.pt"; METRIC="$OUTPUT_ROOT/error_focused_metric.pth"
mkdir -p "$OUTPUT_ROOT" "$RESULT_ROOT"
for path in "$TRAIN" "$CAL"; do test -s "$path" || { echo "Missing prerequisite: $path" >&2; exit 1; }; done
train() {
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_hierarchical_residual_memory_v1.train_metric \
    --train-cards "$TRAIN" --output "$METRIC" --device "${DEVICE:-cuda:0}" --hidden "${MEMORY_HIDDEN:-64}" \
    --steps "${METRIC_STEPS:-1000}" --batch-size "${BATCH_SIZE:-512}" --learning-rate "${LEARNING_RATE:-0.001}" \
    --minimum-purity "${MINIMUM_LABEL_PURITY:-0.60}" --progress-every "${PROGRESS_EVERY:-100}" --seed "$SEED"
}
audit() {
  test -s "$METRIC" || { echo "Missing trained metric: $METRIC" >&2; exit 1; }
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_hierarchical_residual_memory_v1.audit \
    --train-cards "$TRAIN" --calibration-cards "$CAL" --metric-checkpoint "$METRIC" --result-root "$RESULT_ROOT" \
    --device "${DEVICE:-cuda:0}" --capacity-per-event-source "${MEMORY_CAPACITY:-256}" --coarse-k "${COARSE_K:-64}" \
    --topk "${MEMORY_TOPK:-8}" --maximum-per-event "${MAXIMUM_PER_EVENT:-2}" --quality-limit "${QUALITY_LIMIT:-3.0}" \
    --area-ratio-limit "${AREA_RATIO_LIMIT:-2.0}" --score-temperature "${SCORE_TEMPERATURE:-0.10}" --seed "$SEED"
}
verify() { "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_hierarchical_residual_memory_v1.verify_results --result-root "$RESULT_ROOT" --seed "$SEED"; }
case "$MODE" in train) preflight; train;; audit) audit;; verify) verify;; auto) preflight; [[ -s "$METRIC" ]] || train; audit; verify;; esac

