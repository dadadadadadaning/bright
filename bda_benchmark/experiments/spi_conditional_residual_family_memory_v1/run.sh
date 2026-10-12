#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-preflight}"
case "$MODE" in preflight|audit|d2|replay-d2|verify|auto) ;; *) echo "Choose preflight|audit|d2|replay-d2|verify|auto" >&2; exit 2;; esac
BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"; PROJECT_ROOT="$(dirname "$BDA_ROOT")"
export PYTHONPATH="$PROJECT_ROOT:$BDA_ROOT:${PYTHONPATH:-}"
HERE="$BDA_ROOT/experiments/spi_conditional_residual_family_memory_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"; [[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
echo "Conditional Residual Family Memory Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
preflight() {
  "$PYTHON_BIN" -m py_compile "$HERE/core.py" "$HERE/audit.py" "$HERE/d2_attribution.py" "$HERE/verify_results.py" "$HERE/test_core.py"
  "$PYTHON_BIN" -m pytest -q "$HERE/test_core.py"; bash -n "$HERE/run.sh"
  echo "Conditional Residual Family Memory preflight passed"
}
[[ "$MODE" == preflight ]] && { preflight; exit 0; }
SEED="${SEED:-7341}"
CARD_ROOT="${CARD_ROOT:-/root/data/BRIGHT-checkpoints/SPI_INDEPENDENT_BUILDING_MEMORY_V1/seed${SEED}}"
HRM_ROOT="${HRM_ROOT:-/root/data/BRIGHT-checkpoints/SPI_HIERARCHICAL_RESIDUAL_MEMORY_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_CONDITIONAL_RESIDUAL_FAMILY_MEMORY_V1}"
TRAIN="$CARD_ROOT/meta_train_cards.pt"; CAL="$CARD_ROOT/meta_calibration_cards.pt"; METRIC="$HRM_ROOT/error_focused_metric.pth"
mkdir -p "$RESULT_ROOT"
for path in "$TRAIN" "$CAL" "$METRIC"; do test -s "$path" || { echo "Missing prerequisite: $path" >&2; exit 1; }; done
audit() {
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_conditional_residual_family_memory_v1.audit \
    --train-cards "$TRAIN" --calibration-cards "$CAL" --metric-checkpoint "$METRIC" --result-root "$RESULT_ROOT" \
    --device "${DEVICE:-cuda:0}" --risk-quantile "${RISK_QUANTILE:-0.80}" --capacity-per-event-source "${MEMORY_CAPACITY:-256}" \
    --coarse-k "${COARSE_K:-64}" --topk "${MEMORY_TOPK:-8}" --maximum-per-event "${MAXIMUM_PER_EVENT:-2}" \
    --quality-limit "${QUALITY_LIMIT:-3.0}" --area-ratio-limit "${AREA_RATIO_LIMIT:-2.0}" \
    --head-steps "${HEAD_STEPS:-600}" --head-batch-size "${HEAD_BATCH_SIZE:-512}" --learning-rate "${LEARNING_RATE:-0.001}" --seed "$SEED"
}
verify() { "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_conditional_residual_family_memory_v1.verify_results --result-root "$RESULT_ROOT" --seed "$SEED"; }
d2() { "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_conditional_residual_family_memory_v1.d2_attribution --prediction-cache "$RESULT_ROOT/crfm_conditional_family_predictions_seed${SEED}.pt" --output "$RESULT_ROOT/crfm_d2_attribution_seed${SEED}.json"; }
case "$MODE" in audit) audit;; d2) d2;; replay-d2) audit; d2;; verify) verify;; auto) preflight; audit; verify;; esac
