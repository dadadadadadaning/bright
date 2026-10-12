#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
case "$MODE" in preflight|train|audit|verify|auto) ;; *) echo "Choose preflight|train|audit|verify|auto" >&2; exit 2;; esac

BDA_ROOT="${BDA_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
PROJECT_ROOT="$(dirname "$BDA_ROOT")"
export PYTHONPATH="$PROJECT_ROOT:$BDA_ROOT:${PYTHONPATH:-}"
HERE="$BDA_ROOT/experiments/spi_qsd_residual_case_memory_v1"
PYTHON_BIN="${PYTHON_BIN:-/root/data/conda_envs/bright4090/bin/python}"
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=python
echo "QSD Residual Case Memory Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"

preflight() {
  "$PYTHON_BIN" -m py_compile "$HERE/core.py" "$HERE/train_metric.py" "$HERE/audit.py" "$HERE/verify_results.py" "$HERE/test_core.py"
  "$PYTHON_BIN" -m pytest -q "$HERE/test_core.py"
  bash -n "$HERE/run.sh"
  echo "QSD Residual Case Memory preflight passed"
}
[[ "$MODE" == preflight ]] && { preflight; exit 0; }

SEED="${SEED:-7341}"
CARD_ROOT="${CARD_ROOT:-/root/data/BRIGHT-checkpoints/SPI_INDEPENDENT_BUILDING_MEMORY_V1/seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_QSD_RESIDUAL_CASE_MEMORY_V1/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_QSD_RESIDUAL_CASE_MEMORY_V1}"
CONDITIONAL_RESULT="${CONDITIONAL_RESULT:-/root/data/BRIGHT-eval-results/SPI_CONDITIONAL_MEMORY_SCREEN_V2/conditional_memory_screen_v2_seed${SEED}.json}"
TRAIN_CARDS="$CARD_ROOT/meta_train_cards.pt"
CAL_CARDS="$CARD_ROOT/meta_calibration_cards.pt"
METRIC="$OUTPUT_ROOT/qsd_factorized_metric.pth"
mkdir -p "$OUTPUT_ROOT" "$RESULT_ROOT"
for path in "$TRAIN_CARDS" "$CAL_CARDS"; do
  test -s "$path" || { echo "Missing prerequisite: $path" >&2; exit 1; }
done

train_metric() {
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_qsd_residual_case_memory_v1.train_metric \
    --train-cards "$TRAIN_CARDS" --output "$METRIC" --device "${DEVICE:-cuda:0}" \
    --hidden "${MEMORY_HIDDEN:-64}" --steps "${METRIC_STEPS:-1000}" \
    --batch-size "${BATCH_SIZE:-512}" --learning-rate "${LEARNING_RATE:-0.001}" \
    --minimum-purity "${MINIMUM_LABEL_PURITY:-0.60}" --progress-every "${PROGRESS_EVERY:-100}" --seed "$SEED"
}

audit() {
  test -s "$METRIC" || { echo "Missing trained metric: $METRIC" >&2; exit 1; }
  local optional=()
  [[ -s "$CONDITIONAL_RESULT" ]] && optional+=(--conditional-result "$CONDITIONAL_RESULT")
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_qsd_residual_case_memory_v1.audit \
    --train-cards "$TRAIN_CARDS" --calibration-cards "$CAL_CARDS" \
    --metric-checkpoint "$METRIC" --result-root "$RESULT_ROOT" "${optional[@]}" \
    --device "${DEVICE:-cuda:0}" --capacity-per-event-source "${MEMORY_CAPACITY:-256}" \
    --coarse-k "${COARSE_K:-64}" --topk "${MEMORY_TOPK:-8}" \
    --maximum-per-event "${MAXIMUM_PER_EVENT:-2}" --quality-limit "${QUALITY_LIMIT:-3.0}" \
    --area-ratio-limit "${AREA_RATIO_LIMIT:-2.0}" --alpha "${MEMORY_ALPHA:-0.10}" \
    --minimum-similarity "${MINIMUM_SIMILARITY:-0.20}" --minimum-consensus "${MINIMUM_CONSENSUS:-0.75}" \
    --minimum-events "${MINIMUM_EVENTS:-3}" --minimum-ess "${MINIMUM_ESS:-3.0}" \
    --maximum-quality-distance "${MAXIMUM_QUALITY_DISTANCE:-2.0}" --seed "$SEED"
}

case "$MODE" in
  train) preflight; train_metric ;;
  audit) audit ;;
  verify) "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_qsd_residual_case_memory_v1.verify_results --result-root "$RESULT_ROOT" --seed "$SEED" ;;
  auto) preflight; [[ -s "$METRIC" ]] || train_metric; audit; "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_qsd_residual_case_memory_v1.verify_results --result-root "$RESULT_ROOT" --seed "$SEED" ;;
esac

