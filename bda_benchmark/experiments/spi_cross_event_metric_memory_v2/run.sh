#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BDA_ROOT="$(cd "$HERE/../.." && pwd)"
REPO_ROOT="$(cd "$BDA_ROOT/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$BDA_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SEED="${SEED:-7341}"
CACHE_ROOT="${D1_CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed7341}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CROSS_EVENT_METRIC_MEMORY_V2/seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_CROSS_EVENT_METRIC_MEMORY_V2}"
D0_JSON="$RESULT_ROOT/d0_label_retrieval_seed${SEED}.json"
M1_JSON="$OUTPUT_ROOT/m1_screen.json"
M2_JSON="$RESULT_ROOT/m2_hierarchical_memory_seed${SEED}.json"
RESCUE_JSON="$RESULT_ROOT/m1r_selective_rescue_seed${SEED}.json"

preflight() {
  "$PYTHON_BIN" -m pytest -q "$HERE/test_core.py"
  "$PYTHON_BIN" -m py_compile "$HERE/core.py" "$HERE/audit_d0.py" \
    "$HERE/train_m1.py" "$HERE/audit_m2.py" \
    "$HERE/audit_selective_rescue.py"
  echo "Cross-Event Metric Memory V2 preflight passed"
}

run_d0() {
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_cross_event_metric_memory_v2.audit_d0 \
    --train-cache "$CACHE_ROOT/meta_train.pt" \
    --calibration-cache "$CACHE_ROOT/meta_calibration.pt" \
    --output "$D0_JSON" --topk "${MEMORY_TOPK:-4}" --seed "$SEED"
}

run_m1() {
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_cross_event_metric_memory_v2.train_m1 \
    --train-cache "$CACHE_ROOT/meta_train.pt" \
    --calibration-cache "$CACHE_ROOT/meta_calibration.pt" \
    --d0-json "$D0_JSON" --output-dir "$OUTPUT_ROOT" \
    --hidden "${METRIC_HIDDEN:-128}" --steps "${METRIC_STEPS:-1000}" \
    --batch-size "${BATCH_SIZE:-512}" --learning-rate "${LEARNING_RATE:-1e-3}" \
    --temperature "${CONTRAST_TEMPERATURE:-0.10}" \
    --retrieval-temperature "${RETRIEVAL_TEMPERATURE:-0.10}" \
    --capacity-per-class "${MEMORY_CAPACITY_PER_CLASS:-2048}" \
    --minimum-memory-purity "${MINIMUM_MEMORY_PURITY:-0.0}" \
    --topk "${MEMORY_TOPK:-4}" --chunk-size "${CHUNK_SIZE:-512}" \
    --device "${DEVICE:-cuda:0}" --seed "$SEED"
}

run_m2() {
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_cross_event_metric_memory_v2.audit_m2 \
    --train-cache "$CACHE_ROOT/meta_train.pt" \
    --calibration-cache "$CACHE_ROOT/meta_calibration.pt" \
    --m1-json "$M1_JSON" --output "$M2_JSON" \
    --topk "${MEMORY_TOPK:-4}" --retrieval-temperature "${RETRIEVAL_TEMPERATURE:-0.10}" \
    --minimum-memory-purity "${MINIMUM_MEMORY_PURITY:-0.0}" \
    --chunk-size "${CHUNK_SIZE:-512}" --device "${DEVICE:-cuda:0}"
}

run_rescue() {
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_cross_event_metric_memory_v2.audit_selective_rescue \
    --train-cache "$CACHE_ROOT/meta_train.pt" \
    --calibration-cache "$CACHE_ROOT/meta_calibration.pt" \
    --m1-json "$M1_JSON" --output "$RESCUE_JSON" \
    --topk "${MEMORY_TOPK:-4}" \
    --retrieval-temperature "${RETRIEVAL_TEMPERATURE:-0.10}" \
    --minimum-memory-purity "${MINIMUM_MEMORY_PURITY:-0.0}" \
    --chunk-size "${CHUNK_SIZE:-512}" --device "${DEVICE:-cuda:0}"
}

case "${1:-auto}" in
  preflight) preflight ;;
  d0) run_d0 ;;
  m1) run_m1 ;;
  m2) run_m2 ;;
  rescue) run_rescue ;;
  auto)
    preflight
    run_d0
    run_m1
    if "$PYTHON_BIN" -c "import json; assert json.load(open('$M1_JSON'))['decision']=='ADMIT_HIERARCHICAL_MEMORY_M2'"; then
      run_m2
    else
      echo "M1 did not pass; M2 was correctly not executed."
    fi
    ;;
  *) echo "usage: $0 {preflight|d0|m1|m2|rescue|auto}" >&2; exit 2 ;;
esac
