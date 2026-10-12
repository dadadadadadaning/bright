#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BDA_ROOT="$(cd "$HERE/../.." && pwd)"
REPO_ROOT="$(cd "$BDA_ROOT/.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$BDA_ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
SEED="${SEED:-7341}"
CACHE_ROOT="${D1_CACHE_ROOT:-/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed7341}"
RESULT_ROOT="${RESULT_ROOT:-/root/data/BRIGHT-eval-results/SPI_HIERARCHICAL_BUILDING_RETRIEVAL_V1}"
OUTPUT="$RESULT_ROOT/hbr_r0_seed${SEED}.json"
DIRECTION_OUTPUT="$RESULT_ROOT/hbr_direction_d0_seed${SEED}.json"

preflight() {
  "$PYTHON_BIN" -m pytest -q "$HERE/test_core.py"
  "$PYTHON_BIN" -m py_compile "$HERE/core.py" "$HERE/audit_r0.py" "$HERE/audit_direction.py"
  echo "Hierarchical Building Retrieval V1 preflight passed"
}

direction() {
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_hierarchical_building_retrieval_v1.audit_direction \
    --train-cache "$CACHE_ROOT/meta_train.pt" \
    --calibration-cache "$CACHE_ROOT/meta_calibration.pt" \
    --r0-json "$OUTPUT" --output "$DIRECTION_OUTPUT" \
    --coarse-buildings "${COARSE_BUILDINGS:-32}" \
    --topk "${MEMORY_TOPK:-8}" \
    --maximum-per-event "${MAXIMUM_PER_EVENT:-2}" \
    --minimum-positive "${MINIMUM_POSITIVE:-100}" \
    --minimum-negative "${MINIMUM_NEGATIVE:-100}" \
    --minimum-evaluable-events "${MINIMUM_EVALUABLE_EVENTS:-5}" \
    --seed "$SEED"
}

audit() {
  "$PYTHON_BIN" -u -m bda_benchmark.experiments.spi_hierarchical_building_retrieval_v1.audit_r0 \
    --train-cache "$CACHE_ROOT/meta_train.pt" \
    --calibration-cache "$CACHE_ROOT/meta_calibration.pt" \
    --output "$OUTPUT" \
    --coarse-buildings "${COARSE_BUILDINGS:-32}" \
    --topk "${MEMORY_TOPK:-8}" \
    --maximum-per-event "${MAXIMUM_PER_EVENT:-2}" \
    --seed "$SEED"
}

case "${1:-audit}" in
  preflight) preflight ;;
  audit) audit ;;
  direction) direction ;;
  auto) preflight; audit ;;
  *) echo "usage: $0 {preflight|audit|direction|auto}" >&2; exit 2 ;;
esac
