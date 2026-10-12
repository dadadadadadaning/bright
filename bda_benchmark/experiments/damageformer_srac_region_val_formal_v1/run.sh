#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_region_val_formal_v1

case "${1:-preflight}" in
  preflight)
    "$PYTHON" -m unittest "$MODULE.test_core" -v
    "$PYTHON" -m unittest bda_benchmark.experiments.damageformer_srac_region_test_v1.test_core -v
    "$PYTHON" -m "$MODULE.run" --help
    echo 'CPU contracts only; no BRIGHT Val or Test images were read.' ;;
  validate)
    : "${E0_PROTOCOL:?Set the passed legacy/native Val450 E0 protocol directory}"
    : "${REGION_VAL_OUTPUT:?Set a NEW output directory}"
    ARGS=(--protocol "$E0_PROTOCOL" --output "$REGION_VAL_OUTPUT")
    if [[ -n "${REGION_VAL_LIMIT:-}" ]]; then ARGS+=(--limit "$REGION_VAL_LIMIT"); fi
    "$PYTHON" -u -m "$MODULE.run" "${ARGS[@]}" ;;
  report)
    : "${REGION_VAL_OUTPUT:?Set the completed formal Val450 output directory}"
    "$PYTHON" -m "$MODULE.report" --output "$REGION_VAL_OUTPUT" ;;
  *) echo 'Use preflight|validate|report' >&2; exit 2 ;;
esac
