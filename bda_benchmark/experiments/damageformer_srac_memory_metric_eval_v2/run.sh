#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_memory_metric_eval_v2

case "${1:-preflight}" in
  preflight)
    "$PYTHON" -m unittest "$MODULE.test_eval" -v
    "$PYTHON" -m "$MODULE.evaluate" --help
    "$PYTHON" -m "$MODULE.report" --help
    echo 'Help/preflight only; no BRIGHT Val or Test image was read.' ;;
  evaluate)
    : "${E0_PROTOCOL:?Set the passed legacy/native E0 protocol directory}"
    : "${METRIC_V2_OUTPUT:?Set the completed Train-only metric v2 directory}"
    : "${METRIC_V2_VAL_OUTPUT:?Set a NEW Val output directory}"
    ARGS=(--protocol "$E0_PROTOCOL" --metrics "$METRIC_V2_OUTPUT"
          --output "$METRIC_V2_VAL_OUTPUT")
    if [[ -n "${METRIC_V2_VAL_LIMIT:-}" ]]; then
      ARGS+=(--limit "$METRIC_V2_VAL_LIMIT")
    fi
    "$PYTHON" -u -m "$MODULE.evaluate" "${ARGS[@]}" ;;
  report)
    : "${METRIC_V2_VAL_OUTPUT:?Set a completed formal Val450 output directory}"
    "$PYTHON" -m "$MODULE.report" --output "$METRIC_V2_VAL_OUTPUT" ;;
  *) echo 'Use preflight|evaluate|report' >&2; exit 2 ;;
esac
