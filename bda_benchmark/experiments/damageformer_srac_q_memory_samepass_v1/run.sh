#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_q_memory_samepass_v1
case "${1:-preflight}" in
  preflight)
    "$PYTHON" -m unittest "$MODULE.test_gates" bda_benchmark.experiments.damageformer_srac_q_memory_diagnostic_v1.test_stats -v
    "$PYTHON" -m "$MODULE.run" --help
    echo 'CPU synthetic tests only; no BRIGHT Val or Test images were read.' ;;
  diagnose)
    : "${E0_PROTOCOL:?Set the passed legacy/native E0 protocol directory}"
    : "${Q_SAMEPASS_OUTPUT:?Set a NEW output directory}"
    ARGS=(--protocol "$E0_PROTOCOL" --output "$Q_SAMEPASS_OUTPUT")
    if [[ -n "${E1_EVIDENCE:-}" ]]; then ARGS+=(--evidence "$E1_EVIDENCE"); fi
    if [[ -n "${Q_SAMEPASS_LIMIT:-}" ]]; then ARGS+=(--limit "$Q_SAMEPASS_LIMIT"); fi
    "$PYTHON" -u -m "$MODULE.run" "${ARGS[@]}" ;;
  *) echo 'Use preflight|diagnose' >&2; exit 2 ;;
esac
