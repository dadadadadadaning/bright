#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
case "${1:-preflight}" in
  preflight)
    "$PYTHON" -m unittest bda_benchmark.experiments.damageformer_srac_memory_interventions_samepass_v1.test_contracts -v
    "$PYTHON" -m bda_benchmark.experiments.damageformer_srac_memory_interventions_samepass_v1.launch --help
    echo 'CPU synthetic tests only; no BRIGHT Val or Test images were read.' ;;
  diagnose)
    : "${REFERENCE_ROOT:?Set the passed legacy/native E0/E1 root}"
    : "${OUTPUT:?Set a fresh output directory}"
    ARGS=(--reference_root "$REFERENCE_ROOT" --output "$OUTPUT")
    if [[ -n "${RANDOM_SEEDS:-}" ]]; then
      read -r -a SEEDS <<< "$RANDOM_SEEDS"
      ARGS+=(--random_seeds "${SEEDS[@]}")
    fi
    if [[ -n "${SHUFFLE_SEED:-}" ]]; then ARGS+=(--shuffle_seed "$SHUFFLE_SEED"); fi
    "$PYTHON" -u -m bda_benchmark.experiments.damageformer_srac_memory_interventions_samepass_v1.launch "${ARGS[@]}" ;;
  report)
    : "${OUTPUT:?Set the completed same-pass output directory}"
    "$PYTHON" -m bda_benchmark.experiments.damageformer_srac_memory_interventions_samepass_v1.report --output "$OUTPUT" ;;
  *) echo 'Use preflight|diagnose|report' >&2; exit 2 ;;
esac
