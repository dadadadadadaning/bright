#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_memory_bank_v2

case "${1:-preflight}" in
  preflight)
    "$PYTHON" -m unittest "$MODULE.test_bank" "$MODULE.test_retrieve" -v
    "$PYTHON" -m "$MODULE.build" --help
    "$PYTHON" -m "$MODULE.evaluate" --help
    echo 'CPU synthetic checks only; no BRIGHT Val or Test image was read.' ;;
  build)
    : "${E0_PROTOCOL:?Set the passed legacy/native E0 protocol directory}"
    : "${BANK_V2_OUTPUT:?Set a NEW Train bank output directory}"
    ARGS=(--protocol "$E0_PROTOCOL" --output "$BANK_V2_OUTPUT")
    if [[ -n "${BANK_V2_SEED:-}" ]]; then ARGS+=(--seed "$BANK_V2_SEED"); fi
    if [[ -n "${BANK_V2_PER_IMAGE_CAP:-}" ]]; then ARGS+=(--per_image_cap "$BANK_V2_PER_IMAGE_CAP"); fi
    if [[ -n "${BANK_V2_CANDIDATE_MULTIPLIER:-}" ]]; then ARGS+=(--candidate_multiplier "$BANK_V2_CANDIDATE_MULTIPLIER"); fi
    "$PYTHON" -u -m "$MODULE.build" "${ARGS[@]}" ;;
  evaluate)
    : "${E0_PROTOCOL:?Set the passed legacy/native E0 protocol directory}"
    : "${BANK_V2_OUTPUT:?Set the completed new Train bank directory}"
    : "${BANK_V2_VAL_OUTPUT:?Set a NEW Val output directory}"
    ARGS=(--protocol "$E0_PROTOCOL" --bank "$BANK_V2_OUTPUT" --output "$BANK_V2_VAL_OUTPUT")
    if [[ -n "${BANK_V2_LIMIT:-}" ]]; then ARGS+=(--limit "$BANK_V2_LIMIT"); fi
    "$PYTHON" -u -m "$MODULE.evaluate" "${ARGS[@]}" ;;
  report)
    : "${BANK_V2_VAL_OUTPUT:?Set a completed formal Val450 output directory}"
    "$PYTHON" -m "$MODULE.report" --output "$BANK_V2_VAL_OUTPUT" ;;
  *) echo 'Use preflight|build|evaluate|report' >&2; exit 2 ;;
esac
