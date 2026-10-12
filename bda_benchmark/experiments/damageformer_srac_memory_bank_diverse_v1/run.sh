#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_memory_bank_diverse_v1

case "${1:-preflight}" in
  preflight)
    "$PYTHON" -m unittest "$MODULE.test_bank" "$MODULE.test_retrieve" -v
    "$PYTHON" -m "$MODULE.build" --help
    "$PYTHON" -m "$MODULE.evaluate" --help
    echo 'CPU synthetic checks only; no BRIGHT Val or Test image was read.' ;;
  build)
    : "${E0_PROTOCOL:?Set the passed legacy/native E0 protocol directory}"
    : "${BANK_DIV_OUTPUT:?Set a NEW Train bank output directory}"
    ARGS=(--protocol "$E0_PROTOCOL" --output "$BANK_DIV_OUTPUT")
    if [[ -n "${BANK_DIV_SEED:-}" ]]; then ARGS+=(--seed "$BANK_DIV_SEED"); fi
    if [[ -n "${BANK_DIV_PER_IMAGE_CAP:-}" ]]; then ARGS+=(--per_image_cap "$BANK_DIV_PER_IMAGE_CAP"); fi
    if [[ -n "${BANK_DIV_CANDIDATE_MULTIPLIER:-}" ]]; then ARGS+=(--candidate_multiplier "$BANK_DIV_CANDIDATE_MULTIPLIER"); fi
    if [[ -n "${BANK_DIV_QUALITY_WEIGHT:-}" ]]; then ARGS+=(--quality_weight "$BANK_DIV_QUALITY_WEIGHT"); fi
    "$PYTHON" -u -m "$MODULE.build" "${ARGS[@]}" ;;
  evaluate)
    : "${E0_PROTOCOL:?Set the passed legacy/native E0 protocol directory}"
    : "${BANK_DIV_OUTPUT:?Set the completed new Train bank directory}"
    : "${BANK_DIV_VAL_OUTPUT:?Set a NEW Val output directory}"
    ARGS=(--protocol "$E0_PROTOCOL" --bank "$BANK_DIV_OUTPUT" --output "$BANK_DIV_VAL_OUTPUT")
    if [[ -n "${BANK_DIV_LIMIT:-}" ]]; then ARGS+=(--limit "$BANK_DIV_LIMIT"); fi
    "$PYTHON" -u -m "$MODULE.evaluate" "${ARGS[@]}" ;;
  report)
    : "${BANK_DIV_VAL_OUTPUT:?Set a completed formal Val450 output directory}"
    "$PYTHON" -m "$MODULE.report" --output "$BANK_DIV_VAL_OUTPUT" ;;
  *) echo 'Use preflight|build|evaluate|report' >&2; exit 2 ;;
esac
