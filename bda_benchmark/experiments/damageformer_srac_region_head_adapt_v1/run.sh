#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_region_head_adapt_v1

case "${1:-preflight}" in
  preflight)
    "$PYTHON" -m unittest discover -s bda_benchmark/experiments/damageformer_srac_region_head_adapt_v1 -t . -p 'test_*.py' -v
    "$PYTHON" -m "$MODULE.train" --help
    "$PYTHON" -m "$MODULE.evaluate" --help
    "$PYTHON" -m "$MODULE.analyze" --help
    "$PYTHON" -m "$MODULE.recheck_probability" --help
    echo 'CPU synthetic tests only; no BRIGHT Train/Val/Test image was read.' ;;
  locate)
    : "${E0_PROTOCOL:?Set the passed legacy/native E0 protocol directory}"
    ARGS=(--protocol "$E0_PROTOCOL")
    if [[ -n "${REGION_SEARCH_ROOT:-}" ]]; then ARGS+=(--root "$REGION_SEARCH_ROOT"); fi
    "$PYTHON" -m "$MODULE.locate" "${ARGS[@]}" ;;
  analyze)
    : "${REGION_ANALYSIS_OUTPUT:?Set a NEW offline analysis output directory}"
    if [[ -z "${REGION_VAL_OUTPUT:-}" ]]; then
      : "${E0_PROTOCOL:?Set E0_PROTOCOL to locate the completed old Val report}"
      ARGS=(--protocol "$E0_PROTOCOL")
      if [[ -n "${REGION_SEARCH_ROOT:-}" ]]; then ARGS+=(--root "$REGION_SEARCH_ROOT"); fi
      REGION_VAL_OUTPUT="$("$PYTHON" -m "$MODULE.locate" "${ARGS[@]}")"
    fi
    "$PYTHON" -u -m "$MODULE.analyze" --input "$REGION_VAL_OUTPUT" --output "$REGION_ANALYSIS_OUTPUT" ;;
  smoke)
    : "${E0_PROTOCOL:?Set the passed legacy/native E0 protocol directory}"
    : "${REGION_ADAPT_SMOKE_OUTPUT:?Set a NEW smoke fit directory}"
    "$PYTHON" -u -m "$MODULE.train" --protocol "$E0_PROTOCOL" --output "$REGION_ADAPT_SMOKE_OUTPUT" --smoke --steps 20 ;;
  train)
    : "${E0_PROTOCOL:?Set the passed legacy/native E0 protocol directory}"
    : "${REGION_ADAPT_FIT:?Set a NEW formal fitting output directory}"
    "$PYTHON" -u -m "$MODULE.train" --protocol "$E0_PROTOCOL" --output "$REGION_ADAPT_FIT" --steps 5000 ;;
  probe)
    : "${E0_PROTOCOL:?Set the passed legacy/native E0 protocol directory}"
    : "${REGION_ADAPT_PROBE:?Set a NEW optional embedding-probe output directory}"
    "$PYTHON" -u -m "$MODULE.evaluate" --protocol "$E0_PROTOCOL" --output "$REGION_ADAPT_PROBE" --interventions ;;
  evaluate)
    : "${E0_PROTOCOL:?Set the passed legacy/native E0 protocol directory}"
    : "${REGION_ADAPT_FIT:?Set the completed formal fitting directory}"
    : "${REGION_ADAPT_EVAL:?Set a NEW standalone Val output directory}"
    "$PYTHON" -u -m "$MODULE.evaluate" --protocol "$E0_PROTOCOL" --fit "$REGION_ADAPT_FIT" --output "$REGION_ADAPT_EVAL" ;;
  report)
    : "${REGION_ADAPT_EVAL:?Set a completed formal evaluation directory (fit/final_val is valid)}"
    "$PYTHON" -m "$MODULE.report" --output "$REGION_ADAPT_EVAL" ;;
  recheck_probability)
    : "${E0_PROTOCOL:?Set the passed legacy/native E0 protocol directory}"
    : "${REGION_ADAPT_FAILED_EVAL:?Set the preserved failed fit/initial_val directory}"
    : "${REGION_ADAPT_DIAG_OUTPUT:?Set a NEW diagnostic output outside the failed fit directory}"
    "$PYTHON" -u -m "$MODULE.recheck_probability" --protocol "$E0_PROTOCOL" --failed "$REGION_ADAPT_FAILED_EVAL" --output "$REGION_ADAPT_DIAG_OUTPUT" ;;
  *) echo 'Use preflight|locate|analyze|smoke|train|probe|evaluate|report|recheck_probability' >&2; exit 2 ;;
esac
