#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
M=bda_benchmark.experiments.damageformer_roi_memory_selective_test_v1.evaluate
MODE="${1:-preflight}"

case "$MODE" in
  preflight)
    "$PYTHON" -B -m unittest \
      bda_benchmark.experiments.damageformer_roi_memory_selective_test_v1.test_evaluate -v
    "$PYTHON" -B -m "$M" --help
    ;;
  test)
    [[ "${ALLOW_TEST_ACCESS:-}" == YES ]] || {
      echo 'Set ALLOW_TEST_ACCESS=YES for one-time Test access' >&2
      exit 2
    }
    : "${POSTERIOR_RUN:?Set completed posterior Train/Val run}"
    : "${OUTPUT_ROOT:?Set fresh Test output directory}"
    if [[ -e "$OUTPUT_ROOT" ]]; then
      echo "Output exists: $OUTPUT_ROOT" >&2
      exit 2
    fi
    "$PYTHON" -B -u -m "$M" \
      --run_root "$POSTERIOR_RUN" --output "$OUTPUT_ROOT" \
      --device "${DEVICE:-cuda:0}" --workers "${WORKERS:-4}"
    ;;
  *)
    echo 'Use preflight|test' >&2
    exit 2
    ;;
esac
