#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_a1_state_probe_v1
MODE="${1:-preflight}"

case "$MODE" in
  preflight)
    "$PYTHON" -m unittest "$MODULE.test_probe" -v
    "$PYTHON" -m "$MODULE.probe" --help
    echo 'CPU synthetic checks only; no BRIGHT Train, Val or Test images were read.'
    ;;
  smoke|formal_replay)
    : "${E0_PROTOCOL:?Set a passed legacy/native E0 protocol directory}"
    : "${A1_PROBE_OUTPUT:?Set a fresh diagnostic output directory}"
    args=(--protocol "$E0_PROTOCOL" --output "$A1_PROBE_OUTPUT"
          --arm "${A1_PROBE_ARM:-building_reconstruction}"
          --full-every "${A1_PROBE_FULL_EVERY:-20}")
    if [[ "$MODE" == formal_replay ]]; then
      args+=(--formal-replay)
    fi
    "$PYTHON" -u -m "$MODULE.probe" "${args[@]}"
    ;;
  *)
    echo 'Use preflight|smoke|formal_replay' >&2
    exit 2
    ;;
esac
