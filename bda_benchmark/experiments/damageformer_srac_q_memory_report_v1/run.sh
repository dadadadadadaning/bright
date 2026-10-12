#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
if [[ "${1:-}" == preflight ]]; then
  "$PYTHON" -m unittest -v bda_benchmark.experiments.damageformer_srac_q_memory_report_v1.test_report
elif [[ "${1:-}" == report ]]; then
  : "${Q_SAMEPASS_OUTPUT:?Set the completed Q x Memory same-pass run directory}"
  "$PYTHON" -m bda_benchmark.experiments.damageformer_srac_q_memory_report_v1.report \
    --run "$Q_SAMEPASS_OUTPUT" --json-output "$Q_SAMEPASS_OUTPUT/q_memory_report.json"
else
  echo "Usage: bash $0 {preflight|report}" >&2
  exit 2
fi
