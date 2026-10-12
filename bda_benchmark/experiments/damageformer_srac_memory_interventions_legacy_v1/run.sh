#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
: "${REFERENCE_ROOT:?Set the completed legacy/native E0/E1 root}"
: "${OUTPUT:?Set a new, absent output directory}"
"$PYTHON" -m unittest bda_benchmark.experiments.damageformer_srac_memory_interventions_legacy_v1.test_evaluate -v
"$PYTHON" -u -m bda_benchmark.experiments.damageformer_srac_memory_interventions_legacy_v1.launch \
  --reference_root "$REFERENCE_ROOT" \
  --intervention_seed "${INTERVENTION_SEED:-3407}" --output "$OUTPUT"
