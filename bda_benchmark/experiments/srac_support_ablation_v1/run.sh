#!/usr/bin/env bash
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-preflight}"
if [[ "$MODE" == preflight ]]; then
  "$PYTHON_BIN" -m unittest bda_benchmark.experiments.srac_support_ablation_v1.test_core -v
  exit 0
fi
if [[ "$MODE" != all ]]; then echo 'Usage: run.sh preflight|all'; exit 2; fi
: "${TRAIN_RECORDS:?Set TRAIN_RECORDS to an existing sampled-token Train cache}"
: "${VAL_RECORDS:?Set VAL_RECORDS to its matching validation cache}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a NEW isolated experiment directory}"
"$PYTHON_BIN" -u -m bda_benchmark.experiments.srac_support_ablation_v1.run \
  --train "$TRAIN_RECORDS" --val "$VAL_RECORDS" --output "$OUTPUT_ROOT" \
  --sources "${SOURCES:-base_change,explicit_diff}" --steps "${STEPS:-5000}" \
  --batch_size "${BATCH_SIZE:-128}" --dim "${DIM:-128}" --lr "${LR:-0.0001}" \
  --per_bucket "${PER_BUCKET:-64}" --topk "${TOPK:-8}" --repeats "${REPEATS:-5}" \
  --eval_every "${EVAL_EVERY:-500}" --seed "${SEED:-3407}" --device "${DEVICE:-cuda:0}"
