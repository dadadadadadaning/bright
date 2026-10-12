#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_region_test_v1
MODE="${1:-preflight}"

if [[ "$MODE" == preflight ]]; then
  "$PYTHON" -m unittest discover -s "${MODULE//.//}" -t "$ROOT" -p 'test_*.py' -v
  "$PYTHON" -m "$MODULE.evaluate" --help
  exit 0
fi
if [[ "$MODE" != full && "$MODE" != prefix ]]; then
  echo 'Use preflight|full|prefix' >&2
  exit 2
fi

: "${JOINT_RUN:?Set matching completed J1/J2/J3 source directory}"
: "${A1_CKPT:?Set the matching A1 checkpoint}"
: "${PROTOCOL_RUN:?Set an already-passed full E0 protocol directory}"
: "${OUTPUT:?Set a NEW output directory}"
COMMON=(
  --joint_run "$JOINT_RUN" --a1_checkpoint "$A1_CKPT"
  --protocol "$PROTOCOL_RUN"
  --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
  --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}"
  --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}"
  --test_list "${TEST_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/test_set.txt}"
  --output "$OUTPUT" --device "${DEVICE:-cuda:0}" --workers "${WORKERS:-4}"
  --seed "${SEED:-3407}" --profile legacy --pool_mode native
)
if [[ -n "${FRONT_CKPT:-}" ]]; then COMMON+=(--checkpoint "$FRONT_CKPT"); fi
if [[ -n "${CACHE:-}" ]]; then COMMON+=(--cache "$CACHE"); fi
if [[ -n "${OLD_METRIC_CKPT:-}" ]]; then COMMON+=(--metric_checkpoint "$OLD_METRIC_CKPT"); fi
if [[ "$MODE" == prefix ]]; then COMMON+=(--max_images "${MAX_IMAGES:-2}"); fi

"$PYTHON" -u -m "$MODULE.evaluate" "${COMMON[@]}"
