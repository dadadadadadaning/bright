#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
M=bda_benchmark.experiments.damageformer_roi_partition_factorial_v1
MODE="${1:-preflight}"
if [[ "$MODE" == preflight ]]; then
  "$PYTHON" -m unittest "$M.test_core" "$M.test_diagnostics" "$M.test_report" "$M.test_training" -v
  "$PYTHON" -m "$M.train" --help
  exit 0
fi
case "$MODE" in smoke|train) ;; *) echo 'Use preflight|smoke|train'; exit 2 ;; esac
test -s "${FRONT_CKPT:?Set immutable V2 front snapshot, NOT A1 or readout weights}"
: "${OUTPUT_ROOT:?Set new non-existing output directory}"
read -r -a MODES <<< "${VARIANTS:-g0 g1 s0 s2}"
COMMON=(--checkpoint "$FRONT_CKPT" --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
 --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}"
 --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}"
 --output "$OUTPUT_ROOT" --variants "${MODES[@]}"
 --batch "${BATCH:-2}" --accum "${ACCUM:-8}" --workers "${WORKERS:-4}"
 --device "${DEVICE:-cuda:0}" --seed "${SEED:-3407}" --lr "${LR:-1e-4}"
 --threshold "${ROI_THRESHOLD:-0.5}")
if [[ "$MODE" == smoke ]]; then
  "$PYTHON" -u -m "$M.train" "${COMMON[@]}" --steps 20 --eval_every 20 --smoke
else
  "$PYTHON" -u -m "$M.train" "${COMMON[@]}" --steps "${STEPS:-5000}" --eval_every "${EVAL_EVERY:-500}"
fi
"$PYTHON" -m "$M.report" --root "$OUTPUT_ROOT"
