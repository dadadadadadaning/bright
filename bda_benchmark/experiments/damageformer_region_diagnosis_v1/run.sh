#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
M=bda_benchmark.experiments.damageformer_region_diagnosis_v1
MODE="${1:-preflight}"
if [[ "$MODE" == preflight ]]; then
 python -m unittest "$M.test_diagnostics" -v
 python -m "$M.audit" --help
 python -m "$M.train_control" --help
 exit 0
fi
: "${OUTPUT_ROOT:?Set a NEW output directory}"
COMMON=(--dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
 --train_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt"
 --val_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt"
 --output "$OUTPUT_ROOT" --device "${DEVICE:-cuda:0}" --workers "${WORKERS:-4}")
case "$MODE" in
 audit|smoke_audit)
  test -s "${FRONT_CKPT:?Set a fixed V2 front checkpoint}"
  EXTRA=(); if [[ "$MODE" == smoke_audit ]]; then EXTRA=(--smoke); fi
  python -u -m "$M.audit" "${COMMON[@]}" --checkpoint "$FRONT_CKPT" \
   --calibration_batches "${CALIBRATION_BATCHES:-128}" --passes "${CALIBRATION_PASSES:-2}" "${EXTRA[@]}"
  ;;
 train|smoke_train)
  test -s "${A1_CKPT:?Set original audited A1 checkpoint, NOT a V2 checkpoint}"
  EXTRA=(); STEPCOUNT="${CONTROL_STEPS:-5000}"; EVAL="${EVAL_EVERY:-1000}"
  if [[ "$MODE" == smoke_train ]]; then EXTRA=(--smoke); STEPCOUNT=20; EVAL=20; fi
  python -u -m "$M.train_control" "${COMMON[@]}" --a1 "$A1_CKPT" --mode "${2:?Use a1, joint, or joint_frozen_bn}" \
   --steps "$STEPCOUNT" --eval_every "$EVAL" --batch "${BATCH:-2}" --accum "${ACCUM:-8}" \
   --base_lr "${BASE_LR:-1e-5}" --new_lr "${NEW_LR:-1e-4}" --amp "${AMP:-bf16}" "${EXTRA[@]}"
  ;;
 *) echo 'Use preflight|smoke_audit|audit|smoke_train MODE|train MODE'; exit 2;;
esac
