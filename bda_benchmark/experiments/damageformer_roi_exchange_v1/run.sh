#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
M=bda_benchmark.experiments.damageformer_roi_exchange_v1
MODE="${1:-preflight}"
if [[ "$MODE" == preflight ]]; then
  "$PYTHON" -m unittest "$M.test_core" "$M.test_diagnostics" "$M.test_audit" "$M.test_training" "$M.test_report" -v
  "$PYTHON" -m "$M.train" --help
  "$PYTHON" -m "$M.audit" --help
  exit 0
fi
if [[ "$MODE" == inspect ]]; then
  "$PYTHON" -m "$M.inspect_results" --root "${RUN_ROOT:?Set completed factorial run directory}" --pairs "${PAIR:-s0__s2}"
  exit 0
fi
if [[ "$MODE" == report ]]; then
  "$PYTHON" -m "$M.report" --root "${OUTPUT_ROOT:?Set existing results directory}"
  exit 0
fi
case "$MODE" in smoke|train|smoke_audit|audit) ;; *) echo 'Use preflight|inspect|report|smoke_audit|audit|smoke|train'; exit 2 ;; esac
test -s "${FRONT_CKPT:?Set immutable V2 front snapshot, NOT original A1 or readout weights}"
: "${OUTPUT_ROOT:?Set a NEW non-existing output directory}"
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Output exists; keep it and choose a new OUTPUT_ROOT: $OUTPUT_ROOT" >&2
  exit 2
fi
COMMON=(--checkpoint "$FRONT_CKPT" --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
 --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}"
 --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}"
 --output "$OUTPUT_ROOT" --workers "${WORKERS:-4}" --device "${DEVICE:-cuda:0}")
if [[ "$MODE" == audit || "$MODE" == smoke_audit ]]; then
  : "${RUN_ROOT:?Set completed source training directory with matching last.pth readouts}"
  read -r -a NAMES_ARRAY <<< "${NAMES:-s0 s2}"
  EXTRA=()
  if [[ "$MODE" == smoke_audit ]]; then EXTRA+=(--smoke); fi
  "$PYTHON" -u -m "$M.audit" "${COMMON[@]}" --run_root "$RUN_ROOT" --names "${NAMES_ARRAY[@]}" "${EXTRA[@]}"
else
  read -r -a MODES <<< "${VARIANTS:-hard exchange}"
  COMMON+=(--variants "${MODES[@]}" --batch "${BATCH:-2}" --accum "${ACCUM:-8}"
   --seed "${SEED:-3407}" --lr "${LR:-1e-4}"
   --epsilon "${EPSILON:-0.05}" --threshold "${ROI_THRESHOLD:-0.5}")
  if [[ "$MODE" == smoke ]]; then
    "$PYTHON" -u -m "$M.train" "${COMMON[@]}" --steps 20 --eval_every 20 --smoke
  else
    "$PYTHON" -u -m "$M.train" "${COMMON[@]}" --steps "${STEPS:-5000}" --eval_every "${EVAL_EVERY:-500}"
  fi
fi
"$PYTHON" -m "$M.report" --root "$OUTPUT_ROOT"
