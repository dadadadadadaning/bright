#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
M=bda_benchmark.experiments.damageformer_memory_label_v1
MODE="${1:-preflight}"
if [[ "$MODE" == preflight ]]; then
  "$PYTHON" -B -m unittest "$M.test_core" "$M.test_report" "$M.test_pipeline" -v
  "$PYTHON" -B -m "$M.experiment" --help
  "$PYTHON" -B -m "$M.report" --help
  echo 'LOCAL CONTRACTS PASSED. Real CUDA/data smoke still required. No formal training started.'
  exit 0
fi
if [[ "$MODE" == report ]]; then
  "$PYTHON" -B -m "$M.report" --root "${REPORT_ROOT:?Set REPORT_ROOT to the EXISTING completed run, not a new timestamp}"
  exit 0
fi
case "$MODE" in audit|train|smoke_audit|smoke_train) ;; *)
  echo 'Use preflight|smoke_audit|audit|smoke_train|train|report' >&2; exit 2 ;; esac
# Never inherit an old experiment's VARIANTS, MEMORY_ROOT, POOL_MODE or EXP.
: "${FRONT_CKPT:?Set immutable front_snapshot.pth, NOT original A1}"
: "${SOURCE_RUN:?Set completed ROI_EXCHANGE_V1 run containing hard_last.pth}"
: "${BANK_ROOT:?Set EXISTING REGION_MEMORY_V2 bank directory containing bank.pt and manifest.json}"
: "${OUTPUT_ROOT:?Set a NEW isolated LABEL_MEMORY_V1 output directory}"
for FILE in "$FRONT_CKPT" "$SOURCE_RUN/hard_last.pth" "$SOURCE_RUN/manifest.json" "$SOURCE_RUN/summary.json" "$BANK_ROOT/bank.pt" "$BANK_ROOT/manifest.json"; do
  if [[ ! -s "$FILE" ]]; then echo "Missing required file: $FILE" >&2; exit 2; fi
done
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Preserve existing output; choose a new OUTPUT_ROOT: $OUTPUT_ROOT" >&2; exit 2
fi
STEPS_VALUE="${STEPS:-1000}"
EVAL_VALUE="${EVAL_EVERY:-250}"
SMOKE=()
if [[ "$MODE" == smoke_* ]]; then SMOKE=(--smoke); STEPS_VALUE=20; EVAL_VALUE=20; fi
ACTION="${MODE#smoke_}"
echo "Action=$ACTION output=$OUTPUT_ROOT bank=$BANK_ROOT"
"$PYTHON" -B -u -m "$M.experiment" --mode "$ACTION" \
  --checkpoint "$FRONT_CKPT" --run_root "$SOURCE_RUN" --bank "$BANK_ROOT" --output "$OUTPUT_ROOT" \
  --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}" \
  --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}" \
  --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}" \
  --source_head hard --pool_mode hard --read_mode hard --device "${DEVICE:-cuda:0}" --workers "${WORKERS:-4}" \
  --steps "$STEPS_VALUE" --eval_every "$EVAL_VALUE" --batch "${BATCH:-2}" --accum "${ACCUM:-8}" \
  --seed "${SEED:-3407}" --lr "${LABEL_LR:-0.01}" --topk "${LABEL_TOPK:-8}" \
  --query_chunk "${QUERY_CHUNK:-256}" --temperature "${LABEL_TEMPERATURE:-0.1}" \
  --mix_weight "${LABEL_MIX_WEIGHT:-0.1}" "${SMOKE[@]}"
echo 'COMPLETE. Read the evidence report; successful execution does not imply Memory benefit.'
