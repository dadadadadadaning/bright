#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
M=bda_benchmark.experiments.damageformer_roi_memory_balanced_v2
MODE="${1:-preflight}"

if [[ "$MODE" == preflight ]]; then
  "$PYTHON" -B -m unittest "$M.test_core" "$M.test_pipeline" -v
  for ENTRY in build_bank audit train; do
    "$PYTHON" -B -m "$M.$ENTRY" --help
  done
  echo 'CPU contracts and synthetic end-to-end smoke passed. Server data/CUDA smoke still required.'
  exit 0
fi

if [[ "$MODE" == report ]]; then
  EXTRA=()
  [[ -n "${PAIR:-}" ]] && EXTRA+=(--pair "$PAIR")
  "$PYTHON" -B -m "$M.report" --root "${OUTPUT_ROOT:?Set completed audit/train output}" \
    --bootstrap "${BOOTSTRAP:-0}" "${EXTRA[@]}"
  exit 0
fi

case "$MODE" in
  bank|smoke_bank|audit|smoke_audit|train|smoke_train) ;;
  *) echo 'Use preflight|bank|smoke_bank|audit|smoke_audit|train|smoke_train|report' >&2; exit 2 ;;
esac

test -s "${FRONT_CKPT:?Set immutable V2 front_snapshot.pth}"
test -s "${SOURCE_RUN:?Set completed ROI_EXCHANGE_V1 source run}/manifest.json"
: "${OUTPUT_ROOT:?Set a new output directory}"
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.lock" ]]; then
  echo "Output/lock exists; preserve it and choose a new OUTPUT_ROOT: $OUTPUT_ROOT" >&2
  exit 2
fi

COMMON=(--checkpoint "$FRONT_CKPT" --run_root "$SOURCE_RUN"
  --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
  --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}"
  --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}"
  --output "$OUTPUT_ROOT" --source_head hard --pool_mode hard --read_mode hard
  --device "${DEVICE:-cuda:0}" --workers "${WORKERS:-4}")
[[ "$MODE" == smoke_* ]] && COMMON+=(--smoke)

case "$MODE" in
  bank|smoke_bank)
    "$PYTHON" -B -u -m "$M.build_bank" "${COMMON[@]}" \
      --per_bucket "${PER_BUCKET:-64}" --seed "${SEED:-3407}"
    ;;
  audit|smoke_audit)
    test -s "${BANK_ROOT:?Set completed V2 dual bank}/bank_all.pt"
    test -s "$BANK_ROOT/bank_hard.pt"
    EXTRA=()
    [[ "${ALLOW_SAME_EVENT:-0}" == 1 ]] && EXTRA+=(--allow_same_event)
    "$PYTHON" -B -u -m "$M.audit" "${COMMON[@]}" --bank "$BANK_ROOT" \
      --topk "${TOPK:-8}" --temperature "${TEMPERATURE:-0.07}" \
      --seed "${SEED:-3407}" "${EXTRA[@]}"
    ;;
  train|smoke_train)
    test -s "${BANK_ROOT:?Set completed V2 dual bank}/bank_all.pt"
    test -s "$BANK_ROOT/bank_hard.pt"
    COMMON+=(--batch "${BATCH:-2}" --accum "${ACCUM:-8}" \
      --seed "${SEED:-3407}" --lr "${LR:-1e-4}")
    if [[ "$MODE" == smoke_train ]]; then
      COMMON+=(--steps 20 --eval_every 20)
    else
      COMMON+=(--steps "${STEPS:-5000}" --eval_every "${EVAL_EVERY:-500}")
    fi
    EXTRA=()
    [[ "${ALLOW_SAME_EVENT:-0}" == 1 ]] && EXTRA+=(--allow_same_event)
    "$PYTHON" -B -u -m "$M.train" "${COMMON[@]}" --bank "$BANK_ROOT" \
      --topk "${TOPK:-8}" --query_chunk "${QUERY_CHUNK:-256}" "${EXTRA[@]}"
    ;;
esac

echo "COMPLETE $MODE. Results: $OUTPUT_ROOT. No other stage launched. Train/Val only."
