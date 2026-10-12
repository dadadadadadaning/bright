#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
M=bda_benchmark.experiments.damageformer_roi_followup_v1
MODE="${1:-preflight}"
if [[ "$MODE" == preflight ]]; then
  "$PYTHON" -B -m unittest "$M.test_routing" "$M.test_memory" "$M.test_q" "$M.test_report" "$M.test_audit" -v
  for ENTRY in audit_routing build_bank train_memory train_q; do
    "$PYTHON" -B -m "$M.$ENTRY" --help
  done
  echo 'CPU contracts passed. Server data/CUDA smoke still required. Nothing trained.'
  exit 0
fi
if [[ "$MODE" == report ]]; then
  EXTRA=()
  if [[ -n "${PAIR:-}" ]]; then EXTRA+=(--pair "$PAIR"); fi
  "$PYTHON" -B -m "$M.report" --root "${OUTPUT_ROOT:?Set existing completed results}" --bootstrap "${BOOTSTRAP:-0}" "${EXTRA[@]}"
  exit 0
fi
case "$MODE" in audit|smoke_audit|bank|smoke_bank|memory|smoke_memory|q_train|smoke_q) ;; *) echo 'Use preflight|audit|smoke_audit|bank|smoke_bank|memory|smoke_memory|q_train|smoke_q|report' >&2; exit 2 ;; esac
test -s "${FRONT_CKPT:?Set immutable V2 front_snapshot.pth, NOT A1 or a readout}"
test -s "${SOURCE_RUN:?Set completed ROI_EXCHANGE_V1 directory}/manifest.json"
: "${OUTPUT_ROOT:?Set a NEW non-existing output directory}"
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "Output exists; preserve it and choose a new OUTPUT_ROOT: $OUTPUT_ROOT" >&2
  exit 2
fi
COMMON=(--checkpoint "$FRONT_CKPT" --run_root "$SOURCE_RUN"
  --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
  --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}"
  --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}"
  --output "$OUTPUT_ROOT" --source_head "${SOURCE_HEAD:-hard}"
  --pool_mode "${POOL_MODE:-hard}" --read_mode "${READ_MODE:-hard}"
  --device "${DEVICE:-cuda:0}" --workers "${WORKERS:-4}")
if [[ "$MODE" == smoke_* ]]; then COMMON+=(--smoke); fi
case "$MODE" in
  audit|smoke_audit)
    read -r -a HEADS <<< "${AUDIT_HEADS:-hard exchange}"
    read -r -a COMBOS <<< "${COMBINATIONS:-HH EH HE EE}"
    "$PYTHON" -B -u -m "$M.audit_routing" "${COMMON[@]}" --heads "${HEADS[@]}" --combinations "${COMBOS[@]}"
    ;;
  bank|smoke_bank)
    "$PYTHON" -B -u -m "$M.build_bank" "${COMMON[@]}" --per_bucket "${PER_BUCKET:-64}" --seed "${SEED:-3407}"
    ;;
  memory|smoke_memory|q_train|smoke_q)
    COMMON+=(--batch "${BATCH:-2}" --accum "${ACCUM:-8}" --seed "${SEED:-3407}" --lr "${LR:-1e-4}")
    if [[ "$MODE" == smoke_* ]]; then
      COMMON+=(--steps 20 --eval_every 20)
    else
      COMMON+=(--steps "${STEPS:-5000}" --eval_every "${EVAL_EVERY:-500}")
    fi
    if [[ "$MODE" == memory || "$MODE" == smoke_memory ]]; then
      test -s "${BANK_ROOT:?Set matching complete bank directory}/bank.pt"
      EXTRA=()
      if [[ "${ALLOW_SAME_EVENT:-0}" == 1 ]]; then EXTRA+=(--allow_same_event); fi
      "$PYTHON" -B -u -m "$M.train_memory" "${COMMON[@]}" --bank "$BANK_ROOT" --topk "${TOPK:-8}" --query_chunk "${QUERY_CHUNK:-256}" "${EXTRA[@]}"
    else
      "$PYTHON" -B -u -m "$M.train_q" "${COMMON[@]}"
    fi
    ;;
esac
echo "COMPLETE $MODE. Results: $OUTPUT_ROOT. No other stage launched. Train/Val only."
