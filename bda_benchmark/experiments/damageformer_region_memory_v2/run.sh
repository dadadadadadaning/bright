#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
M=bda_benchmark.experiments.damageformer_region_memory_v2
MODE="${1:-preflight}"

if [[ "$MODE" == preflight ]]; then
  "$PYTHON" -B -m unittest "$M.test_core" "$M.test_integration" "$M.test_diagnostics" "$M.test_pipeline" -v
  for ENTRY in build_bank train report; do "$PYTHON" -B -m "$M.$ENTRY" --help; done
  echo 'LOCAL CONTRACTS PASSED. Server CUDA/data smoke still required. No real training started.'
  exit 0
fi
if [[ "$MODE" == report ]]; then
  "$PYTHON" -B -m "$M.report" --root "${OUTPUT_ROOT:-${MEMORY_ROOT:?Set MEMORY_ROOT}/train}" --bootstrap "${BOOTSTRAP:-0}"
  exit 0
fi
case "$MODE" in bank|smoke_bank|train|smoke_train|all|smoke_all) ;; *)
  echo 'Use preflight|bank|train|all|smoke_all|smoke_bank|smoke_train|report' >&2; exit 2 ;; esac
test -s "${FRONT_CKPT:?Set immutable front_snapshot.pth, NOT original A1}"
test -s "${SOURCE_RUN:?Set completed ROI_EXCHANGE_V1 run}/manifest.json"
test -s "$SOURCE_RUN/hard_last.pth"
# This experiment deliberately does not inherit old VARIANTS/POOL_MODE/READ_MODE.
COMMON=(--checkpoint "$FRONT_CKPT" --run_root "$SOURCE_RUN"
  --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
  --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}"
  --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}"
  --source_head hard --pool_mode hard --read_mode hard
  --device "${DEVICE:-cuda:0}" --workers "${WORKERS:-4}")
SMOKE=()
if [[ "$MODE" == smoke_* ]]; then SMOKE=(--smoke); fi

build_bank() {
  "$PYTHON" -B -u -m "$M.build_bank" "${COMMON[@]}" "${SMOKE[@]}" --output "$1" \
    --per_bucket "${PER_BUCKET:-64}" --seed "${SEED:-3407}"
}
train_heads() {
  local steps="${STEPS:-5000}" interval="${EVAL_EVERY:-500}"
  if [[ "$MODE" == smoke_* ]]; then steps=20; interval=20; fi
  local extra=()
  if [[ "${ALLOW_SAME_EVENT:-0}" == 1 ]]; then extra=(--allow_same_event); fi
  "$PYTHON" -B -u -m "$M.train" "${COMMON[@]}" "${SMOKE[@]}" \
    --output "$1" --bank "$2" --steps "$steps" --eval_every "$interval" \
    --batch "${BATCH:-2}" --accum "${ACCUM:-8}" --lr "${LR:-1e-4}" --seed "${SEED:-3407}" \
    --topk "${TOPK:-8}" --query_chunk "${QUERY_CHUNK:-256}" "${extra[@]}"
}
case "$MODE" in
  all|smoke_all)
    : "${MEMORY_ROOT:?Set a NEW isolated MEMORY_ROOT}"
    if [[ -e "$MEMORY_ROOT" ]]; then echo "Preserve existing results; choose a new MEMORY_ROOT: $MEMORY_ROOT" >&2; exit 2; fi
    build_bank "$MEMORY_ROOT/bank"
    train_heads "$MEMORY_ROOT/train" "$MEMORY_ROOT/bank"
    ;;
  bank|smoke_bank)
    build_bank "${OUTPUT_ROOT:-${MEMORY_ROOT:?Set OUTPUT_ROOT or MEMORY_ROOT}/bank}"
    ;;
  train|smoke_train)
    train_heads "${OUTPUT_ROOT:-${MEMORY_ROOT:?Set OUTPUT_ROOT or MEMORY_ROOT}/train}" \
                "${BANK_ROOT:-${MEMORY_ROOT:?Set BANK_ROOT or MEMORY_ROOT}/bank}"
    ;;
esac
echo "COMPLETE $MODE. Source weights unchanged. Train/Val only."
