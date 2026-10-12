#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_transformer_v1
MODE="${1:-preflight}"

if [[ "$MODE" == preflight ]]; then
  "$PYTHON" -m unittest "$MODULE.test_contracts" -v
  for entry in train_front build_memory build_neighbors audit_retrieval train_reader evaluate; do
    "$PYTHON" -m "$MODULE.$entry" --help >/dev/null
  done
  echo "Transformer-SRAC contracts passed. Real BRIGHT/CUDA smoke is still required. Test is forbidden."
  exit 0
fi

case "$MODE" in
  smoke_all|pipeline|front|build_memory|build_neighbors|audit_retrieval|train_readers|evaluate) ;;
  *) echo "Use preflight|smoke_all|front|build_memory|build_neighbors|audit_retrieval|train_readers|evaluate|pipeline"; exit 2 ;;
esac

: "${RUN_ROOT:?Set a NEW isolated RUN_ROOT}"
: "${A1_CKPT:?Set the audited original A1 checkpoint path}"
test -s "$A1_CKPT"

DATASET="${DATASET:-$ROOT/bda_benchmark/dataset}"
TRAIN_LIST="$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt"
VAL_LIST="$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt"
DEVICE="${DEVICE:-cuda:0}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-3407}"
TOPK="${TOPK:-8}"
RETRIEVAL_POLICY="${RETRIEVAL_POLICY:-global}"
EXCLUSION="${EXCLUSION:-cross_event}"
QUERY_SAMPLING="${QUERY_SAMPLING:-uniform}"

COMMON=(
  --dataset "$DATASET"
  --train_list "$TRAIN_LIST"
  --val_list "$VAL_LIST"
  --workers "$WORKERS"
  --device "$DEVICE"
)
EXTRA=()
FRONT_STEPS=50000
FRONT_EVAL=1000
READER_STEPS=5000
READER_EVAL=500
if [[ "$MODE" == smoke_all ]]; then
  EXTRA=(--smoke)
  FRONT_STEPS=20
  FRONT_EVAL=20
  READER_STEPS=20
  READER_EVAL=20
fi

FRONT_CKPT="$RUN_ROOT/front/best.pth"
CACHE="$RUN_ROOT/train_memory"
NEIGHBORS="$RUN_ROOT/neighbors_${RETRIEVAL_POLICY}_${EXCLUSION}"

front() {
  CONTEXT_ARGUMENT=()
  if [[ "${REGION_CONTEXT:-0}" == 1 ]]; then CONTEXT_ARGUMENT=(--region_context); fi
  "$PYTHON" -u -m "$MODULE.train_front" \
    "${COMMON[@]}" "${EXTRA[@]}" \
    --a1_checkpoint "$A1_CKPT" \
    --output "$RUN_ROOT/front" \
    --steps "$FRONT_STEPS" --eval_every "$FRONT_EVAL" \
    --batch "${BATCH:-2}" --accum "${ACCUM:-8}" \
    --seed "$SEED" --amp "${AMP:-bf16}" \
    --dim "${DIM:-128}" --decoder_width "${DECODER_WIDTH:-64}" \
    --stride "${STRIDE:-8}" --iterations "${ITERATIONS:-3}" \
    --temperature "${TEMPERATURE:-0.2}" \
    --spatial_weight "${SPATIAL_WEIGHT:-0.1}" \
    --building_weight "${BUILDING_WEIGHT:-0.5}" \
    "${CONTEXT_ARGUMENT[@]}"
}

build_memory() {
  "$PYTHON" -u -m "$MODULE.build_memory" \
    "${COMMON[@]}" "${EXTRA[@]}" \
    --checkpoint "$FRONT_CKPT" --a1_checkpoint "$A1_CKPT" \
    --output "$CACHE" --per_event_class "${PER_EVENT_CLASS:-64}" \
    --min_purity "${MIN_PURITY:-0.0}" --seed "$SEED"
}

build_neighbors() {
  "$PYTHON" -u -m "$MODULE.build_neighbors" \
    --checkpoint "$FRONT_CKPT" --cache "$CACHE" --output "$NEIGHBORS" \
    --topk "$TOPK" --policy "$RETRIEVAL_POLICY" --exclusion "$EXCLUSION" \
    --device "$DEVICE"
}

audit_retrieval() {
  "$PYTHON" -u -m "$MODULE.audit_retrieval" \
    --checkpoint "$FRONT_CKPT" --cache "$CACHE" --neighbors "$NEIGHBORS" \
    --output "$RUN_ROOT/retrieval_audit_${RETRIEVAL_POLICY}_${EXCLUSION}" \
    --topk "$TOPK" --policy "$RETRIEVAL_POLICY" --exclusion "$EXCLUSION"
}

train_readers() {
  for READER_MODE in nomemory memory; do
    NEIGHBOR_ARGUMENT=()
    if [[ "$READER_MODE" == memory ]]; then NEIGHBOR_ARGUMENT=(--neighbors "$NEIGHBORS"); fi
    "$PYTHON" -u -m "$MODULE.train_reader" \
      "${COMMON[@]}" "${EXTRA[@]}" \
      --checkpoint "$FRONT_CKPT" --a1_checkpoint "$A1_CKPT" \
      --cache "$CACHE" "${NEIGHBOR_ARGUMENT[@]}" \
      --mode "$READER_MODE" --output "$RUN_ROOT/${READER_MODE}_${RETRIEVAL_POLICY}_${EXCLUSION}" \
      --steps "$READER_STEPS" --eval_every "$READER_EVAL" \
      --batch "${READER_BATCH:-128}" --topk "$TOPK" --seed "$SEED" \
      --retrieval_policy "$RETRIEVAL_POLICY" --exclusion "$EXCLUSION" \
      --query_sampling "$QUERY_SAMPLING"
  done
}

evaluate() {
  local NOM="$RUN_ROOT/nomemory_${RETRIEVAL_POLICY}_${EXCLUSION}/best.pth"
  local MEM="$RUN_ROOT/memory_${RETRIEVAL_POLICY}_${EXCLUSION}/best.pth"
  if [[ ! -e "$RUN_ROOT/evaluation/front" ]]; then
    "$PYTHON" -u -m "$MODULE.evaluate" \
      "${COMMON[@]}" "${EXTRA[@]}" \
      --checkpoint "$FRONT_CKPT" --a1_checkpoint "$A1_CKPT" \
      --output "$RUN_ROOT/evaluation/front"
  fi
  "$PYTHON" -u -m "$MODULE.evaluate" \
    "${COMMON[@]}" "${EXTRA[@]}" \
    --checkpoint "$FRONT_CKPT" --a1_checkpoint "$A1_CKPT" \
    --cache "$CACHE" --head "$NOM" \
    --output "$RUN_ROOT/evaluation/nomemory_${RETRIEVAL_POLICY}_${EXCLUSION}"
  for INTERVENTION in normal random shuffle_labels off no_metadata; do
    "$PYTHON" -u -m "$MODULE.evaluate" \
      "${COMMON[@]}" "${EXTRA[@]}" \
      --checkpoint "$FRONT_CKPT" --a1_checkpoint "$A1_CKPT" \
      --cache "$CACHE" --head "$MEM" --intervention "$INTERVENTION" \
      --output "$RUN_ROOT/evaluation/memory_${RETRIEVAL_POLICY}_${EXCLUSION}_${INTERVENTION}"
  done
}

case "$MODE" in
  smoke_all|pipeline)
    if [[ -e "$RUN_ROOT" ]]; then
      echo "Pipeline requires a completely NEW RUN_ROOT"; exit 2
    fi
    front
    build_memory
    build_neighbors
    audit_retrieval
    train_readers
    evaluate
    ;;
  *) "$MODE" ;;
esac

echo "COMPLETE $MODE. Train/Val only; Test was not accessed."

