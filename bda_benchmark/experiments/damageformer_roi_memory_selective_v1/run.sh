#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
M=bda_benchmark.experiments.damageformer_roi_memory_selective_v1
MODE="${1:-preflight}"

if [[ "$MODE" == preflight ]]; then
  if ! compgen -G \
    'bda_benchmark/experiments/damageformer_roi_memory_selective_v1/test_*.py' \
    >/dev/null; then
    echo 'Selective-Memory tests are missing; do not treat a zero-test run as preflight.' >&2
    exit 2
  fi
  "$PYTHON" -B -m unittest discover \
    -s bda_benchmark/experiments/damageformer_roi_memory_selective_v1 \
    -t . -p 'test_*.py' -v
  "$PYTHON" -B -m "$M.audit" --help
  "$PYTHON" -B -m "$M.train" --help
  "$PYTHON" -B -m "$M.report" --help
  echo 'Selective-Memory contracts passed. Real BRIGHT/CUDA smoke is still required.'
  exit 0
fi

if [[ "$MODE" == report ]]; then
  "$PYTHON" -B -m "$M.report" \
    --root "${OUTPUT_ROOT:?Set a completed audit/train output}"
  exit 0
fi

case "$MODE" in
  audit|smoke_audit|retrieval|smoke_retrieval|posterior|smoke_posterior) ;;
  *)
    echo 'Use preflight|audit|smoke_audit|retrieval|smoke_retrieval|posterior|smoke_posterior|report' >&2
    exit 2
    ;;
esac

test -s "${FRONT_CKPT:?Set immutable V2 front_snapshot.pth}"
test -s "${SOURCE_RUN:?Set completed ROI_EXCHANGE_V1 source run}/manifest.json"
test -s "${BANK_ROOT:?Set completed balanced-v2 Train bank}/bank_all.pt"
test -s "$BANK_ROOT/bank_hard.pt"
test -s "$BANK_ROOT/manifest.json"
: "${OUTPUT_ROOT:?Set a new output directory}"
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.lock" ]]; then
  echo "Output/lock exists; preserve it and choose a new OUTPUT_ROOT: $OUTPUT_ROOT" >&2
  exit 2
fi

COMMON=(
  --checkpoint "$FRONT_CKPT"
  --run_root "$SOURCE_RUN"
  --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
  --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}"
  --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}"
  --output "$OUTPUT_ROOT"
  --source_head hard --pool_mode hard --read_mode hard
  --device "${DEVICE:-cuda:0}"
  --workers "${WORKERS:-4}"
)
[[ "$MODE" == smoke_* ]] && COMMON+=(--smoke)

SELECTIVE=(
  --bank "$BANK_ROOT"
  --topk "${TOPK:-8}"
  --preselect "${PRESELECT:-32}"
  --per_class_cap "${PER_CLASS_CAP:-4}"
  --quality_quantile "${QUALITY_QUANTILE:-0.25}"
  --min_cosine "${MIN_COSINE:--1.0}"
  --temperature "${TEMPERATURE:-0.07}"
)
[[ "${ALLOW_SAME_EVENT:-0}" == 1 ]] && SELECTIVE+=(--allow_same_event)

case "$MODE" in
  audit|smoke_audit)
    "$PYTHON" -B -u -m "$M.audit" "${COMMON[@]}" "${SELECTIVE[@]}" \
      --seed "${SEED:-3407}"
    ;;
  retrieval|smoke_retrieval|posterior|smoke_posterior)
    STAGE=retrieval
    [[ "$MODE" == posterior || "$MODE" == smoke_posterior ]] && STAGE=posterior
    COMMON+=(
      --batch "${BATCH:-2}"
      --accum "${ACCUM:-8}"
      --seed "${SEED:-3407}"
      --lr "${LR:-1e-4}"
    )
    if [[ "$MODE" == smoke_* ]]; then
      COMMON+=(--steps 20 --eval_every 20)
    else
      COMMON+=(
        --steps "${STEPS:-5000}"
        --eval_every "${EVAL_EVERY:-500}"
      )
    fi
    "$PYTHON" -B -u -m "$M.train" "${COMMON[@]}" "${SELECTIVE[@]}" \
      --stage "$STAGE" --query_chunk "${QUERY_CHUNK:-256}" \
      --posterior_retrieval "${POSTERIOR_RETRIEVAL:-adaptive_cap}"
    ;;
esac

echo "COMPLETE $MODE. Results: $OUTPUT_ROOT. No other stage launched. Train/Val only."
