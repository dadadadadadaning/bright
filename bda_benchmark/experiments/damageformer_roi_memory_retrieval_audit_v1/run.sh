#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
M=bda_benchmark.experiments.damageformer_roi_memory_retrieval_audit_v1
MODE="${1:-preflight}"

if [[ "$MODE" == preflight ]]; then
  "$PYTHON" -B -m unittest "$M.test_audit" -v
  "$PYTHON" -B -m "$M.audit" --help
  "$PYTHON" -B -m "$M.report" --help
  echo 'Retrieval-audit contracts passed. Real BRIGHT/CUDA smoke is still required.'
  exit 0
fi

if [[ "$MODE" == report ]]; then
  "$PYTHON" -B -m "$M.report" \
    --root "${OUTPUT_ROOT:?Set a completed audit output}"
  exit 0
fi

case "$MODE" in
  audit|smoke) ;;
  *) echo 'Use preflight|smoke|audit|report' >&2; exit 2 ;;
esac

test -s "${FRONT_CKPT:?Set immutable V2 front_snapshot.pth}"
test -s "${SOURCE_RUN:?Set completed ROI_EXCHANGE_V1 source run}/manifest.json"
test -s "${BANK_ROOT:?Set completed balanced-v2 Train bank}/bank_all.pt"
test -s "$BANK_ROOT/bank_hard.pt"
test -s "$BANK_ROOT/manifest.json"
: "${OUTPUT_ROOT:?Set a new output directory}"
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.lock" ]]; then
  echo "Output/lock exists; choose a new OUTPUT_ROOT: $OUTPUT_ROOT" >&2
  exit 2
fi

ARGS=(
  --checkpoint "$FRONT_CKPT"
  --run_root "$SOURCE_RUN"
  --bank "$BANK_ROOT"
  --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
  --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}"
  --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}"
  --output "$OUTPUT_ROOT"
  --source_head hard --pool_mode hard --read_mode hard
  --topk "${TOPK:-8}"
  --preselect "${PRESELECT:-32}"
  --per_class_cap "${PER_CLASS_CAP:-4}"
  --quality_quantile "${QUALITY_QUANTILE:-0.25}"
  --min_cosine "${MIN_COSINE:--1.0}"
  --temperature "${TEMPERATURE:-0.07}"
  --seed "${SEED:-3407}"
  --device "${DEVICE:-cuda:0}"
  --workers "${WORKERS:-4}"
)
# Use the *formal* source and formal bank for a two-image pipeline check.  The
# parent bank contract intentionally rejects mixing a smoke bank and formal
# source (or vice versa), so this is a limit rather than a different identity.
[[ "$MODE" == smoke ]] && ARGS+=(--limit_images 2)

"$PYTHON" -B -u -m "$M.audit" "${ARGS[@]}"
echo "COMPLETE $MODE. Results: $OUTPUT_ROOT. No training; Train-bank/Val-query only; no Test."

