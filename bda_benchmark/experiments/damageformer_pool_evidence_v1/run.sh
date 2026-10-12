#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
M=bda_benchmark.experiments.damageformer_pool_evidence_v1
MODE="${1:-preflight}"
case "$MODE" in
  preflight)
    "$PYTHON" -B -m unittest "$M.test_core" "$M.test_audit" -v
    "$PYTHON" -B -m "$M.audit" --help
    "$PYTHON" -B -m "$M.report" --help
    echo 'Local CPU contracts only. No real Val/GPU audit or training executed.'
    exit 0 ;;
  report)
    "$PYTHON" -B -m "$M.report" --root "${OUTPUT_ROOT:?Set OUTPUT_ROOT to existing audit run or JSON}"
    exit 0 ;;
  audit|smoke) ;;
  *) echo 'Use preflight | smoke | audit | report. Training intentionally unavailable.' >&2; exit 2 ;;
esac
: "${FRONT_CKPT:?Set immutable front_snapshot.pth, NOT original A1}"
: "${SOURCE_RUN:?Set completed ROI_EXCHANGE_V1 source run}"
: "${OUTPUT_ROOT:?Set NEW output directory; do not mkdir it}"
for FILE in "$FRONT_CKPT" "$SOURCE_RUN/hard_last.pth" "$SOURCE_RUN/manifest.json" "$SOURCE_RUN/summary.json"; do
  if [[ ! -s "$FILE" ]]; then echo "Missing required source: $FILE" >&2; exit 2; fi
done
if [[ -e "$OUTPUT_ROOT" ]]; then echo "Preserve existing results; choose NEW OUTPUT_ROOT: $OUTPUT_ROOT" >&2; exit 2; fi
SMOKE=()
if [[ "$MODE" == smoke ]]; then SMOKE=(--smoke); fi
exec "$PYTHON" -B -u -m "$M.audit" \
  --checkpoint "$FRONT_CKPT" --run_root "$SOURCE_RUN" --output "$OUTPUT_ROOT" \
  --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}" \
  --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}" \
  --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}" \
  --source_head hard --pool_mode hard --read_mode hard \
  --device "${DEVICE:-cuda:0}" --workers "${WORKERS:-4}" "${SMOKE[@]}"
