#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_selective_readout_v1
MODE="${1:-preflight}"
case "$MODE" in
  preflight)
    "$PYTHON" -m unittest "$MODULE.test_core" "$MODULE.test_evidence" "$MODULE.test_entry" "$MODULE.test_probability_check" -v
    for entry in boundary_audit train report; do "$PYTHON" -m "$MODULE.$entry" --help >/dev/null; done
    echo "Selective-readout CPU contracts passed. Real BRIGHT/CUDA smoke still required. No training launched."
    exit 0 ;;
  snapshot)
    : "${SELECTIVE_ROOT:?Set a new isolated output root}"
    : "${PARENT_RUN:?Set the SRAC_TRANSFORMER_V1 run containing front/best.pth}"
    "$PYTHON" -m bda_benchmark.experiments.damageformer_srac_cause_audit_v1.snapshot \
      --source "$PARENT_RUN/front/best.pth" --output "$SELECTIVE_ROOT/snapshot"
    exit 0 ;;
  report)
    : "${SELECTIVE_REPORT:?Set the actual boundary/readout output directory}"
    "$PYTHON" -m "$MODULE.report" --root "$SELECTIVE_REPORT"
    exit 0 ;;
  smoke_boundary|boundary|boundary_probe|smoke_readout|train) ;;
  *) echo "Use preflight|snapshot|smoke_boundary|boundary|boundary_probe|smoke_readout|train|report"; exit 2 ;;
esac
: "${SELECTIVE_ROOT:?Set a new isolated output root}"
: "${SNAPSHOT_CKPT:?Set front_snapshot.pth with adjacent snapshot.json, NOT old V2/A1 weights}"
: "${A1_CKPT:?Set original audited A1 best_val_miou.pth}"
test -s "$SNAPSHOT_CKPT"
test -s "$(dirname "$SNAPSHOT_CKPT")/snapshot.json"
test -s "$A1_CKPT"
COMMON=(--checkpoint "$SNAPSHOT_CKPT" --a1_checkpoint "$A1_CKPT"
  --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
  --train_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt"
  --val_list "$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt"
  --device "${DEVICE:-cuda:0}" --workers "${WORKERS:-4}" --seed "${SEED:-3407}")
EXTRA=()
if [[ "$MODE" == smoke_* ]]; then EXTRA=(--smoke); fi
if [[ "$MODE" == boundary_probe ]]; then
  EXTRA=(--start_index "${BOUNDARY_START_INDEX:-330}" --max_images "${BOUNDARY_MAX_IMAGES:-10}")
fi
case "$MODE" in
  smoke_boundary|boundary|boundary_probe)
    "$PYTHON" -u -m "$MODULE.boundary_audit" "${COMMON[@]}" "${EXTRA[@]}" \
      --output "${SELECTIVE_OUTPUT:-$SELECTIVE_ROOT/$MODE}"
    ;;
  smoke_readout|train)
    OUTPUT_NAME=readout
    if [[ "$MODE" == smoke_readout ]]; then OUTPUT_NAME=readout_smoke; fi
    "$PYTHON" -u -m "$MODULE.train" "${COMMON[@]}" "${EXTRA[@]}" \
      --output "${SELECTIVE_OUTPUT:-$SELECTIVE_ROOT/$OUTPUT_NAME}" \
      --steps "${SELECTIVE_STEPS:-5000}" --eval_every "${SELECTIVE_EVAL_EVERY:-500}" \
      --batch "${SELECTIVE_BATCH:-2}" --accum "${SELECTIVE_ACCUM:-8}" \
      --lr "${SELECTIVE_LR:-1e-4}" --amp "${SELECTIVE_AMP:-none}" \
      --score_dim "${SELECTIVE_SCORE_DIM:-32}" --chunk_size "${SELECTIVE_CHUNK:-4096}"
    ;;
esac
echo "COMPLETE $MODE. No other stage launched. Parent unchanged. No Memory. No Test."
