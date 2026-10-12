#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
M=bda_benchmark.experiments.damageformer_memory_readout_audit_v1
MODE="${1:-preflight}"

die() { echo "$*" >&2; exit 2; }
absolute_path() {
  [[ "$2" == /* ]] || die "$1 must be an absolute server path: $2"
}
nonempty_file() {
  [[ -s "$1" ]] || die "Missing or empty required file: $1"
}

[[ "$#" -le 1 ]] || die 'Use one action: preflight|smoke|audit|report; configure paths through environment variables.'
case "$MODE" in
  preflight)
    "$PYTHON" -B -m unittest "$M.test_core" "$M.test_report" \
      "$M.test_provenance" "$M.test_audit" -v
    "$PYTHON" -B -m "$M.audit" --help
    "$PYTHON" -B -m "$M.report" --help
    echo 'CPU synthetic contracts passed. Real checkpoint/data/CUDA verification still requires smoke and full audit.'
    exit 0
    ;;
  report)
    : "${REPORT_ROOT:?Set REPORT_ROOT explicitly to an existing audit result directory}"
    absolute_path REPORT_ROOT "$REPORT_ROOT"
    [[ -d "$REPORT_ROOT" ]] || die "REPORT_ROOT does not exist: $REPORT_ROOT"
    # Reporting is independent of GPU, checkpoints, source paths and OUTPUT_ROOT.
    "$PYTHON" -B -m "$M.report" --root "$REPORT_ROOT" --bootstrap "${BOOTSTRAP:-0}"
    exit 0
    ;;
  smoke|audit) ;;
  *) die 'Use preflight|smoke|audit|report. This package has no train action.' ;;
esac

: "${FRONT_CKPT:?Set the immutable formal front_snapshot.pth}"
: "${SOURCE_RUN:?Set the completed formal ROI_EXCHANGE_V1 source run}"
: "${MEMORY_RUN:?Set the completed formal ROI_MEMORY_BALANCED_V2 training run}"
: "${BANK_ROOT:?Set the completed formal Train-only dual-bank directory}"
: "${OUTPUT_ROOT:?Set a new absolute audit output directory}"
absolute_path FRONT_CKPT "$FRONT_CKPT"
absolute_path SOURCE_RUN "$SOURCE_RUN"
absolute_path MEMORY_RUN "$MEMORY_RUN"
absolute_path BANK_ROOT "$BANK_ROOT"
absolute_path OUTPUT_ROOT "$OUTPUT_ROOT"
nonempty_file "$FRONT_CKPT"
nonempty_file "$SOURCE_RUN/manifest.json"
nonempty_file "$SOURCE_RUN/summary.json"
nonempty_file "$MEMORY_RUN/manifest.json"
nonempty_file "$MEMORY_RUN/summary.json"
nonempty_file "$BANK_ROOT/manifest.json"
nonempty_file "$BANK_ROOT/bank_all.pt"
nonempty_file "$BANK_ROOT/bank_hard.pt"
if [[ -e "$OUTPUT_ROOT" || -L "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.lock" || -L "${OUTPUT_ROOT}.lock" ]]; then
  die "Output or lock already exists; preserve it and choose a new OUTPUT_ROOT: $OUTPUT_ROOT"
fi

if [[ -n "${TOPK+x}" || -n "${QUERY_CHUNK+x}" || -n "${SEED+x}" ]]; then
  die 'TOPK, QUERY_CHUNK and SEED overrides are unsupported. Unset them; values come from the saved training manifest.'
fi

ARGS=(
  --checkpoint "$FRONT_CKPT"
  --run_root "$SOURCE_RUN"
  --memory_run "$MEMORY_RUN"
  --bank "$BANK_ROOT"
  --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
  --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}"
  --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}"
  --output "$OUTPUT_ROOT"
  --source_head hard --pool_mode hard --read_mode hard
  --device "${DEVICE:-cuda:0}"
  --workers "${WORKERS:-4}"
)
[[ "$MODE" == smoke ]] && ARGS+=(--smoke)

"$PYTHON" -B -u -m "$M.audit" "${ARGS[@]}"
echo "COMPLETE $MODE. Results: $OUTPUT_ROOT. Read-only trained-reader audit; no training; no Test access."
