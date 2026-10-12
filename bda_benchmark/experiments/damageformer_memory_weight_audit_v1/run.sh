#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_memory_weight_audit_v1
MODE="${1:-preflight}"
die() { echo "$*" >&2; exit 2; }
[[ "$#" -le 1 ]] || die 'Use preflight|smoke|audit|report. No training action.'

case "$MODE" in
  preflight)
    "$PYTHON" -B -m unittest "$MODULE.test_core" "$MODULE.test_inputs" \
      "$MODULE.test_audit" "$MODULE.test_report" -v
    "$PYTHON" -B -m "$MODULE.audit" --help
    "$PYTHON" -B -m "$MODULE.report" --help
    echo 'CPU contracts only. Real trained checkpoints/CUDA still require smoke and formal Val audit.'
    exit 0
    ;;
  report)
    : "${REPORT_ROOT:?Set REPORT_ROOT to an existing MEMORY_WEIGHT_AUDIT_V1 output}"
    [[ "$REPORT_ROOT" == /* && -d "$REPORT_ROOT" ]] || die 'REPORT_ROOT must be an existing absolute directory'
    "$PYTHON" -B -m "$MODULE.report" --root "$REPORT_ROOT" --bootstrap "${BOOTSTRAP:-0}"
    exit 0
    ;;
  smoke|audit) ;;
  *) die 'Use preflight|smoke|audit|report. Do not train or rebuild a bank for this experiment.' ;;
esac

: "${PARENT_AUDIT:?Set the completed MEMORY_READOUT_AUDIT_V1 full directory}"
: "${OUTPUT_ROOT:?Set a NEW absolute MEMORY_WEIGHT_AUDIT_V1 output directory}"
[[ "$PARENT_AUDIT" == /* && -d "$PARENT_AUDIT" ]] || die 'PARENT_AUDIT must exist and be an absolute directory'
[[ "$OUTPUT_ROOT" == /* ]] || die 'OUTPUT_ROOT must be absolute'
[[ -s "$PARENT_AUDIT/manifest.json" && -s "$PARENT_AUDIT/summary.json" ]] || die 'Parent audit manifest/summary missing'
if [[ -e "$OUTPUT_ROOT" || -L "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.lock" || -L "${OUTPUT_ROOT}.lock" ]]; then
  die "Preserve existing results/lock; choose a new OUTPUT_ROOT: $OUTPUT_ROOT"
fi
# Only the completed parent selects files and retrieval settings. Old exports
# such as FRONT_CKPT, MEMORY_RUN, BANK_ROOT, TOPK and SEED cannot alter this run.
ARGS=(--parent_audit "$PARENT_AUDIT" --output "$OUTPUT_ROOT"
      --device "${DEVICE:-cuda:0}" --workers "${WORKERS:-4}")
[[ "$MODE" == smoke ]] && ARGS+=(--smoke)
echo "Reading immutable inputs/settings from: $PARENT_AUDIT"
echo "New output: $OUTPUT_ROOT"
"$PYTHON" -B -u -m "$MODULE.audit" "${ARGS[@]}"
echo "COMPLETE $MODE: $OUTPUT_ROOT. Inference interventions only; no optimizer, no Test."
