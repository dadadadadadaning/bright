#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_hardquery_memory_v1
MODE="${1:-preflight}"

case "$MODE" in
  preflight)
    "$PYTHON" -m unittest discover -s "${MODULE//.//}" -t "$ROOT" -p 'test_*.py' -v
    for entry in protocol evidence prepare_queries train_metric train_reader report; do
      "$PYTHON" -m "$MODULE.$entry" --help
    done
    echo 'CPU contracts only; formal BRIGHT/CUDA validation remains required.'
    exit 0 ;;
  report)
    : "${RUN_ROOT:?Set the existing experiment root}"
    "$PYTHON" -m "$MODULE.report" --root "$RUN_ROOT"
    exit 0 ;;
  compare)
    : "${LEFT_RUN:?Set first E0 directory}"
    : "${RIGHT_RUN:?Set second E0 directory}"
    : "${OUTPUT:?Set a new comparison output directory}"
    "$PYTHON" -m "$MODULE.protocol" compare --left "$LEFT_RUN" --right "$RIGHT_RUN" --output "$OUTPUT"
    exit 0 ;;
  protocol|e0_pair|evidence|queries|metric|reader|smoke|pipeline) ;;
  *) echo 'Use preflight|e0_pair|protocol|compare|evidence|queries|metric|reader|smoke|pipeline|report'; exit 2 ;;
esac

: "${JOINT_RUN:?Set the completed original J1/J2/J3 source directory}"
: "${A1_CKPT:?Set the exact original A1 checkpoint}"
: "${RUN_ROOT:?Set a new experiment root (stages will not overwrite existing outputs)}"
COMMON=(--joint_run "$JOINT_RUN" --a1_checkpoint "$A1_CKPT"
  --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
  --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}"
  --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}"
  --device "${DEVICE:-cuda:0}" --workers "${WORKERS:-4}" --seed "${SEED:-3407}"
  --pool_mode "${POOL_MODE:-native}")
if [[ -n "${FRONT_CKPT:-}" ]]; then COMMON+=(--checkpoint "$FRONT_CKPT"); fi
if [[ -n "${CACHE:-}" ]]; then COMMON+=(--cache "$CACHE"); fi
if [[ -n "${OLD_METRIC_CKPT:-}" ]]; then COMMON+=(--metric_checkpoint "$OLD_METRIC_CKPT"); fi
OVERRIDES=()
if [[ -n "${MATMUL_TF32:-}" ]]; then OVERRIDES+=(--matmul_tf32 "$MATMUL_TF32"); fi
if [[ -n "${CUDNN_TF32:-}" ]]; then OVERRIDES+=(--cudnn_tf32 "$CUDNN_TF32"); fi
if [[ -n "${DETERMINISTIC:-}" ]]; then OVERRIDES+=(--deterministic "$DETERMINISTIC"); fi

if [[ "$MODE" == e0_pair ]]; then
  if (( ${#OVERRIDES[@]} > 0 )); then echo 'e0_pair requires no numeric overrides; use protocol for single-factor contrasts'; exit 2; fi
  STATUS=0
  # Fresh process for every repeat. A failed reproduction stays failed, while
  # the second profile is still evaluated for diagnosis.
  for PROFILE_VALUE in legacy strict; do
    for REPEAT in 1 2; do
      "$PYTHON" -u -m "$MODULE.protocol" "${COMMON[@]}" --profile "$PROFILE_VALUE" \
        --scope a1 --output "$RUN_ROOT/e0_${PROFILE_VALUE}_a1_${REPEAT}" || STATUS=1
    done
  done
  echo "Compare e0_legacy_a1_1 with e0_legacy_a1_2, then with e0_strict_a1_1 using compare."
  exit "$STATUS"
fi

SETTINGS=(--profile "${PROFILE:-legacy}" "${OVERRIDES[@]}")
PROTOCOL_RUN="${PROTOCOL_RUN:-$RUN_ROOT/protocol}"
QUERY_RUN="${QUERY_RUN:-$RUN_ROOT/queries}"
METRIC_RUN="${METRIC_RUN:-$RUN_ROOT/metric}"
run_protocol() {
  "$PYTHON" -u -m "$MODULE.protocol" "${COMMON[@]}" "${SETTINGS[@]}" \
    --scope "${SCOPE:-all}" --output "${OUTPUT:-$RUN_ROOT/protocol}" "$@"
}
run_evidence() {
  "$PYTHON" -u -m "$MODULE.evidence" "${COMMON[@]}" "${SETTINGS[@]}" --protocol "$PROTOCOL_RUN" \
    --temperature "${TEMPERATURE:-0.1}" --output "${OUTPUT:-$RUN_ROOT/evidence}" "$@"
}
run_queries() {
  "$PYTHON" -u -m "$MODULE.prepare_queries" "${COMMON[@]}" "${SETTINGS[@]}" --protocol "$PROTOCOL_RUN" \
    --output "${OUTPUT:-$RUN_ROOT/queries}" "$@"
}
run_metric() {
  "$PYTHON" -u -m "$MODULE.train_metric" "${COMMON[@]}" "${SETTINGS[@]}" --protocol "$PROTOCOL_RUN" \
    --queries "$QUERY_RUN" --arms bank hard --output "${OUTPUT:-$RUN_ROOT/metric}" "$@"
}
run_reader() {
  "$PYTHON" -u -m "$MODULE.train_reader" "${COMMON[@]}" "${SETTINGS[@]}" --protocol "$PROTOCOL_RUN" \
    --metrics "$METRIC_RUN" --temperature "${TEMPERATURE:-0.1}" --output "${OUTPUT:-$RUN_ROOT/reader}" "$@"
}

case "$MODE" in
  protocol) run_protocol ;;
  evidence) run_evidence ;;
  queries) run_queries ;;
  metric) run_metric ;;
  reader) run_reader ;;
  smoke|pipeline)
    if [[ -n "${OUTPUT:-}" || "${SCOPE:-all}" != all ]]; then
      echo 'smoke/pipeline requires OUTPUT unset and SCOPE=all'; exit 2
    fi
    if [[ "$MODE" == smoke ]]; then
      run_protocol --smoke
      run_evidence --smoke
      run_queries --smoke
      run_metric --smoke --steps 20
      run_reader --smoke --steps 20 --eval_every 20
    else
      run_protocol
      run_evidence
      run_queries
      run_metric
      run_reader
    fi
    "$PYTHON" -m "$MODULE.report" --root "$RUN_ROOT"
    ;;
esac
