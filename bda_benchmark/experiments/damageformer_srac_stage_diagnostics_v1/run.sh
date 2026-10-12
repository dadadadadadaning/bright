#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_stage_diagnostics_v1
MODE="${1:-preflight}"
case "$MODE" in
  preflight)
    "$PYTHON" -m unittest "$MODULE.test_feature" "$MODULE.test_memory" "$MODULE.test_entry" \
      "$MODULE.test_grouped_pool" "$MODULE.test_recheck" -v
    "$PYTHON" -m "$MODULE.feature_audit" --help
    "$PYTHON" -m "$MODULE.memory_audit" --help
    "$PYTHON" -m "$MODULE.recheck_feature" --help
    "$PYTHON" -m "$MODULE.report" --help
    echo "Local stage-diagnostic contracts passed. Real-data/CUDA smoke is still required. No training."
    exit 0 ;;
  report)
    : "${DIAG_OUTPUT:?Set the existing diagnostic output directory}"
    "$PYTHON" -m "$MODULE.report" --root "$DIAG_OUTPUT"
    exit 0 ;;
  recheck_feature)
    : "${DIAG_FAILED_RUN:?Set the existing FAILED feature-audit directory}"
    : "${DIAG_OUTPUT:?Set a NEW directory outside the failed run}"
    export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG-:4096:8}"
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
    EXTRA=()
    case "${DIAG_RECHECK_PREFIX:-0}" in
      0) ;;
      1) EXTRA+=(--prefix) ;;
      *) echo "DIAG_RECHECK_PREFIX must be 0 or 1"; exit 2 ;;
    esac
    "$PYTHON" -u -m "$MODULE.recheck_feature" --failed_run "$DIAG_FAILED_RUN" \
      --output "$DIAG_OUTPUT" "${EXTRA[@]}"
    echo "COMPLETE recheck_feature. Old failed output preserved. No training. No Test."
    exit 0 ;;
  smoke_feature|feature|smoke_memory|memory) ;;
  *) echo "Use preflight|smoke_feature|feature|recheck_feature|smoke_memory|memory|report. No training command exists."; exit 2 ;;
esac
: "${A1_CKPT:?Set the exact audited original A1 checkpoint}"
: "${DIAG_OUTPUT:?Set a NEW directory; do not mkdir this output in advance}"
test -s "$A1_CKPT"
if [[ -e "$DIAG_OUTPUT" ]]; then
  echo "Output exists; preserve it and choose a new DIAG_OUTPUT: $DIAG_OUTPUT"
  exit 2
fi
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG-:4096:8}"
case "$CUBLAS_WORKSPACE_CONFIG" in :4096:8|:16:8) ;; *) echo "Invalid CUBLAS_WORKSPACE_CONFIG"; exit 2 ;; esac
COMMON=(
  --a1_checkpoint "$A1_CKPT"
  --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
  --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}"
  --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}"
  --output "$DIAG_OUTPUT" --device "${DEVICE:-cuda:0}"
  --workers "${WORKERS:-4}" --seed "${SEED:-3407}"
)
EXTRA=()
if [[ "$MODE" == smoke_* ]]; then EXTRA+=(--smoke); fi
if [[ "$MODE" == *feature ]]; then
  : "${SNAPSHOT_CKPT:?Set a stable snapshot with adjacent snapshot.json, NOT live best.pth}"
  test -s "$SNAPSHOT_CKPT"
  test -s "$(dirname "$SNAPSHOT_CKPT")/snapshot.json"
  if [[ "$MODE" != smoke_* ]]; then
    EXTRA+=(--start_index "${DIAG_START_INDEX:-0}" --max_images "${DIAG_MAX_IMAGES:-0}")
  fi
  if [[ "${DIAG_EMBEDDING_INTERVENTIONS:-0}" == 1 ]]; then EXTRA+=(--embedding_interventions); fi
  "$PYTHON" -u -m "$MODULE.feature_audit" "${COMMON[@]}" --checkpoint "$SNAPSHOT_CKPT" "${EXTRA[@]}"
else
  : "${DIAG_JOINT_RUN:?Set the completed directory containing j1_last.pth, j2_last.pth, j3_last.pth and manifest.json}"
  test -s "$DIAG_JOINT_RUN/manifest.json"
  for head in j1 j2 j3; do test -s "$DIAG_JOINT_RUN/${head}_last.pth"; done
  for setting in FRONT CACHE METRIC; do
    variable="DIAG_MEMORY_${setting}"
    value="${!variable:-}"
    if [[ -n "$value" ]]; then
      case "$setting" in FRONT) EXTRA+=(--checkpoint "$value");; CACHE) EXTRA+=(--cache "$value");; METRIC) EXTRA+=(--metric_checkpoint "$value");; esac
    fi
  done
  inventory_default=1
  if [[ "$MODE" == smoke_* ]]; then inventory_default=0; fi
  if [[ "${DIAG_INVENTORY_SHARDS:-$inventory_default}" == 1 ]]; then EXTRA+=(--inventory_shards); fi
  read -r -a audit_arms <<< "${DIAG_INTERVENTION_ARMS:-j2 j3}"
  "$PYTHON" -u -m "$MODULE.memory_audit" "${COMMON[@]}" --joint_run "$DIAG_JOINT_RUN" \
    --temperature "${DIAG_MEMORY_TEMPERATURE:-0.1}" --intervention_arms "${audit_arms[@]}" "${EXTRA[@]}"
fi
echo "COMPLETE $MODE. Only this audit ran. No training. No Test."
