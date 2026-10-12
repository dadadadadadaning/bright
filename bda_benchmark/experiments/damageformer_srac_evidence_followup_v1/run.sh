#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_evidence_followup_v1
MODE="${1:-preflight}"

case "$MODE" in
  preflight)
    "$PYTHON" -m unittest discover -s "${MODULE//.//}" -t "$ROOT" -p 'test_*.py' -v
    for entry in feature_audit memory_audit recheck_memory replay_pool replay_captured report build_banks train_quota; do
      "$PYTHON" -m "$MODULE.$entry" --help
    done
    echo "CPU/interface contracts only. Real BRIGHT/CUDA and full Val gates still required."
    exit 0 ;;
  report)
    EXTRA=()
    if [[ -n "${FOLLOWUP_FEATURE_RUN:-}" ]]; then EXTRA+=(--feature "$FOLLOWUP_FEATURE_RUN"); fi
    if [[ -n "${FOLLOWUP_MEMORY_RUN:-}" ]]; then EXTRA+=(--memory "$FOLLOWUP_MEMORY_RUN"); fi
    if (( ${#EXTRA[@]} == 0 )); then echo 'Set FOLLOWUP_FEATURE_RUN and/or FOLLOWUP_MEMORY_RUN'; exit 2; fi
    "$PYTHON" -m "$MODULE.report" "${EXTRA[@]}"
    exit 0 ;;
  recheck_memory|replay_pool|replay_captured|smoke_feature|feature|smoke_memory|memory|smoke_banks|banks|smoke_train|train) ;;
  *) echo 'Use preflight|report|recheck_memory|replay_pool|replay_captured|smoke_feature|feature|smoke_memory|memory|smoke_banks|banks|smoke_train|train'; exit 2 ;;
esac
: "${FOLLOWUP_OUTPUT:?Set a NEW output directory; do not create it beforehand}"
if [[ -e "$FOLLOWUP_OUTPUT" ]]; then echo "Preserve existing output: $FOLLOWUP_OUTPUT"; exit 2; fi
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG-:4096:8}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
POOL=(--pool_mode "${POOL_MODE:-native}" --pool_capture "${POOL_CAPTURE:-failure}")
case "${POOL_CHECK_REFERENCE:-0}" in
  0) ;;
  1) POOL+=(--pool_check_reference) ;;
  *) echo 'POOL_CHECK_REFERENCE must be 0 or 1'; exit 2 ;;
esac
if [[ "$MODE" == recheck_memory ]]; then
  : "${FAILED_MEMORY:?Set the preserved failed Memory-audit directory}"
  EXTRA=()
  case "${RECHECK_PREFIX:-0}" in 0) ;; 1) EXTRA+=(--prefix) ;; *) exit 2 ;; esac
  "$PYTHON" -u -m "$MODULE.recheck_memory" --failed_run "$FAILED_MEMORY" --output "$FOLLOWUP_OUTPUT" "${POOL[@]}" "${EXTRA[@]}"
  exit 0
fi
if [[ "$MODE" == replay_pool ]]; then
  : "${POOL_INPUT:?Set a captured CPU pool-input .pth file}"
  "$PYTHON" -u -m "$MODULE.replay_pool" --input "$POOL_INPUT" --output "$FOLLOWUP_OUTPUT" \
    --backend "${POOL_BACKEND:-grouped}" --device "${DEVICE:-cuda:0}" --repeats "${POOL_REPEATS:-3}" \
    --layout "${POOL_LAYOUT:-captured}"
  exit 0
fi
if [[ "$MODE" == replay_captured ]]; then
  : "${POOL_CAPTURE_RUN:?Set the NEW single-image recheck containing pool_debug/input_*.pth}"
  "$PYTHON" -u -m "$MODULE.replay_captured" --capture_run "$POOL_CAPTURE_RUN" \
    --output "$FOLLOWUP_OUTPUT" --device "${DEVICE:-cuda:0}" --repeats "${POOL_REPEATS:-3}"
  exit 0
fi
: "${A1_CKPT:?Set the exact A1 checkpoint}"
: "${JOINT_RUN:?Set the COMPLETED formal directory containing j1_last.pth, j2_last.pth and j3_last.pth}"
test -s "$A1_CKPT"
test -s "$JOINT_RUN/manifest.json"
COMMON=(--joint_run "$JOINT_RUN" --a1_checkpoint "$A1_CKPT"
  --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
  --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}"
  --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}"
  --output "$FOLLOWUP_OUTPUT" --seed "${SEED:-3407}")
EXTRA=()
if [[ "$MODE" == smoke_* ]]; then EXTRA+=(--smoke); fi
if [[ "$MODE" == *feature ]]; then
  "$PYTHON" -u -m "$MODULE.feature_audit" "${COMMON[@]}" "${POOL[@]}" "${EXTRA[@]}" \
    --device "${DEVICE:-cuda:0}" --workers "${WORKERS:-4}" \
    --start_index "${FOLLOWUP_START_INDEX:-0}" --max_images "${FOLLOWUP_MAX_IMAGES:-0}"
elif [[ "$MODE" == *memory ]]; then
  if [[ "${INVENTORY_SHARDS:-1}" == 1 ]]; then EXTRA+=(--inventory_shards); fi
  "$PYTHON" -u -m "$MODULE.memory_audit" "${COMMON[@]}" "${POOL[@]}" "${EXTRA[@]}" \
    --device "${DEVICE:-cuda:0}" --workers "${WORKERS:-4}" \
    --start_index "${FOLLOWUP_START_INDEX:-0}" --max_images "${FOLLOWUP_MAX_IMAGES:-0}" \
    --temperature "${MEMORY_TEMPERATURE:-0.1}" --intervention_arms j2 j3
else
  if [[ "$MODE" != smoke_* ]]; then
    : "${FOLLOWUP_FEATURE_RUN:?Set the complete full-Val feature audit}"
    : "${FOLLOWUP_MEMORY_RUN:?Set the complete full-Val Memory audit}"
    : "${QUOTA_EVIDENCE_NOTE:?State the observed quota/coverage evidence; do not use a generic placeholder}"
    if [[ "${CONFIRM_QUOTA_TEST:-0}" != 1 ]]; then echo 'Set CONFIRM_QUOTA_TEST=1 only after reviewing full audit evidence'; exit 2; fi
    EXTRA+=(--feature_audit "$FOLLOWUP_FEATURE_RUN" --memory_audit "$FOLLOWUP_MEMORY_RUN"
      --evidence_note "$QUOTA_EVIDENCE_NOTE" --confirm_quota_test)
  fi
  if [[ "$MODE" == *banks ]]; then
    "$PYTHON" -u -m "$MODULE.build_banks" "${COMMON[@]}" "${EXTRA[@]}" --floor "${QUOTA_FLOOR:-16}"
  else
    : "${QUOTA_BANKS:?Set the new matched M0/M1 bank directory}"
    TRAIN_STEPS="${STEPS:-5000}"
    TRAIN_EVAL="${EVAL_EVERY:-500}"
    if [[ "$MODE" == smoke_train ]]; then TRAIN_STEPS=20; TRAIN_EVAL=20; fi
    "$PYTHON" -u -m "$MODULE.train_quota" "${COMMON[@]}" "${EXTRA[@]}" \
      --banks "$QUOTA_BANKS" --device "${DEVICE:-cuda:0}" --workers "${WORKERS:-4}" \
      --steps "$TRAIN_STEPS" --eval_every "$TRAIN_EVAL" --lr "${LR:-1e-4}" \
      --pool_mode "${POOL_MODE:-native}"
  fi
fi
echo "COMPLETE $MODE. No other stage was launched. Train/Val only."
