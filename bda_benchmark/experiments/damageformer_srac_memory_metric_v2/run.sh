#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
MODULE=bda_benchmark.experiments.damageformer_srac_memory_metric_v2

case "$MODE" in
  preflight)
    "$PYTHON" -m unittest -v "$MODULE.test_sampling"
    "$PYTHON" -m "$MODULE.train" --help
    echo 'CPU synthetic contracts only; no Train/Val/Test images read.'
    exit 0 ;;
  train) ;;
  *) echo 'Use preflight or train' >&2; exit 2 ;;
esac

: "${JOINT_RUN:?Set the completed original J1/J2/J3 source directory}"
: "${A1_CKPT:?Set the exact original A1 checkpoint}"
: "${E0_PROTOCOL:?Set the passed legacy/native E0 protocol directory}"
: "${QUERY_RUN:?Set completed Train-only hardquery query cache directory}"
: "${METRIC_V2_OUTPUT:?Set a fresh output directory for matched projector fitting}"

ARGS=(
  --joint_run "$JOINT_RUN"
  --a1_checkpoint "$A1_CKPT"
  --protocol "$E0_PROTOCOL"
  --queries "$QUERY_RUN"
  --dataset "${DATASET:-$ROOT/bda_benchmark/dataset}"
  --train_list "${TRAIN_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/train_set.txt}"
  --val_list "${VAL_LIST:-$ROOT/bda_benchmark/dataset/splitname/standard_ML/val_set.txt}"
  --output "$METRIC_V2_OUTPUT"
  --device "${DEVICE:-cuda:0}"
  --workers "${WORKERS:-4}"
  --seed "${SEED:-3407}"
  --profile "${PROFILE:-legacy}"
  --pool_mode "${POOL_MODE:-native}"
)
if [[ -n "${FRONT_CKPT:-}" ]]; then ARGS+=(--checkpoint "$FRONT_CKPT"); fi
if [[ -n "${CACHE:-}" ]]; then ARGS+=(--cache "$CACHE"); fi
if [[ -n "${OLD_METRIC_CKPT:-}" ]]; then ARGS+=(--metric_checkpoint "$OLD_METRIC_CKPT"); fi
if [[ -n "${MATMUL_TF32:-}" ]]; then ARGS+=(--matmul_tf32 "$MATMUL_TF32"); fi
if [[ -n "${CUDNN_TF32:-}" ]]; then ARGS+=(--cudnn_tf32 "$CUDNN_TF32"); fi
if [[ -n "${DETERMINISTIC:-}" ]]; then ARGS+=(--deterministic "$DETERMINISTIC"); fi
if [[ "${METRIC_V2_SMOKE:-0}" == 1 ]]; then ARGS+=(--smoke --steps 20); fi
"$PYTHON" -u -m "$MODULE.train" "${ARGS[@]}"
