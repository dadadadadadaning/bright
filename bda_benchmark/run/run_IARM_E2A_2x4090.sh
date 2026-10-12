#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
OUTPUT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/IARM_E2A"
LOG_ROOT="/root/shared-nvme/BRIGHT-train-logs/IARM_E2A"

GPU_IDS="${GPU_IDS:-0,1}"
MASTER_PORT="${MASTER_PORT:-29621}"
RESIDUAL_STAGES="${RESIDUAL_STAGES:-4}"

# E2-A is a diagnostic experiment. Start with 10k optimizer steps.
# Extend to 50k only after Validation Exist F1 has clearly improved.
OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-10000}"
EFFECTIVE_BATCH=16
MAX_ITERS=$((OPTIMIZER_STEPS * EFFECTIVE_BATCH))

MICRO_BATCH="${MICRO_BATCH:-4}"
ACCUM_STEPS="${ACCUM_STEPS:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MIN_FREE_GB="${MIN_FREE_GB:-8}"

EXIST_WEIGHT_INTACT="${EXIST_WEIGHT_INTACT:-1.0}"
EXIST_WEIGHT_NON_INTACT="${EXIST_WEIGHT_NON_INTACT:-2.5}"

INIT_CHECKPOINT="${INIT_CHECKPOINT:-/root/shared-nvme/BRIGHT-checkpoints/IARM/BRIGHT/IARM_E1_CMU_NORMAL_Stage4_2x4090_EBS16_50k_E1_CMU_NORMAL_20260714_223753/best_val_miou.pth}"
RESUME="${RESUME:-}"

if (( MICRO_BATCH * 2 * ACCUM_STEPS != EFFECTIVE_BATCH )); then
  echo "ERROR: MICRO_BATCH * 2 GPUs * ACCUM_STEPS must equal 16."
  echo "Current: ${MICRO_BATCH} * 2 * ${ACCUM_STEPS}"
  exit 2
fi

if [[ -z "${RESUME}" && ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "ERROR: E1 initialization checkpoint not found:"
  echo "  ${INIT_CHECKPOINT}"
  exit 3
fi

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python - "${OUTPUT_ROOT}" "${MIN_FREE_GB}" <<'PY'
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1])
required = float(sys.argv[2])
free = shutil.disk_usage(root).free / 1024**3
print(f"Checkpoint filesystem free space: {free:.2f} GB")
if free < required:
    raise SystemExit(
        f"Insufficient checkpoint space: {free:.2f} GB < {required:.2f} GB"
    )
PY

TS="$(date +%Y%m%d_%H%M%S)"
TAG="IARM_E2A_ExistOnly_Stage${RESIDUAL_STAGES//,/}_2x4090_EBS16_${OPTIMIZER_STEPS}steps"
LOG_FILE="${LOG_ROOT}/${TAG}_${TS}.log"

EXTRA_ARGS=()
if [[ -n "${RESUME}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME}")
else
  EXTRA_ARGS+=(--init_checkpoint "${INIT_CHECKPOINT}")
fi

echo "============================================================"
echo "Experiment          : E2A_EXIST_ONLY"
echo "GPUs                : ${GPU_IDS}"
echo "Residual stages     : ${RESIDUAL_STAGES}"
echo "Residual fusion     : OFF"
echo "Main branch         : FROZEN"
echo "Micro batch / GPU   : ${MICRO_BATCH}"
echo "Accumulation        : ${ACCUM_STEPS}"
echo "Effective batch     : ${EFFECTIVE_BATCH}"
echo "Optimizer steps     : ${OPTIMIZER_STEPS}"
echo "Training crops      : ${MAX_ITERS}"
echo "Exist weights       : ${EXIST_WEIGHT_INTACT} / ${EXIST_WEIGHT_NON_INTACT}"
echo "E1 initialization   : ${INIT_CHECKPOINT}"
echo "Output root         : ${OUTPUT_ROOT}"
echo "Log                 : ${LOG_FILE}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" torchrun \
  --nproc_per_node=2 \
  --master_port="${MASTER_PORT}" \
  script/standard_ML/train_DamageFormer_IARM_E2A_ddp.py \
  --dataset BRIGHT \
  --train_dataset_path "${PROJECT_ROOT}/dataset" \
  --train_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt" \
  --val_dataset_path "${PROJECT_ROOT}/dataset" \
  --val_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt" \
  --test_dataset_path "${PROJECT_ROOT}/dataset" \
  --test_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/test_set.txt" \
  --experiment E2A_EXIST_ONLY \
  --residual_stages "${RESIDUAL_STAGES}" \
  --residual_dim 128 \
  --flow_hidden_dim 64 \
  --max_offset_pixels 2.0 \
  --detach_backbone_for_residual \
  --residual_gamma_max 0.25 \
  --roi_alpha 0.1 \
  --normal_weight 0.05 \
  --cmu_weight 0.02 \
  --flow_smooth_weight 0.005 \
  --flow_magnitude_weight 0.001 \
  --damage_exist_weight 0.20 \
  --anchor_weight 0.0 \
  --cross_event_contrast_weight 0.0 \
  --exist_weight_intact "${EXIST_WEIGHT_INTACT}" \
  --exist_weight_non_intact "${EXIST_WEIGHT_NON_INTACT}" \
  --grad_clip_norm 5.0 \
  --gaussian_step_momentum 0.99 \
  --gaussian_variance_floor 0.002 \
  --max_pixels_per_image_class 256 \
  --cmu_temperature 0.1 \
  --cmu_max_samples 256 \
  --train_batch_size_per_gpu "${MICRO_BATCH}" \
  --eval_batch_size 1 \
  --accumulation_steps "${ACCUM_STEPS}" \
  --optimizer_steps "${OPTIMIZER_STEPS}" \
  --max_iters "${MAX_ITERS}" \
  --crop_size 640 \
  --learning_rate 1e-4 \
  --weight_decay 5e-3 \
  --eval_every 500 \
  --log_every 10 \
  --num_workers "${NUM_WORKERS}" \
  --seed 3407 \
  --model_type "${TAG}" \
  --model_param_path "${OUTPUT_ROOT}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"
