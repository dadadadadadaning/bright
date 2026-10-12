#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
CKPT="/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth"
S9A_JSON="$(dirname "${CKPT}")/S9A_PredictedInstance/s9a_predicted_instance_val.json"
OUT_DIR="$(dirname "${CKPT}")/S9A2_ProbabilitySeed_Watershed"

if [[ ! -f "${S9A_JSON}" ]]; then
  echo "ERROR: locked S9-A result not found:"
  echo "  ${S9A_JSON}"
  exit 2
fi

mkdir -p "${OUT_DIR}"
cd "${PROJECT_ROOT}"

export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${PROJECT_ROOT}:${PROJECT_ROOT}/script/standard_ML:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" python \
  script/standard_ML/eval_S9A2_probability_watershed.py \
  --checkpoint "${CKPT}" \
  --s9a_json "${S9A_JSON}" \
  --output_dir "${OUT_DIR}" \
  --device cuda:0 \
  --num_workers "${NUM_WORKERS:-8}" \
  --matching greedy \
  --match_iou 0.5 \
  --merge_split_overlap 0.10 \
  --expected_parent_miou 69.76121287610887 \
  --target_instance_recall 65.0 \
  --target_instance_f1 66.0 \
  --min_mean_matched_iou 0.75 \
  --min_bag_macro_f1 72.0 \
  --min_damaged_recall 60.0 \
  --min_destroyed_recall 70.0 \
  2>&1 | tee "${OUT_DIR}/s9a2_validation.log"
