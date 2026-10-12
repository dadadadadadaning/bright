#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/BRIGHT-master
export PYTHONPATH=/root/shared-nvme/BRIGHT-master:/root/shared-nvme/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

DATASET_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset"
TEST_LIST="/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/test_set.txt"
EXP_PARENT="/root/shared-nvme/BRIGHT-master/saved_weights/BGCA_final/A4_ExistHead_dilatedROI_crop640_iter120k_bs2"
EXP_DIR="${EXP_DIR:-$(ls -td "${EXP_PARENT}"/DamageFormer_BGCA_* 2>/dev/null | head -n 1)}"
OUT_ROOT="/root/shared-nvme/BRIGHT-master/eval_results/unified/A4_ExistHead_120k"

if [[ -z "${EXP_DIR}" || ! -d "${EXP_DIR}" ]]; then
  echo "Cannot find EXP_DIR. Set EXP_DIR=/path/to/experiment." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
for CKPT in best_balanced_score.pth best_valid_miou.pth best_damaged_iou.pth best_destroyed_iou.pth; do
  [[ -f "${EXP_DIR}/${CKPT}" ]] || { echo "Skip missing ${CKPT}"; continue; }
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python full_evaluate_unified.py \
    --eval_protocol official \
    --model_type DamageFormer_BGCA \
    --model_path "${EXP_DIR}/${CKPT}" \
    --dataset_path "${DATASET_ROOT}" \
    --data_list_path "${TEST_LIST}" \
    --image_size 640 \
    --batch_size 1 \
    --num_workers 2 \
    --save_dir "${OUT_ROOT}/${CKPT%.pth}" \
    --roi_mode dilated_soft \
    --roi_fusion multiplicative \
    --roi_beta 1.0 \
    --roi_alpha 0.1 \
    --roi_dilate 5 \
    --use_mddm \
    --mddm_variant base \
    --mddm_gamma_init 0.0 \
    --mddm_use_roi \
    --use_damage_exist_head \
    --lambda_exist 0.1
done
