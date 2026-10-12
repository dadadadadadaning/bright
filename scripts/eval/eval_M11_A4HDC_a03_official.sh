#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/BRIGHT-master
export PYTHONPATH=/root/shared-nvme/BRIGHT-master:/root/shared-nvme/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

DATASET_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset"
TEST_LIST="/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/test_set.txt"
EXP_NAME="M11_A4HDC_a03_ROId5_C640_I120k_BS2"
EXP_PARENT="/root/shared-nvme/BRIGHT-master/saved_weights/BGCA_clean/${EXP_NAME}"
EXP_DIR="${EXP_DIR:-$(ls -td "${EXP_PARENT}"/DamageFormer_BGCA_* 2>/dev/null | head -n 1)}"
OUT_ROOT="/root/shared-nvme/BRIGHT-master/eval_results_official/${EXP_NAME}"

if [[ -z "${EXP_DIR}" || ! -d "${EXP_DIR}" ]]; then
  echo "Cannot find EXP_DIR. Set EXP_DIR=/path/to/experiment." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
for CKPT in best_balanced_score.pth best_damaged_iou.pth best_destroyed_iou.pth best_valid_miou.pth; do
  CKPT_PATH="${EXP_DIR}/${CKPT}"
  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "WARNING: skip missing checkpoint: ${CKPT_PATH}"
    continue
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python full_evaluate_unified.py \
    --eval_protocol official \
    --model_type DamageFormer_BGCA \
    --model_path "${CKPT_PATH}" \
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
    --use_hdc \
    --hdc_mode blend \
    --hdc_alpha 0.3 \
    --lambda_exist 0.1 \
    --lambda_severity 0.2 \
    --lambda_hdc 0.5 \
    --exist_loss_type bce
done

python scripts/eval/summarize_official_results.py --root /root/shared-nvme/BRIGHT-master/eval_results_official
