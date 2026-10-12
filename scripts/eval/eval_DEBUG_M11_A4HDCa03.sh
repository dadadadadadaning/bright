#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Evaluate DEBUG_M11_A4HDCa03 checkpoints
# BRIGHT: pre-disaster RGB + post-disaster SAR building damage assessment
# ============================================================

cd /root/shared-nvme/BRIGHT-master

# 如果你的环境名不是 bright_venv/bright，请手动先执行：
# conda activate bright
# 或 source /path/to/bright_venv/bin/activate

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PROJECT_ROOT=/root/shared-nvme/BRIGHT-master
DATA_ROOT=${PROJECT_ROOT}/bda_benchmark/dataset

# 默认评估 test_set；如果你只想看验证集，把 test_set.txt 改成 val_set.txt
DATA_LIST=${DATA_ROOT}/splitname/standard_ML/test_set.txt

# 推荐统一使用你现在扩展后的 official 评估脚本
# 如果你的脚本不在根目录，请把这里改成真实路径
EVAL_PY=${PROJECT_ROOT}/full_evaluate_unified.py

# 如果暂时没有 full_evaluate_unified.py，可以临时改成：
# EVAL_PY=${PROJECT_ROOT}/full_evaluate_BGCA.py

EXP_DIR=/root/shared-nvme/BRIGHT-master/saved_weights/BGCA_clean/DEBUG_M11_A4HDCa03/DamageFormer_BGCA_custom_dilated_soft_multiplicative_beta1.0_alpha0.1_d5_noalign_nodiff_mddm_base_gamma0.0_roi1_noirc_nodg_exist0.1_bce_g2.0_hdc_blend_a0.3_lh0.5_ls0.2_ce1.0_lovasz1.0_noordinal_20260623_210045

OUT_ROOT=${PROJECT_ROOT}/eval_results_official/DEBUG_M11_A4HDCa03_20260623_210045
mkdir -p "${OUT_ROOT}"

CKPTS=(
  "best_valid_miou.pth"
  "best_damaged_iou.pth"
  "best_destroyed_iou.pth"
  "best_balanced_score.pth"
  "latest_resume.pth"
)

echo "============================================================"
echo "Project root : ${PROJECT_ROOT}"
echo "Dataset root : ${DATA_ROOT}"
echo "Data list    : ${DATA_LIST}"
echo "Eval script  : ${EVAL_PY}"
echo "Exp dir      : ${EXP_DIR}"
echo "Output root  : ${OUT_ROOT}"
echo "============================================================"

if [ ! -f "${EVAL_PY}" ]; then
  echo "[ERROR] Cannot find evaluation script: ${EVAL_PY}"
  echo "Please check whether full_evaluate_unified.py exists."
  exit 1
fi

for CKPT_NAME in "${CKPTS[@]}"; do
  CKPT_PATH="${EXP_DIR}/${CKPT_NAME}"

  if [ ! -f "${CKPT_PATH}" ]; then
    echo "[SKIP] Missing checkpoint: ${CKPT_PATH}"
    continue
  fi

  TAG="${CKPT_NAME%.pth}"
  SAVE_DIR="${OUT_ROOT}/${TAG}"
  mkdir -p "${SAVE_DIR}"

  echo ""
  echo "============================================================"
  echo "[EVAL] ${TAG}"
  echo "Checkpoint: ${CKPT_PATH}"
  echo "Save dir  : ${SAVE_DIR}"
  echo "============================================================"

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python "${EVAL_PY}" \
    --eval_protocol official \
    --model_type DamageFormer_BGCA \
    --model_path "${CKPT_PATH}" \
    --dataset_path "${DATA_ROOT}" \
    --data_list_path "${DATA_LIST}" \
    --save_dir "${SAVE_DIR}" \
    --crop_size 640 \
    --eval_batch_size 1 \
    --num_workers 0 \
    \
    --roi_mode dilated_soft \
    --roi_fusion multiplicative \
    --roi_beta 1.0 \
    --roi_alpha 0.1 \
    --roi_dilate 5 \
    \
    --use_mddm \
    --mddm_variant base \
    --mddm_gamma_init 0.0 \
    --mddm_use_roi \
    \
    --use_hdc \
    --hdc_mode blend \
    --hdc_alpha 0.3 \
    --lambda_exist 0.1 \
    --lambda_hdc 0.5 \
    --lambda_severity 0.2 \
    --exist_loss_type bce \
    --exist_focal_gamma 2.0 \
    \
    --ce_weight 1.0 \
    --lovasz_weight 1.0 \
    2>&1 | tee "${SAVE_DIR}/eval_${TAG}.log"

done

echo ""
echo "============================================================"
echo "All evaluations finished."
echo "Results saved to: ${OUT_ROOT}"
echo "============================================================"
