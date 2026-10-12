#!/bin/bash
set -e

cd /root/shared-nvme/BRIGHT-master

export PYTHONPATH=/root/shared-nvme/BRIGHT-master:/root/shared-nvme/BRIGHT-master/bda_benchmark:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT_DIR="/root/shared-nvme/BRIGHT-master/saved_weights/BGCA_final/M06_BGCA_dilatedROI_align_noOrdinal/BRIGHT/DamageFormer_BGCA_custom_dilated_soft_alpha0.1_d5_aligndamage_aware_lambda0.001_ce1.0_lovasz1.0_noordinal_20260604_215250"

DATASET_PATH="/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset"
DATA_LIST_PATH="/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/test_set.txt"

OUT_ROOT="/root/shared-nvme/BRIGHT-master/eval_results/BGCA_final/M06_BGCA_dilatedROI_align_noOrdinal_checkpoints"

mkdir -p "${OUT_ROOT}"

# 自动选择评估脚本路径
if [ -f "tools/full_evaluate_BGCA_Method05_06.py" ]; then
    EVAL_SCRIPT="tools/full_evaluate_BGCA_Method05_06.py"
elif [ -f "full_evaluate_BGCA_Method05_06.py" ]; then
    EVAL_SCRIPT="full_evaluate_BGCA_Method05_06.py"
elif [ -f "tools/full_evaluate_BGCA.py" ]; then
    EVAL_SCRIPT="tools/full_evaluate_BGCA.py"
else
    EVAL_SCRIPT="full_evaluate_BGCA.py"
fi

echo "Using eval script: ${EVAL_SCRIPT}"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "Output root: ${OUT_ROOT}"

CKPTS=(
  "iter_5000.pth"
  "iter_10000.pth"
  "iter_15000.pth"
  "iter_20000.pth"
  "iter_25000.pth"
  "iter_30000.pth"
  "iter_35000.pth"
  "best_model.pth"
)

for CKPT in "${CKPTS[@]}"
do
    MODEL_PATH="${CKPT_DIR}/${CKPT}"

    if [ ! -f "${MODEL_PATH}" ]; then
        echo "Skip missing checkpoint: ${MODEL_PATH}"
        continue
    fi

    CKPT_NAME="${CKPT%.pth}"
    SAVE_DIR="${OUT_ROOT}/${CKPT_NAME}"

    mkdir -p "${SAVE_DIR}"

    echo "============================================================"
    echo "Evaluating ${CKPT_NAME}"
    echo "Model: ${MODEL_PATH}"
    echo "Save : ${SAVE_DIR}"
    echo "============================================================"

    python "${EVAL_SCRIPT}" \
      --model_type "DamageFormer_BGCA" \
      --model_path "${MODEL_PATH}" \
      --dataset_path "${DATASET_PATH}" \
      --data_list_path "${DATA_LIST_PATH}" \
      --image_size 512 \
      --batch_size 2 \
      --num_workers 0 \
      --roi_mode "dilated_soft" \
      --roi_alpha 0.1 \
      --roi_dilate 5 \
      --use_align \
      --save_dir "${SAVE_DIR}" \
      > "${SAVE_DIR}/eval.log" 2>&1

    echo "Finished ${CKPT_NAME}"
    tail -n 30 "${SAVE_DIR}/eval.log"
done

echo "All M06 checkpoints evaluated."