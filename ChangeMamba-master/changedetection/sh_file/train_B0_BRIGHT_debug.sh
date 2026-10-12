#!/usr/bin/env bash
set -euo pipefail

# =========================
# B0: Original ChangeMamba-BRIGHT Debug Training
# =========================

PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/ChangeMamba-master"
CHANGEDIR="${PROJECT_ROOT}/changedetection"

# 这里是 BRIGHT 数据集根目录，下面必须有 pre-event / post-event / target
# 如果你的数据不在这个位置，先改这里
DATA_ROOT="/root/shared-nvme/BRIGHT-master/datasets/BRIGHT"

# 如果上面的 DATA_ROOT 不存在，自动尝试搜索
if [ ! -d "${DATA_ROOT}/pre-event" ]; then
  echo "[WARN] DATA_ROOT=${DATA_ROOT} not found or missing pre-event."
  echo "[INFO] Searching BRIGHT data root under /root/shared-nvme/BRIGHT-master ..."
  PRE_DIR=$(find /root/shared-nvme/BRIGHT-master -maxdepth 6 -type d -name "pre-event" | head -n 1 || true)
  if [ -n "${PRE_DIR}" ]; then
    DATA_ROOT=$(dirname "${PRE_DIR}")
    echo "[INFO] Auto detected DATA_ROOT=${DATA_ROOT}"
  else
    echo "[ERROR] Cannot find pre-event directory. Please set DATA_ROOT manually."
    exit 1
  fi
fi

LIST_ROOT="${CHANGEDIR}/datasets/data_name_list/BRIGHT"
CFG_PATH="${CHANGEDIR}/configs/vssm1/vssm_tiny_224_0229flex.yaml"

# VMamba 预训练权重。如果不存在，脚本会自动跳过预训练参数
PRETRAIN="${PROJECT_ROOT}/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"

SAVE_ROOT="${PROJECT_ROOT}/results/checkpoints_B0_debug"

export PYTHONPATH="${PROJECT_ROOT}:${CHANGEDIR}:${PYTHONPATH:-}"

export TORCH_LIB=$(python - <<'PYTORCH'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PYTORCH
)

export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"

echo "TORCH_LIB=${TORCH_LIB}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"


echo "========== B0 DEBUG TRAINING =========="
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHANGEDIR=${CHANGEDIR}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "LIST_ROOT=${LIST_ROOT}"
echo "CFG_PATH=${CFG_PATH}"
echo "PRETRAIN=${PRETRAIN}"
echo "SAVE_ROOT=${SAVE_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "========== Checking files =========="
test -f "${CHANGEDIR}/script/train_MambaBDA_bright.py"
test -f "${CFG_PATH}"
test -d "${DATA_ROOT}/pre-event"
test -d "${DATA_ROOT}/post-event"
test -d "${DATA_ROOT}/target"
test -f "${LIST_ROOT}/train_set.txt"
test -f "${LIST_ROOT}/val_set.txt"
test -f "${LIST_ROOT}/test_set.txt"

mkdir -p "${SAVE_ROOT}"

CMD=(
python "${CHANGEDIR}/script/train_MambaBDA_bright.py"
  --cfg "${CFG_PATH}"
  --dataset "BRIGHT"
  --model_type "ChangeMamba-MMBDA_B0_debug"
  --model_param_path "${SAVE_ROOT}"
  --train_dataset_path "${DATA_ROOT}"
  --train_data_list_path "${LIST_ROOT}/train_set.txt"
  --val_dataset_path "${DATA_ROOT}"
  --val_data_list_path "${LIST_ROOT}/val_set.txt"
  --test_dataset_path "${DATA_ROOT}"
  --test_data_list_path "${LIST_ROOT}/test_set.txt"
  --train_batch_size 2
  --num_workers 0
  --crop_size 512
  --learning_rate 1e-4
  --weight_decay 5e-3
  --max_iters 20
)

if [ -f "${PRETRAIN}" ]; then
  echo "[INFO] Found pretrained weight: ${PRETRAIN}"
  CMD+=(--encoder_pretrained_path "${PRETRAIN}")
else
  echo "[WARN] Pretrained weight not found, debug training will run without pretrained weight."
fi

echo "========== Running command =========="
printf '%q ' "${CMD[@]}"
echo
"${CMD[@]}"
