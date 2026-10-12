#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# B0: Original ChangeMamba-BRIGHT Full Training
# 原始 ChangeMamba-BRIGHT，不加 B1/B2/B3/B4 改进
# ============================================================

PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/ChangeMamba-master"
CHANGEDIR="${PROJECT_ROOT}/changedetection"

# BRIGHT 数据集路径
DATA_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset"

# 数据划分文件
LIST_ROOT="${CHANGEDIR}/datasets/data_name_list/BRIGHT"

# VMamba tiny 配置文件
CFG_PATH="${CHANGEDIR}/configs/vssm1/vssm_tiny_224_0229flex.yaml"

# VMamba 预训练权重
# 如果没有该文件，脚本会自动跳过预训练，从头训练
PRETRAIN="${PROJECT_ROOT}/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"

# 保存路径
SAVE_ROOT="${PROJECT_ROOT}/results/checkpoints_B0_full"

# 日志路径
LOG_ROOT="${PROJECT_ROOT}/logs"
mkdir -p "${SAVE_ROOT}"
mkdir -p "${LOG_ROOT}"

# Python 路径
export PYTHONPATH="${PROJECT_ROOT}:${CHANGEDIR}:${PYTHONPATH:-}"

# PyTorch 动态库路径，解决 libc10.so / torch_cuda.so 找不到的问题
export TORCH_LIB=$(python - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)

export LD_LIBRARY_PATH="${TORCH_LIB}:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "============================================================"
echo "B0 Original ChangeMamba-BRIGHT Full Training"
echo "============================================================"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "CHANGEDIR=${CHANGEDIR}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "LIST_ROOT=${LIST_ROOT}"
echo "CFG_PATH=${CFG_PATH}"
echo "PRETRAIN=${PRETRAIN}"
echo "SAVE_ROOT=${SAVE_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "TORCH_LIB=${TORCH_LIB}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "============================================================"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY

echo "============================================================"
echo "Checking required files and directories"
echo "============================================================"

test -f "${CHANGEDIR}/script/train_MambaBDA_bright.py"
test -f "${CFG_PATH}"

test -d "${DATA_ROOT}/pre-event"
test -d "${DATA_ROOT}/post-event"
test -d "${DATA_ROOT}/target"

test -f "${LIST_ROOT}/train_set.txt"
test -f "${LIST_ROOT}/val_set.txt"
test -f "${LIST_ROOT}/test_set.txt"

echo "[OK] file check passed"

echo "============================================================"
echo "Checking selective_scan CUDA extension"
echo "============================================================"

python - <<'PY'
mods = [
    "selective_scan_cuda_core",
    "selective_scan_cuda_ndstate",
    "selective_scan_cuda_oflex",
]
for m in mods:
    try:
        __import__(m)
        print("[OK]", m)
    except Exception as e:
        print("[MISSING]", m, "=>", repr(e))
        raise
PY

echo "============================================================"
echo "Start B0 full training"
echo "============================================================"

CMD=(
python "${CHANGEDIR}/script/train_MambaBDA_bright.py"
  --cfg "${CFG_PATH}"
  --dataset "BRIGHT"
  --model_type "ChangeMamba-MMBDA_B0_cat"
  --model_param_path "${SAVE_ROOT}"
  --train_dataset_path "${DATA_ROOT}"
  --train_data_list_path "${LIST_ROOT}/train_set.txt"
  --val_dataset_path "${DATA_ROOT}"
  --val_data_list_path "${LIST_ROOT}/val_set.txt"
  --test_dataset_path "${DATA_ROOT}"
  --test_data_list_path "${LIST_ROOT}/test_set.txt"
  --train_batch_size 2
  --num_workers 4
  --crop_size 512
  --learning_rate 1e-4
  --weight_decay 5e-3
  --max_iters 80000
)

if [ -f "${PRETRAIN}" ]; then
  echo "[INFO] Found pretrained weight: ${PRETRAIN}"
  CMD+=(--encoder_pretrained_path "${PRETRAIN}")
else
  echo "[WARN] Pretrained weight not found. Training will start from scratch."
fi

echo "============================================================"
echo "Running command:"
printf '%q ' "${CMD[@]}"
echo
echo "============================================================"

"${CMD[@]}"

echo "============================================================"
echo "B0 full training finished"
echo "============================================================"
