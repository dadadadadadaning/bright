#!/usr/bin/env bash

set -e

# 当前脚本所在目录：
# /root/shared-nvme/BRIGHT-master/bda_benchmark/run
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载通用环境变量
source "${RUN_DIR}/common_env.sh"

# 注意：必须在 source common_env.sh 之后重新确定路径，
# 防止 common_env.sh 覆盖 PROJECT_ROOT。
BDA_ROOT="$(cd "${RUN_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BDA_ROOT}/.." && pwd)"

TRAIN_SCRIPT="${BDA_ROOT}/script/standard_ML/train_Priority_Modular.py"

CHECKPOINT_ROOT="/root/shared-nvme/BRIGHT-checkpoints/PriorityModules/BRIGHT"
EXP_NAME="B0A_EBS16_25k"
OUTDIR="${CHECKPOINT_ROOT}/${EXP_NAME}"
OPTIMIZER_STEPS=25000

# 兼容项目内部的两种 import：
# import dataset...
# import bda_benchmark.dataset...
export PYTHONPATH="${REPO_ROOT}:${BDA_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTDIR}"

# 启动前检查训练脚本
if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: training script not found:"
    echo "  ${TRAIN_SCRIPT}"
    exit 1
fi

echo "============================================================"
echo "Stage             : B0A"
echo "Experiment        : ${EXP_NAME}"
echo "Optimizer steps   : ${OPTIMIZER_STEPS}"
echo "Repository root   : ${REPO_ROOT}"
echo "BDA root          : ${BDA_ROOT}"
echo "Training script   : ${TRAIN_SCRIPT}"
echo "Checkpoint root   : ${CHECKPOINT_ROOT}"
echo "Output directory  : ${OUTDIR}"
echo "CUDA device       : 0"
echo "PYTHONPATH        : ${PYTHONPATH}"
echo "============================================================"

cd "${BDA_ROOT}"

CUDA_VISIBLE_DEVICES=0 python -u \
    "${TRAIN_SCRIPT}" \
    "${BASE_ARGS[@]}" \
    "$@" \
    --stage B0A \
    --exp_name "${EXP_NAME}" \
    --output_root "${CHECKPOINT_ROOT}" \
    --optimizer_steps "${OPTIMIZER_STEPS}"
