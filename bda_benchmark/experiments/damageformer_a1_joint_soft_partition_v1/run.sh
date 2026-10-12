#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-preflight}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_PATH}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_PATH}/splitname/standard_ML/val_set.txt}"
A1_INIT_CKPT="${A1_INIT_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a1_long35000_seed3407/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_A1_JOINT_SOFT_PARTITION_V1}"
SEED="${SEED:-3407}"
stage(){ case "$1" in smoke) echo 'smoke20 20 10 8';; train) echo 'joint20000 20000 500 0';; *) exit 2;; esac; }
preflight(){
  test -s experiments/damageformer_a1_cross_modal_superpixel_v1/model.py
  test -s "${A1_INIT_CKPT}"
  test -s "${TRAIN_LIST}"; test -s "${VAL_LIST}"
  python -m py_compile experiments/damageformer_a1_joint_soft_partition_v1/model.py experiments/damageformer_a1_joint_soft_partition_v1/train.py
  python - <<'PY'
import torch
print('torch=',torch.__version__,'cuda=',torch.cuda.is_available(),'gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
print('A1 initialized from best checkpoint: YES')
print('A1 frozen: NO')
print('Partition trainable: YES')
print('Joint steps: 20000')
print('Validation every: 500')
print('Test: FORBIDDEN')
PY
  echo 'PREFLIGHT PASSED'
}
train_one(){ local ph="$1" tag steps ev vmax out; read -r tag steps ev vmax <<< "$(stage "$ph")"; out="${OUTPUT_ROOT}/a1_joint_partition_${tag}_seed${SEED}"; mkdir -p "$out"; local -a resume=(); [[ -s "$out/latest.pth" ]] && resume+=(--resume "$out/latest.pth"); python -u -m bda_benchmark.experiments.damageformer_a1_joint_soft_partition_v1.train --dataset_path "$DATASET_PATH" --train_list "$TRAIN_LIST" --val_list "$VAL_LIST" --a1_init_ckpt "$A1_INIT_CKPT" --output_dir "$out" --optimizer_steps "$steps" --eval_every "$ev" --save_every "$ev" --val_max_batches "$vmax" --batch_size "${BATCH_SIZE:-2}" --eval_batch_size "${EVAL_BATCH_SIZE:-2}" --accumulation_steps "${ACCUMULATION_STEPS:-8}" --num_workers "${NUM_WORKERS:-8}" --a1_lr "${A1_LR:-2e-5}" --plugin_lr "${PLUGIN_LR:-1e-4}" --weight_decay "${WEIGHT_DECAY:-5e-3}" --region_stride "${REGION_STRIDE:-8}" --device "${DEVICE:-cuda:0}" --amp_dtype "${AMP_DTYPE:-bf16}" --seed "$SEED" "${resume[@]}"; }
case "$MODE" in preflight) preflight;; smoke) preflight; train_one smoke;; train) preflight; train_one train;; status) pgrep -af 'damageformer_a1_joint_soft_partition_v1.train' || true;; *) echo 'Usage: run.sh {preflight|smoke|train|status}' >&2; exit 2;; esac
