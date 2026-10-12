#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
REPO_ROOT="/root/shared-nvme/BRIGHT-master"
CKPT="/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth"
OUT_DIR="$(dirname "${CKPT}")"
LOCKED_SHA256="45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7"
DATASET_ROOT="${PROJECT_ROOT}/dataset"
TEST_LIST="${DATASET_ROOT}/splitname/standard_ML/test_set.txt"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "${LOCKED_SHA256}  ${CKPT}" | sha256sum --check -

python "${REPO_ROOT}/full_evaluate_unified.py" \
  --eval_protocol official \
  --model_type DamageFormer_S8_MICB_DSMIL \
  --model_path "${CKPT}" \
  --strict_checkpoint \
  --dataset_path "${DATASET_ROOT}" \
  --data_list_path "${TEST_LIST}" \
  --image_size 1024 \
  --batch_size 1 \
  --num_workers 8 \
  --save_dir "${OUT_DIR}/fixed_test_official" \
  2>&1 | tee "${OUT_DIR}/fixed_test.log"
