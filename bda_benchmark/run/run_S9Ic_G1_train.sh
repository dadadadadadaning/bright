#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${RUN_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/.." && pwd)"
BASE="${S9_BASE:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16}"

TRAINER="${ROOT}/script/standard_ML/train_S9_instance_geometry.py"
S8_CKPT="${S8_CKPT:-${BASE}/best_val_miou.pth}"
S9IB_JSON="${S9IB_JSON:-${BASE}/S9Ib_ConservativeCenter/S9IB_FINAL_VALIDATION_LOCKED.json}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
OUT_DIR="${OUT_DIR:-${BASE}/S9Ic_ReliableGeometry_G1_EDT_v2}"
LOG_FILE="${OUT_DIR}/train_S9Ic_G1.log"

for required in \
  "${TRAINER}" \
  "${S8_CKPT}" \
  "${S9IB_JSON}" \
  "${TRAIN_LIST}" \
  "${VAL_LIST}"
do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: missing required file: ${required}" >&2
    if [[ "${required}" == "${TRAINER}" ]]; then
      echo "Upload the complete local bda_benchmark folder before training." >&2
    fi
    exit 2
  fi
done
if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "ERROR: missing dataset directory: ${DATASET_ROOT}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
cd "${ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${ROOT}:${ROOT}/script/standard_ML:${PYTHONPATH:-}"

echo "============================================================"
echo "Experiment       : S9-Ic-G1"
echo "S8               : FROZEN"
echo "Feature          : fused_roi / 256 channels"
echo "Center target    : source-scale deterministic EDT Gaussian / pseudo-instance"
echo "Center loss      : CenterNet-style focal"
echo "Center decoder   : native 3x3 local-maximum NMS"
echo "Boundary Head    : TRAINED"
echo "Boundary Gate    : DISABLED in G1"
echo "Rollback         : DISABLED in G1"
echo "Optimizer steps  : 3000"
echo "Validation       : every 1000 steps"
echo "Test             : NOT ACCESSED"
echo "Dataset          : ${DATASET_ROOT}"
echo "Train list       : ${TRAIN_LIST}"
echo "Validation list  : ${VAL_LIST}"
echo "Output           : ${OUT_DIR}"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" \
python -u "${TRAINER}" \
  --variant g1 \
  --checkpoint "${S8_CKPT}" \
  --s9ib_json "${S9IB_JSON}" \
  --output_dir "${OUT_DIR}" \
  --train_dataset_path "${DATASET_ROOT}" \
  --train_data_list_path "${TRAIN_LIST}" \
  --val_dataset_path "${DATASET_ROOT}" \
  --val_data_list_path "${VAL_LIST}" \
  --device cuda:0 \
  --train_batch_size 4 \
  --accumulation_steps 4 \
  --optimizer_steps 3000 \
  --max_iters 48000 \
  --crop_size 640 \
  --eval_every 1000 \
  --log_every 20 \
  --num_workers "${NUM_WORKERS:-8}" \
  --learning_rate 1e-3 \
  --weight_decay 1e-4 \
  --hidden_channels 64 \
  --amp bf16 \
  --seed 3407 \
  2>&1 | tee "${LOG_FILE}"
