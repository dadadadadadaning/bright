#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
BASE="/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16"

S8_CKPT="${BASE}/best_val_miou.pth"
HEAD_CKPT="${BASE}/S9I_CenterBoundary/best_val_instance_f1.pth"
S9A_JSON="${BASE}/S9A_PredictedInstance/s9a_predicted_instance_val.json"
S9A2B_JSON="${BASE}/S9A2b_ConditionalWatershed/s9a2b_decision.json"
S9I_JSON="${BASE}/S9I_CenterBoundary/s9i_final_val.json"

OUT_DIR="${BASE}/S9Ib_ConservativeCenter"
OUT_JSON="${OUT_DIR}/s9ib_final_val.json"
LOG_FILE="${OUT_DIR}/s9ib_final_val.log"

for required in \
  "${S8_CKPT}" \
  "${HEAD_CKPT}" \
  "${S9A_JSON}" \
  "${S9A2B_JSON}" \
  "${S9I_JSON}"
do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: required file not found: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${OUT_DIR}"
cd "${ROOT}"

export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${ROOT}:${ROOT}/script/standard_ML:${PYTHONPATH:-}"

echo "============================================================"
echo "Experiment : S9-Ib fixed conservative learned-center"
echo "Split      : Validation only"
echo "Test       : NOT ACCESSED"
echo "Head       : ${HEAD_CKPT}"
echo "Output     : ${OUT_JSON}"
echo "Config     : center=.50 area=512 center_area=4 distance=10 markers=4"
echo "============================================================"

CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" \
python script/standard_ML/eval_S9Ib_conservative_center.py \
  --s8_checkpoint "${S8_CKPT}" \
  --geometry_checkpoint "${HEAD_CKPT}" \
  --s9a_json "${S9A_JSON}" \
  --s9a2b_json "${S9A2B_JSON}" \
  --s9i_json "${S9I_JSON}" \
  --output_json "${OUT_JSON}" \
  --device cuda:0 \
  --num_workers "${NUM_WORKERS:-8}" \
  2>&1 | tee "${LOG_FILE}"

echo
echo "S9-Ib completed."
echo "Report: ${OUT_JSON}"
