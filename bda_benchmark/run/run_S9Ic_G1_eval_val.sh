#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <checkpoint_step_1000.pth|checkpoint_step_2000.pth|checkpoint_step_3000.pth>" >&2
  exit 2
fi

ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
BASE="/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16"

S8_CKPT="${BASE}/best_val_miou.pth"
GEOM_CKPT="$1"
S9IB_JSON="${BASE}/S9Ib_ConservativeCenter/S9IB_FINAL_VALIDATION_LOCKED.json"
S9A2B_JSON="${BASE}/S9A2b_ConditionalWatershed/s9a2b_decision.json"
S9I_JSON="${BASE}/S9I_CenterBoundary/s9i_final_val.json"
STEP="$(python - "${GEOM_CKPT}" <<'PY'
import sys
import torch
x = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(x["step"]))
PY
)"
OUT_DIR="${BASE}/S9Ic_ReliableGeometry_G1"
OUT_JSON="${OUT_DIR}/s9ic_g1_step_${STEP}_final_val.json"
LOG_FILE="${OUT_DIR}/s9ic_g1_step_${STEP}_final_val.log"

for required in \
  "${S8_CKPT}" \
  "${GEOM_CKPT}" \
  "${S9IB_JSON}" \
  "${S9A2B_JSON}" \
  "${S9I_JSON}"
do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: missing required file: ${required}" >&2
    exit 3
  fi
done

cd "${ROOT}"
export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${ROOT}:${ROOT}/script/standard_ML:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${GPU_ID:-0}" \
python script/standard_ML/eval_S9Ic_G1_val.py \
  --s8_checkpoint "${S8_CKPT}" \
  --geometry_checkpoint "${GEOM_CKPT}" \
  --s9ib_json "${S9IB_JSON}" \
  --s9a2b_json "${S9A2B_JSON}" \
  --s9i_json "${S9I_JSON}" \
  --output_json "${OUT_JSON}" \
  --device cuda:0 \
  --num_workers "${NUM_WORKERS:-8}" \
  2>&1 | tee "${LOG_FILE}"

echo "Report: ${OUT_JSON}"
