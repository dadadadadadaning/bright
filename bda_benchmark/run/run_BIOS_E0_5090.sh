#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
H6_A2_CKPT="${H6_A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
FEATURE_LOCK_CKPT="${FEATURE_LOCK_CKPT:-${Q0_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/Q0_SAR_READOUT/Q0_B0_screen1500_seed3407/best_readout.pth}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/BIOS}"
PREREGISTRATION="${PREREGISTRATION:-${PROJECT_ROOT}/configs/bios_e0_preregistered.json}"
SEED="${SEED:-3407}"

case "${MODE}" in
  preflight)
    bash -n run/run_BIOS_E0_5090.sh
    python -m py_compile \
      script/standard_ML/bios_e0_runtime.py \
      script/standard_ML/audit_BIOS_E0_relation_evidence.py
    python -c 'import json; json.load(open("configs/bios_e0_preregistered.json", encoding="utf-8"))'
    python -m unittest -v tests.test_bios_e0
    echo "BIOS E0 preflight passed."
    exit 0
    ;;
  smoke)
    MAX_IMAGES="${MAX_IMAGES:-20}"
    NUM_WORKERS="${NUM_WORKERS:-2}"
    TAG="smoke${MAX_IMAGES}"
    ;;
  val)
    MAX_IMAGES=0
    NUM_WORKERS="${NUM_WORKERS:-8}"
    TAG="val"
    ;;
  eval_test|test)
    echo "BIOS E0 is validation-only; test access is forbidden." >&2
    exit 2
    ;;
  *)
    echo "Usage: $0 preflight|smoke|val" >&2
    exit 2
    ;;
esac

for required in "${VAL_LIST}" "${SAR_STATS_JSON}" "${P3_CKPT}" "${H6_A2_CKPT}" "${FEATURE_LOCK_CKPT}" "${PREREGISTRATION}"; do
  test -s "${required}" || { echo "Missing BIOS E0 input: ${required}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_ROOT}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_ROOT}/BIOS_E0_relation_evidence_${TAG}_seed${SEED}.json}"

echo "================================================================================================"
echo "Protocol       : BIOS E0 validation-only parameter-free relation evidence audit"
echo "P3             : ${P3_CKPT}"
echo "H6-A2          : ${H6_A2_CKPT}"
echo "Feature lock    : ${FEATURE_LOCK_CKPT} (Q0 head probabilities are not used)"
echo "Preregistration : ${PREREGISTRATION}"
echo "Supports        : GT diagnostic ROI + frozen-P3 connected regions"
echo "Primary         : Stage3 local cosine-relation mean / 7x7"
echo "Secondary       : Stage4/MS/global/SD/heterogeneity diagnostics only"
echo "Signals        : local/global/LG x SIM/SD x mean/std/max/Q75/Q90/top20/area/concentration"
echo "Stress         : SAR-token shifts 1 2 3 + spatial shuffle"
echo "Training       : NONE"
echo "Test accessed  : NEVER"
echo "Output         : ${OUTPUT_JSON}"
echo "================================================================================================"

python -u script/standard_ML/audit_BIOS_E0_relation_evidence.py \
  --p3_checkpoint "${P3_CKPT}" \
  --a2_checkpoint "${H6_A2_CKPT}" \
  --feature_lock_checkpoint "${FEATURE_LOCK_CKPT}" \
  --dataset_path "${DATASET_ROOT}" \
  --val_list "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" \
  --output_json "${OUTPUT_JSON}" \
  --preregistration "${PREREGISTRATION}" \
  --device cuda:0 \
  --crop_size 640 \
  --batch_size "${EVAL_BATCH_SIZE:-1}" \
  --num_workers "${NUM_WORKERS}" \
  --supports gt p3 \
  --grid_size 7 \
  --minimum_area 16 \
  --proposal_threshold 0.5 \
  --minimum_building_fraction 0.5 \
  --minimum_class_purity 0.75 \
  --minimum_token_mass 0.25 \
  --minimum_valid_tokens 4 \
  --neighbour_count 4 \
  --relation_threshold 0.25 \
  --maximum_gt_p3_auc_gap 0.10 \
  --shift_magnitudes 1 2 3 \
  --max_images "${MAX_IMAGES}" \
  --seed "${SEED}"
