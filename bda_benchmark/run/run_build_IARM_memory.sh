#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
CKPT_DIR="${1:?Usage: run_build_IARM_memory.sh CKPT_DIR [CHECKPOINT_NAME]}"
CHECKPOINT_NAME="${2:-best_val_event_macro_target3.pth}"
OUTPUT="${CKPT_DIR}/train_building_memory.pt"
cd "${PROJECT_ROOT}"
export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${PROJECT_ROOT}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=0 python memory/extract_IARM_memory.py \
  --checkpoint "${CKPT_DIR}/${CHECKPOINT_NAME}" \
  --config_json "${CKPT_DIR}/config.json" \
  --dataset_path "${PROJECT_ROOT}/dataset" \
  --train_data_list_path "${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt" \
  --device cuda:0 --num_workers 4 \
  --min_component_area 16 --min_label_purity 0.60 \
  --max_candidates_per_event_class 2000 \
  --max_memory_per_event_class 500 \
  --output "${OUTPUT}" 2>&1 | tee "${CKPT_DIR}/build_memory.log"
python memory/analyze_IARM_memory.py \
  --memory "${OUTPUT}" --topk 5 \
  --output_json "${CKPT_DIR}/memory_analysis.json" \
  2>&1 | tee "${CKPT_DIR}/memory_analysis.log"
