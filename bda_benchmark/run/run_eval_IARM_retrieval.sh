#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark"
CKPT_DIR="${1:?Usage: run_eval_IARM_retrieval.sh CKPT_DIR [val|test] [standard|cross_event]}"
SPLIT="${2:-val}"
MODE="${3:-standard}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best_val_event_macro_target3.pth}"
if [[ "${SPLIT}" == "val" ]]; then LIST="${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt"; else LIST="${PROJECT_ROOT}/dataset/splitname/standard_ML/test_set.txt"; fi
EXTRA=(); if [[ "${MODE}" == "cross_event" ]]; then EXTRA+=(--cross_event_only); fi
OUTPUT="${CKPT_DIR}/retrieval_${SPLIT}_${MODE}.json"
cd "${PROJECT_ROOT}"
export PYTHONPATH="/root/shared-nvme/BRIGHT-master:${PROJECT_ROOT}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=0 python memory/eval_IARM_retrieval.py \
  --checkpoint "${CKPT_DIR}/${CHECKPOINT_NAME}" \
  --config_json "${CKPT_DIR}/config.json" \
  --memory "${CKPT_DIR}/train_building_memory.pt" \
  --dataset_path "${PROJECT_ROOT}/dataset" --data_list_path "${LIST}" --split_name "${SPLIT}" \
  --device cuda:0 --num_workers 4 \
  --building_threshold 0.5 --min_component_area 16 \
  --topk_per_class 8 --route_top_classes 2 \
  --cache_beta 8.0 --alpha 1.0 --prototype_weight 0.10 \
  --similarity_threshold 0.55 --similarity_scale 0.05 --gate_threshold 0.15 \
  "${EXTRA[@]}" --output_json "${OUTPUT}" 2>&1 | tee "${OUTPUT%.json}.log"
