#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
H6_A2_CKPT="${H6_A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
MMOV_ROOT="${MMOV_ROOT:-/root/shared-nvme/BRIGHT-master/third_party/MM-OVSeg/MM-OVSeg}"
SAR_DINO_CKPT="${SAR_DINO_CKPT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/P3_RESET}"
SEED="${SEED:-3407}"

case "${MODE}" in
  preflight)
    bash -n run/run_P3_reset_5090.sh run/run_P3_reset_sequence.sh
    python -m py_compile \
      model/p3_reset/P3_Reset_Model.py \
      model/p3_reset/p3_feature_audit.py \
      model/p3_reset/sar_sources/*.py \
      model/p3_reset/fusion/*.py model/p3_reset/heads/*.py \
      script/standard_ML/train_P3_reset_feature_audit.py \
      script/standard_ML/summarize_P3_reset_feature_audit.py \
      script/standard_ML/train_P3_reset_model.py
    python -m unittest -v tests.test_p3_reset
    echo "P3 reset preflight passed."
    exit 0 ;;
  feature_smoke|feature_train)
    FEATURE_VARIANT="${FEATURE_VARIANT:?FEATURE_VARIANT is required}"
    if [[ "${FEATURE_VARIANT}" == "b2" ]]; then
      test -s "${SAR_DINO_CKPT}" || { echo "Missing SAR-DINO checkpoint: ${SAR_DINO_CKPT}" >&2; exit 1; }
    fi
    STEPS="$([[ "${MODE}" == feature_smoke ]] && echo "${OPTIMIZER_STEPS:-20}" || echo "${OPTIMIZER_STEPS:-1500}")"
    EVAL_EVERY="$([[ "${MODE}" == feature_smoke ]] && echo "${EVAL_EVERY:-20}" || echo "${EVAL_EVERY:-500}")"
    VAL_MAX="$([[ "${MODE}" == feature_smoke ]] && echo "${VAL_MAX_BATCHES:-2}" || echo 0)"
    OUT="${OUTPUT_ROOT}/FEATURE_AUDIT/${FEATURE_VARIANT^^}"
    python -u script/standard_ML/train_P3_reset_feature_audit.py \
      --variant "${FEATURE_VARIANT}" --p3_checkpoint "${P3_CKPT}" \
      --h6_checkpoint "${H6_A2_CKPT}" --sar_dino_checkpoint "${SAR_DINO_CKPT}" \
      --mmov_source_root "${MMOV_ROOT}" --dataset_path "${DATASET_ROOT}" \
      --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}" --sar_stats_path "${SAR_STATS_JSON}" \
      --output_dir "${OUT}" --steps "${STEPS}" --eval_every "${EVAL_EVERY}" \
      --batch_size "${TRAIN_BATCH_SIZE:-2}" --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
      --workers "${NUM_WORKERS:-8}" --val_max_batches "${VAL_MAX}" --seed "${SEED}"
    exit 0 ;;
  summarize_features)
    python -u script/standard_ML/summarize_P3_reset_feature_audit.py \
      --root "${OUTPUT_ROOT}/FEATURE_AUDIT" \
      --output "${OUTPUT_ROOT}/FEATURE_AUDIT/feature_source_decision.json"
    exit 0 ;;
  model_smoke|model_train)
    VARIANT="${VARIANT:?VARIANT is required}"; SAR_SOURCE="${SAR_SOURCE:-b0}"
    STEPS="$([[ "${MODE}" == model_smoke ]] && echo "${OPTIMIZER_STEPS:-20}" || echo "${OPTIMIZER_STEPS:-3000}")"
    EVAL_EVERY="$([[ "${MODE}" == model_smoke ]] && echo "${EVAL_EVERY:-20}" || echo "${EVAL_EVERY:-500}")"
    VAL_MAX="$([[ "${MODE}" == model_smoke ]] && echo "${VAL_MAX_BATCHES:-2}" || echo 0)"
    OUT="${OUTPUT_ROOT}/MODEL/${VARIANT^^}_${SAR_SOURCE^^}_${STEPS}_seed${SEED}"
    python -u script/standard_ML/train_P3_reset_model.py \
      --variant "${VARIANT}" --sar_source "${SAR_SOURCE}" \
      --interaction "${INTERACTION:-c1}" --fusion_checkpoint "${FUSION_CKPT:-}" \
      --feature_decision "${OUTPUT_ROOT}/FEATURE_AUDIT/feature_source_decision.json" \
      --p3_checkpoint "${P3_CKPT}" --h6_checkpoint "${H6_A2_CKPT}" \
      --sar_dino_checkpoint "${SAR_DINO_CKPT}" --mmov_source_root "${MMOV_ROOT}" \
      --dataset_path "${DATASET_ROOT}" --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" --output_dir "${OUT}" \
      --steps "${STEPS}" --eval_every "${EVAL_EVERY}" --val_max_batches "${VAL_MAX}" \
      --train_batch_size "${TRAIN_BATCH_SIZE:-2}" --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
      --accumulation "${ACCUMULATION_STEPS:-8}" --workers "${NUM_WORKERS:-8}" --seed "${SEED}"
    exit 0 ;;
  *) echo "Usage: $0 preflight|feature_smoke|feature_train|summarize_features|model_smoke|model_train" >&2; exit 2 ;;
esac
