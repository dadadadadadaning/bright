#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
VARIANT_WAS_SET="${VARIANT+x}"
VARIANT="${VARIANT:-a1}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H4_RGBSAR}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H4_RGBSAR}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
DAMAGE_PRIORS_JSON="${DAMAGE_PRIORS_JSON:-${OUTPUT_ROOT}/damage_train_priors.json}"
SOURCE_H3_CKPT="${SOURCE_H3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/HDF_H3_full8000_seed3407_fullv1/best_val_miou.pth}"
SAR_PRETRAIN_DIR="${SAR_PRETRAIN_DIR:-${OUTPUT_ROOT}/H4_SAR_MAE_trainonly_seed${SEED}}"
SAR_PRETRAIN_CKPT="${SAR_PRETRAIN_CKPT:-${SAR_PRETRAIN_DIR}/sar_encoder_final.pth}"
EXP_NAME="${EXP_NAME:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${MODE}" in
  preflight)
    bash -n run/run_H4_RGBSAR_5090.sh
    bash -n run/run_H4_RGBSAR_sequence.sh
    python -m py_compile \
      model/DamageFormer_RGBSAR_H4.py \
      script/standard_ML/compute_H4_damage_priors.py \
      script/standard_ML/pretrain_H4_sar_mae.py \
      script/standard_ML/train_DamageFormer_RGBSAR_H4.py \
      script/standard_ML/summarize_H4_results.py
    python -m unittest -v tests.test_damageformer_rgbsar_h4
    echo "H4 RGB/SAR preflight passed."
    exit 0
    ;;
  stats)
    mkdir -p "$(dirname "${SAR_STATS_JSON}")"
    if [[ -s "${SAR_STATS_JSON}" && "${FORCE_RERUN}" != "1" ]]; then
      echo "Frozen train-only SAR stats already exist: ${SAR_STATS_JSON}"
      exit 0
    fi
    python -u script/standard_ML/compute_HeterogeneousSAR_stats.py \
      --dataset_path "${DATASET_ROOT}" \
      --train_data_list_path "${TRAIN_LIST}" \
      --output_json "${SAR_STATS_JSON}"
    exit 0
    ;;
  priors)
    mkdir -p "$(dirname "${DAMAGE_PRIORS_JSON}")"
    if [[ -s "${DAMAGE_PRIORS_JSON}" && "${FORCE_RERUN}" != "1" ]]; then
      echo "Frozen train-only damage priors already exist: ${DAMAGE_PRIORS_JSON}"
      exit 0
    fi
    python -u script/standard_ML/compute_H4_damage_priors.py \
      --dataset_path "${DATASET_ROOT}" \
      --train_data_list_path "${TRAIN_LIST}" \
      --output_json "${DAMAGE_PRIORS_JSON}"
    exit 0
    ;;
  summary)
    mkdir -p "${OUTPUT_ROOT}"
    python -u script/standard_ML/summarize_H4_results.py \
      --output_root "${OUTPUT_ROOT}" \
      --baseline_json "${BASELINE_JSON:-}" \
      --output_json "${OUTPUT_ROOT}/H4_validation_comparison.json"
    exit 0
    ;;
  sar_smoke)
    SAR_SMOKE_DIR="${SAR_PRETRAIN_DIR}_smoke20"
    mkdir -p "${SAR_SMOKE_DIR}"
    RESUME_ARGS=()
    if [[ -s "${SAR_SMOKE_DIR}/latest_resume.pth" ]]; then
      RESUME_ARGS=(--resume "${SAR_SMOKE_DIR}/latest_resume.pth")
    fi
    python -u script/standard_ML/pretrain_H4_sar_mae.py \
      --source_h3_checkpoint "${SOURCE_H3_CKPT}" \
      --train_dataset_path "${DATASET_ROOT}" \
      --train_data_list_path "${TRAIN_LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" \
      --output_dir "${SAR_SMOKE_DIR}" \
      --device cuda:0 \
      --optimizer_steps 20 \
      --save_every 20 \
      --log_every 5 \
      --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
      --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
      --num_workers "${NUM_WORKERS:-8}" \
      --amp "${AMP:-bf16}" \
      --seed "${SEED}" \
      "${RESUME_ARGS[@]}"
    exit 0
    ;;
  sar_pretrain)
    test -s "${SAR_STATS_JSON}"
    mkdir -p "${SAR_PRETRAIN_DIR}"
    if [[ -s "${SAR_PRETRAIN_DIR}/training_summary.json" && "${FORCE_RERUN}" != "1" ]]; then
      echo "SAR-MAE pretraining already completed: ${SAR_PRETRAIN_DIR}"
      exit 0
    fi
    RESUME_ARGS=()
    if [[ -s "${SAR_PRETRAIN_DIR}/latest_resume.pth" ]]; then
      RESUME_ARGS=(--resume "${SAR_PRETRAIN_DIR}/latest_resume.pth")
    fi
    python -u script/standard_ML/pretrain_H4_sar_mae.py \
      --source_h3_checkpoint "${SOURCE_H3_CKPT}" \
      --train_dataset_path "${DATASET_ROOT}" \
      --train_data_list_path "${TRAIN_LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" \
      --output_dir "${SAR_PRETRAIN_DIR}" \
      --device cuda:0 \
      --optimizer_steps "${SAR_PRETRAIN_STEPS:-5000}" \
      --save_every "${SAVE_EVERY:-1000}" \
      --log_every "${LOG_EVERY:-20}" \
      --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
      --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
      --num_workers "${NUM_WORKERS:-8}" \
      --learning_rate "${SAR_PRETRAIN_LR:-1e-4}" \
      --amp "${AMP:-bf16}" \
      --seed "${SEED}" \
      "${RESUME_ARGS[@]}"
    exit 0
    ;;
  guard)
    if [[ -z "${VARIANT_WAS_SET}" ]]; then
      VARIANT="a0"
    fi
    OPTIMIZER_STEPS=0
    EVAL_EVERY=1
    SAVE_EVERY=1
    VAL_MAX_BATCHES=0
    WARMUP_STEPS=0
    TAG="guard"
    VALIDATION_ARGS=(--validation_only)
    ;;
  smoke)
    OPTIMIZER_STEPS=20
    EVAL_EVERY=20
    SAVE_EVERY=20
    VAL_MAX_BATCHES=2
    WARMUP_STEPS=5
    TAG="smoke20"
    VALIDATION_ARGS=()
    ;;
  pilot)
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-3000}"
    EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-1000}"
    VAL_MAX_BATCHES=0
    WARMUP_STEPS="${WARMUP_STEPS:-500}"
    TAG="pilot${OPTIMIZER_STEPS}"
    VALIDATION_ARGS=()
    ;;
  full)
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-8000}"
    EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-1000}"
    VAL_MAX_BATCHES=0
    WARMUP_STEPS="${WARMUP_STEPS:-500}"
    TAG="full${OPTIMIZER_STEPS}"
    VALIDATION_ARGS=()
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|stats|priors|summary|sar_smoke|sar_pretrain|guard|smoke|pilot|full" >&2
    exit 2
    ;;
esac

case "${VARIANT}" in
  a0|a1|a2|a3|a4) ;;
  *)
    echo "Unknown H4 variant '${VARIANT}'; choose a0|a1|a2|a3|a4" >&2
    exit 2
    ;;
esac

test -s "${SOURCE_H3_CKPT}" || {
  echo "Missing H3 source checkpoint: ${SOURCE_H3_CKPT}" >&2
  exit 1
}
test -s "${SAR_STATS_JSON}" || {
  echo "Missing SAR stats: ${SAR_STATS_JSON}; run '$0 stats' first." >&2
  exit 1
}
test -s "${TRAIN_LIST}"
test -s "${VAL_LIST}"

EXTRA_ARGS=()
if [[ "${VARIANT}" == "a3" || "${VARIANT}" == "a4" ]]; then
  test -s "${SAR_PRETRAIN_CKPT}" || {
    echo "Missing SAR-MAE encoder: ${SAR_PRETRAIN_CKPT}; run '$0 sar_pretrain' first." >&2
    exit 1
  }
  EXTRA_ARGS+=(--sar_pretrained_checkpoint "${SAR_PRETRAIN_CKPT}")
fi
if [[ "${VARIANT}" == "a4" ]]; then
  test -s "${DAMAGE_PRIORS_JSON}" || {
    echo "Missing damage priors: ${DAMAGE_PRIORS_JSON}; run '$0 priors' first." >&2
    exit 1
  }
  EXTRA_ARGS+=(--damage_priors_json "${DAMAGE_PRIORS_JSON}")
fi

if [[ -z "${EXP_NAME}" ]]; then
  EXP_NAME="H4_${VARIANT^^}_${TAG}_seed${SEED}"
fi
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
if [[ -s "${OUTPUT_DIR}/training_summary.json" && "${FORCE_RERUN}" != "1" ]]; then
  echo "Completed H4 experiment already exists; skipped: ${OUTPUT_DIR}"
  exit 0
fi

RESUME_ARGS=()
if [[ -s "${OUTPUT_DIR}/latest_resume.pth" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT_DIR}/latest_resume.pth")
fi
SECONDARY_ARGS=()
if [[ "${SAVE_SECONDARY_BEST:-0}" == "1" ]]; then
  SECONDARY_ARGS=(--save_secondary_best)
fi

echo "================================================================================"
echo "Experiment : ${EXP_NAME}"
echo "Variant    : ${VARIANT}"
echo "H3 source  : ${SOURCE_H3_CKPT}"
echo "SAR stats  : ${SAR_STATS_JSON}"
echo "SAR MAE    : ${SAR_PRETRAIN_CKPT}"
echo "Priors     : ${DAMAGE_PRIORS_JSON}"
echo "Output     : ${OUTPUT_DIR}"
echo "Steps/EBS  : ${OPTIMIZER_STEPS}/16"
echo "================================================================================"

python -u script/standard_ML/train_DamageFormer_RGBSAR_H4.py \
  --variant "${VARIANT}" \
  --source_h3_checkpoint "${SOURCE_H3_CKPT}" \
  --train_dataset_path "${DATASET_ROOT}" \
  --train_data_list_path "${TRAIN_LIST}" \
  --val_dataset_path "${DATASET_ROOT}" \
  --val_data_list_path "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --crop_size 640 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
  --optimizer_steps "${OPTIMIZER_STEPS}" \
  --new_module_warmup_steps "${WARMUP_STEPS}" \
  --eval_every "${EVAL_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --log_every "${LOG_EVERY:-20}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --val_max_batches "${VAL_MAX_BATCHES}" \
  --new_module_learning_rate "${NEW_LR:-1e-4}" \
  --sar_learning_rate "${SAR_LR:-1e-5}" \
  --core_learning_rate "${CORE_LR:-1e-6}" \
  --weight_decay "${WEIGHT_DECAY:-5e-3}" \
  --amp "${AMP:-bf16}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}" \
  "${SECONDARY_ARGS[@]}" \
  "${VALIDATION_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
