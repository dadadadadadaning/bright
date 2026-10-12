#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
VARIANT="${VARIANT:-h3}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/HeterogeneousDamageFormer}"
SAR_STATS_JSON="${SAR_STATS_JSON:-${OUTPUT_ROOT}/sar_train_stats.json}"
EXP_NAME="${EXP_NAME:-}"
EXP_SUFFIX="${EXP_SUFFIX:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"
if [[ ! "${EXP_SUFFIX}" =~ ^[A-Za-z0-9_.-]*$ ]]; then
  echo "EXP_SUFFIX contains unsupported characters: ${EXP_SUFFIX}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${MODE}" in
  preflight)
    python -m py_compile \
      model/DamageFormer_RGBSAR_Heterogeneous.py \
      dataset/make_data_loader_heterogeneous_rgbsar.py \
      script/standard_ML/compute_HeterogeneousSAR_stats.py \
      script/standard_ML/train_DamageFormer_RGBSAR_Heterogeneous.py
    python -m unittest -v tests.test_damageformer_rgbsar_heterogeneous
    echo "Heterogeneous DamageFormer preflight passed."
    exit 0
    ;;
  stats)
    mkdir -p "$(dirname "${SAR_STATS_JSON}")"
    if [[ -s "${SAR_STATS_JSON}" && "${FORCE_RERUN}" != "1" ]]; then
      python - "${SAR_STATS_JSON}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert payload["protocol"] == "HeterogeneousDamageFormer_train_only_SAR_stats_v1"
print("Frozen train-only SAR stats already exist:", sys.argv[1])
PY
      exit 0
    fi
    python -u script/standard_ML/compute_HeterogeneousSAR_stats.py \
      --dataset_path "${DATASET_ROOT}" \
      --train_data_list_path "${TRAIN_LIST}" \
      --output_json "${SAR_STATS_JSON}"
    exit 0
    ;;
  smoke)
    OPTIMIZER_STEPS=20
    EVAL_EVERY=20
    SAVE_EVERY=20
    VAL_MAX_BATCHES=2
    TAG="smoke20"
    ;;
  pilot)
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-3000}"
    EVAL_EVERY="${EVAL_EVERY:-1000}"
    SAVE_EVERY="${SAVE_EVERY:-1000}"
    VAL_MAX_BATCHES=0
    TAG="pilot${OPTIMIZER_STEPS}"
    ;;
  full)
    # 8,000 optimizer steps x effective batch 16 = 128,000 training crops.
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-8000}"
    EVAL_EVERY="${EVAL_EVERY:-1000}"
    SAVE_EVERY="${SAVE_EVERY:-1000}"
    VAL_MAX_BATCHES=0
    TAG="full${OPTIMIZER_STEPS}"
    ;;
  paper)
    # Original paper-scale protocol: 50,000 steps x EBS16 = 800,000 crops.
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-50000}"
    EVAL_EVERY="${EVAL_EVERY:-1000}"
    SAVE_EVERY="${SAVE_EVERY:-1000}"
    VAL_MAX_BATCHES=0
    TAG="paper${OPTIMIZER_STEPS}"
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|stats|smoke|pilot|full|paper" >&2
    exit 2
    ;;
esac

if [[ "${VARIANT}" != "h0" ]]; then
  test -s "${SAR_STATS_JSON}" || {
    echo "Missing SAR stats: ${SAR_STATS_JSON}; run '$0 stats' first." >&2
    exit 1
  }
fi
test -s "${TRAIN_LIST}"
test -s "${VAL_LIST}"

if [[ -z "${EXP_NAME}" ]]; then
  EXP_NAME="HDF_${VARIANT^^}_${TAG}_seed${SEED}"
  if [[ -n "${EXP_SUFFIX}" ]]; then
    EXP_NAME="${EXP_NAME}_${EXP_SUFFIX}"
  fi
fi
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
if [[ -s "${OUTPUT_DIR}/training_summary.json" && "${FORCE_RERUN}" != "1" ]]; then
  echo "Completed experiment already exists; skipped: ${OUTPUT_DIR}"
  exit 0
fi

RESUME_ARGS=()
if [[ -s "${OUTPUT_DIR}/latest_resume.pth" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT_DIR}/latest_resume.pth")
fi
MILESTONE_ARGS=()
if [[ "${KEEP_MILESTONE_CHECKPOINTS:-0}" == "1" ]]; then
  MILESTONE_ARGS=(--keep_milestone_checkpoints)
fi
STATS_ARGS=()
if [[ "${VARIANT}" != "h0" ]]; then
  STATS_ARGS=(--sar_stats_path "${SAR_STATS_JSON}")
fi

echo "================================================================================"
echo "Experiment : ${EXP_NAME}"
echo "Variant    : ${VARIANT}"
echo "Teacher    : NONE"
echo "Dataset    : ${DATASET_ROOT}"
echo "Stats      : ${SAR_STATS_JSON}"
echo "Output     : ${OUTPUT_DIR}"
echo "Steps/EBS  : ${OPTIMIZER_STEPS}/16"
echo "================================================================================"

python -u script/standard_ML/train_DamageFormer_RGBSAR_Heterogeneous.py \
  --variant "${VARIANT}" \
  --train_dataset_path "${DATASET_ROOT}" \
  --train_data_list_path "${TRAIN_LIST}" \
  --val_dataset_path "${DATASET_ROOT}" \
  --val_data_list_path "${VAL_LIST}" \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --crop_size 640 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
  --optimizer_steps "${OPTIMIZER_STEPS}" \
  --eval_every "${EVAL_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --log_every "${LOG_EVERY:-20}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --val_max_batches "${VAL_MAX_BATCHES}" \
  --rgb_learning_rate "${RGB_LR:-1e-5}" \
  --sar_learning_rate "${SAR_LR:-1e-4}" \
  --fusion_learning_rate "${FUSION_LR:-1e-4}" \
  --weight_decay "${WEIGHT_DECAY:-5e-3}" \
  --amp "${AMP:-bf16}" \
  --seed "${SEED}" \
  "${MILESTONE_ARGS[@]}" \
  "${STATS_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
