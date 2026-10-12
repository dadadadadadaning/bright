#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

VARIANT="${VARIANT:-n0}"
SEED="${SEED:-3407}"
PHASE="${OABH_PHASE:-joint}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/OA_HSDNET}"
EXP_NAME="${EXP_NAME:-}"

case "${MODE}" in
  preflight)
    python -m py_compile \
      model/DamageFormer_RGBSAR_OABH.py \
      dataset/make_data_loader_heterogeneous_rgbsar.py \
      script/standard_ML/train_OABH.py \
      script/standard_ML/summarize_OABH.py \
      script/standard_ML/summarize_OABH_event_style.py \
      dataset/event_balanced_sampling.py \
      script/standard_ML/train_DamageFormer_RGBSAR_Heterogeneous.py
    python -m unittest -v tests.test_oabh
    echo "OABH preflight passed."
    exit 0
    ;;
  smoke)
    STEPS=20; EVAL_EVERY=20; SAVE_EVERY=20; VAL_MAX_BATCHES=2
    PHASE="heads"; TAG="smoke20"
    ;;
  heads)
    STEPS="${OPTIMIZER_STEPS:-500}"; EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-500}"; VAL_MAX_BATCHES=0
    PHASE="heads"; TAG="heads${STEPS}"
    ;;
  deep)
    STEPS="${OPTIMIZER_STEPS:-1500}"; EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-1500}"; VAL_MAX_BATCHES=0
    PHASE="deep"; TAG="deep${STEPS}"
    ;;
  joint)
    STEPS="${OPTIMIZER_STEPS:-3000}"; EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-3000}"; VAL_MAX_BATCHES=0
    PHASE="joint"; TAG="joint${STEPS}"
    ;;
  eval_val|eval_test)
    test -s "${OABH_CKPT:-}" || {
      echo "Set OABH_CKPT to a non-empty OABH checkpoint." >&2
      exit 1
    }
    if [[ "${MODE}" == "eval_test" && "${ALLOW_TEST:-0}" != "1" ]]; then
      echo "Test is locked. Set ALLOW_TEST=1 only after the validation protocol is frozen." >&2
      exit 1
    fi
    if [[ "${MODE}" == "eval_test" ]]; then
      VAL_LIST="${TEST_LIST:-${DATASET_ROOT}/splitname/standard_ML/test_set.txt}"
      TAG=independent_test
    else
      TAG=independent_val
    fi
    STEPS=0; EVAL_EVERY=1; SAVE_EVERY=1; VAL_MAX_BATCHES=0
    PHASE="joint"; EVAL_ONLY=1
    ;;
  *)
    echo "Usage: $0 preflight|smoke|heads|deep|joint|eval_val|eval_test" >&2
    exit 2
    ;;
esac

case "${VARIANT}" in
  n0|n1_shared|n1_stage|n1_cond|n2) ;;
  *) echo "VARIANT must be n0, n1_shared, n1_stage, n1_cond, or n2" >&2
  exit 2
  ;;
esac
test -s "${SAR_STATS_JSON}" || {
  echo "Missing frozen train-only SAR statistics: ${SAR_STATS_JSON}" >&2
  exit 1
}

if [[ -z "${EXP_NAME}" ]]; then
  EXP_NAME="OA_HSDNET_${VARIANT^^}_${TAG}_seed${SEED}"
fi
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"

echo "================================================================================"
echo "Protocol          : OA-HSDNet asymmetric hierarchical RGB-SAR model"
echo "Variant/phase     : ${VARIANT}/${PHASE}"
echo "Prediction        : native dual-stream main model; no P3 forward/residual/router"
echo "Task scales       : Occurrence=Stage2+3 | Destruction=Stage3+4"
echo "SAR style mode    : ${OABH_SAR_STYLE_MODE:-none}"
echo "Probability heads : Building -> Affected|Building -> Destroyed|Affected"
echo "Selector          : validation event-worst only; test never selects"
echo "Steps/output      : ${STEPS}/${OUTPUT_DIR}"
echo "================================================================================"

EXTRA_ARGS=()
if [[ "${EVENT_BALANCED_SAMPLING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--event_balanced_sampling)
fi
if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
  unset OABH_H6_INIT_CKPT OABH_FULL_INIT_CKPT
  EXTRA_ARGS+=(--validation_only --no_pretrained_rgb --resume "${OABH_CKPT}")
fi

export OABH_PHASE="${PHASE}"
python -u script/standard_ML/train_OABH.py \
  --variant "${VARIANT}" \
  --train_dataset_path "${DATASET_ROOT}" \
  --train_data_list_path "${TRAIN_LIST}" \
  --val_dataset_path "${DATASET_ROOT}" \
  --val_data_list_path "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda:0 \
  --crop_size "${CROP_SIZE:-640}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
  --optimizer_steps "${STEPS}" \
  --eval_every "${EVAL_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --log_every "${LOG_EVERY:-20}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --val_max_batches "${VAL_MAX_BATCHES}" \
  --rgb_learning_rate "${RGB_LR:-5e-6}" \
  --sar_learning_rate "${SAR_LR:-1e-5}" \
  --fusion_learning_rate "${FUSION_LR:-1e-4}" \
  --weight_decay "${WEIGHT_DECAY:-5e-3}" \
  --loc_ce_weight 1 \
  --loc_lovasz_weight 0.5 \
  --damage_ce_weight 1 \
  --damage_lovasz_weight 0.75 \
  --sar_aux_weight 0 \
  --representation_weight 0 \
  --selection_metric event_worst_official_style_4class_miou \
  --select_initial_checkpoint \
  --amp "${AMP:-bf16}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"
