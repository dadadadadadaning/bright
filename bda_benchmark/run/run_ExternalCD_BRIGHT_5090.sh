#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

VARIANT="${VARIANT:-x1}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/ExternalCD_BRIGHT}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
EXP_NAME="${EXP_NAME:-}"

case "${MODE}" in
  preflight)
    python -m py_compile \
      model/DamageFormer_RGBSAR_ExternalCD.py \
      dataset/make_data_loader_heterogeneous_rgbsar.py \
      script/standard_ML/train_ExternalCD_BRIGHT.py \
      script/standard_ML/train_DamageFormer_RGBSAR_Heterogeneous.py
    python -m unittest -v \
      tests.test_external_cd_bright \
      tests.test_external_cd_sequence_contract
    if [[ "${CHECK_HETECD:-0}" == "1" ]]; then
      test -s "${REPOSITORY_ROOT}/third_party/HeteCD/models/hetecd.py" || {
        echo "Missing official HeteCD source: ${REPOSITORY_ROOT}/third_party/HeteCD" >&2
        exit 1
      }
      python -c "from bda_benchmark.model.DamageFormer_RGBSAR_ExternalCD import build_external_cd_bright; m=build_external_cd_bright('x2', pretrained_rgb=False); print('Official HeteCD adapter parameters:', sum(p.numel() for p in m.parameters()))"
    fi
    echo "ExternalCD-BRIGHT preflight passed."
    exit 0
    ;;
  smoke)
    STEPS=20; EVAL_EVERY=20; SAVE_EVERY=20; VAL_MAX_BATCHES=2; TAG=smoke20 ;;
  screen)
    STEPS="${OPTIMIZER_STEPS:-1500}"; EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-1500}"; VAL_MAX_BATCHES=0; TAG=screen${STEPS} ;;
  pilot)
    STEPS="${OPTIMIZER_STEPS:-3000}"; EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-3000}"; VAL_MAX_BATCHES=0; TAG=pilot${STEPS} ;;
  eval_val|eval_test)
    test -s "${EXTERNAL_CD_CKPT:-}" || {
      echo "Set EXTERNAL_CD_CKPT to a non-empty checkpoint." >&2
      exit 1
    }
    if [[ "${MODE}" == "eval_test" && "${ALLOW_TEST:-0}" != "1" ]]; then
      echo "Test evaluation is locked. Set ALLOW_TEST=1 only after val protocol is frozen." >&2
      exit 1
    fi
    if [[ "${MODE}" == "eval_test" ]]; then
      VAL_LIST="${TEST_LIST:-${DATASET_ROOT}/splitname/standard_ML/test_set.txt}"
      TAG=independent_test
    else
      TAG=independent_val
    fi
    STEPS=0; EVAL_EVERY=1; SAVE_EVERY=1; VAL_MAX_BATCHES=0
    EXP_NAME="${EVAL_EXP_NAME:-ExternalCD_${VARIANT^^}_${TAG}_seed${SEED}}"
    EVAL_ONLY=1
    ;;
  *)
    echo "Usage: $0 preflight|smoke|screen|pilot|eval_val|eval_test" >&2
    exit 2
    ;;
esac

[[ "${VARIANT}" == "x1" || "${VARIANT}" == "x2" ]] || {
  echo "VARIANT must be x1 (plain dual) or x2 (official HeteCD adapter)" >&2
  exit 2
}
test -s "${SAR_STATS_JSON}" || {
  echo "Missing train-only SAR statistics: ${SAR_STATS_JSON}" >&2
  echo "Run: bash run/run_HeterogeneousDamageFormer_5090.sh stats" >&2
  exit 1
}
if [[ "${VARIANT}" == "x2" ]]; then
  test -s "${REPOSITORY_ROOT}/third_party/HeteCD/models/hetecd.py" || {
    echo "X2 requires official source at ${REPOSITORY_ROOT}/third_party/HeteCD." >&2
    echo "Clone from repository root: git clone https://github.com/weiAI1996/HeteCD.git third_party/HeteCD" >&2
    exit 1
  }
fi

if [[ -z "${EXP_NAME}" ]]; then
  EXP_NAME="ExternalCD_${VARIANT^^}_${TAG}_seed${SEED}"
fi
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"
if [[ "${EVAL_ONLY:-0}" != "1" && -s "${OUTPUT_DIR}/training_summary.json" && "${FORCE_RERUN:-0}" != "1" ]]; then
  echo "Completed experiment already exists; skipped: ${OUTPUT_DIR}"
  exit 0
fi

echo "================================================================================"
echo "Protocol  : External optical-SAR whole-model baseline on BRIGHT"
echo "Variant   : ${VARIANT}"
echo "Objective : one 4-class CE + Lovasz loss"
echo "Loc head  : derived from 4-class logits; no separate localisation loss"
echo "SAR input : native 1-channel + frozen train-split statistics"
echo "Test      : NEVER used by this training entry"
echo "Output    : ${OUTPUT_DIR}"
echo "================================================================================"

EXTRA_ARGS=()
if [[ "${EVAL_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS=(--validation_only --no_pretrained_rgb --resume "${EXTERNAL_CD_CKPT}")
elif [[ -s "${OUTPUT_DIR}/latest_resume.pth" ]]; then
  EXTRA_ARGS=(--resume "${OUTPUT_DIR}/latest_resume.pth")
fi

python -u script/standard_ML/train_ExternalCD_BRIGHT.py \
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
  --rgb_learning_rate "${RGB_LR:-1e-5}" \
  --sar_learning_rate "${SAR_LR:-1e-4}" \
  --fusion_learning_rate "${FUSION_LR:-1e-4}" \
  --weight_decay "${WEIGHT_DECAY:-5e-3}" \
  --loc_ce_weight 0 \
  --loc_lovasz_weight 0 \
  --damage_ce_weight 1 \
  --damage_lovasz_weight 0.75 \
  --sar_aux_weight 0 \
  --representation_weight "${FCA_WEIGHT:-0}" \
  --amp "${AMP:-bf16}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"
