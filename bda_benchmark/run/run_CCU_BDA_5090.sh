#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
VARIANT="${VARIANT:-c3}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
TEST_LIST="${TEST_LIST:-${DATASET_ROOT}/splitname/standard_ML/test_set.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/CCU_BDA}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/CCU_BDA}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
H6_A2_CKPT="${H6_A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
AUDIT_ROOT="${AUDIT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CCU_BDA}"
EXP_NAME="${EXP_NAME:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${MODE}" in
  preflight)
    bash -n run/run_CCU_BDA_5090.sh
    bash -n run/run_CCU_BDA_sequence.sh
    bash -n run/run_CCU_E1_E3_sequence.sh
    bash -n run/run_CCU_E0_E2_sequence.sh
    python -m py_compile \
      model/DamageFormer_RGBSAR_CCU_BDA.py \
      script/standard_ML/audit_CCU_local_correspondence.py \
      script/standard_ML/ccu_bda_runtime.py \
      script/standard_ML/train_DamageFormer_RGBSAR_CCU_BDA.py \
      script/standard_ML/eval_DamageFormer_RGBSAR_CCU_BDA.py \
      script/standard_ML/summarize_CCU_BDA.py
    python -m unittest -v \
      tests.test_damageformer_rgbsar_ccu_bda \
      tests.test_ccu_correspondence_audit
    echo "CCU-BDA preflight passed."
    exit 0
    ;;
  audit_smoke|audit_val)
    for required in "${H6_A2_CKPT}" "${SAR_STATS_JSON}" "${VAL_LIST}"; do
      test -s "${required}" || {
        echo "Missing E0 correspondence-audit input: ${required}" >&2
        exit 1
      }
    done
    if [[ "${MODE}" == "audit_smoke" ]]; then
      AUDIT_MAX_BATCHES="${AUDIT_MAX_BATCHES:-2}"
      AUDIT_TAG="smoke"
    else
      AUDIT_MAX_BATCHES=0
      AUDIT_TAG="val"
    fi
    AUDIT_OUTPUT_JSON="${AUDIT_OUTPUT_JSON:-${AUDIT_ROOT}/E0_H6A2_local_correspondence_${AUDIT_TAG}.json}"
    echo "================================================================================"
    echo "Protocol         : CCU-BDA E0 validation-only correspondence audit"
    echo "Locked H6-A2     : ${H6_A2_CKPT}"
    echo "Validation list  : ${VAL_LIST}"
    echo "Training         : NONE"
    echo "Test accessed    : NEVER"
    echo "Output           : ${AUDIT_OUTPUT_JSON}"
    echo "================================================================================"
    python -u script/standard_ML/audit_CCU_local_correspondence.py \
      --checkpoint "${H6_A2_CKPT}" \
      --dataset_path "${DATASET_ROOT}" \
      --data_list_path "${VAL_LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" \
      --output_json "${AUDIT_OUTPUT_JSON}" \
      --device cuda:0 --crop_size 640 \
      --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
      --num_workers "${NUM_WORKERS:-8}" \
      --max_batches "${AUDIT_MAX_BATCHES}" \
      --window_size "${CORRELATION_WINDOW:-5}" \
      --temperature "${CORRELATION_TEMPERATURE:-0.10}" \
      --min_class_purity "${AUDIT_MIN_CLASS_PURITY:-0.75}" \
      --seed "${SEED}"
    exit 0
    ;;
  stats)
    mkdir -p "$(dirname "${SAR_STATS_JSON}")"
    if [[ -s "${SAR_STATS_JSON}" && "${FORCE_RERUN}" != "1" ]]; then
      python - "${SAR_STATS_JSON}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert "normalization" in payload
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
  guard)
    OPTIMIZER_STEPS=0; EVAL_EVERY=1; SAVE_EVERY=1
    VAL_MAX_BATCHES=0; TAG="guard"; VALIDATION_ARGS=(--validation_only)
    ;;
  smoke)
    OPTIMIZER_STEPS=20; EVAL_EVERY=20; SAVE_EVERY=20
    VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-2}"
    TAG="smoke20"; VALIDATION_ARGS=()
    ;;
  pilot)
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-3000}"
    EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-3000}"
    VAL_MAX_BATCHES=0; TAG="pilot${OPTIMIZER_STEPS}"; VALIDATION_ARGS=()
    ;;
  controlled5000)
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-5000}"
    EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-5000}"
    VAL_MAX_BATCHES=0; TAG="controlled${OPTIMIZER_STEPS}"; VALIDATION_ARGS=()
    ;;
  full)
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-8000}"
    EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-1000}"
    VAL_MAX_BATCHES=0; TAG="full${OPTIMIZER_STEPS}"; VALIDATION_ARGS=()
    ;;
  paper)
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-50000}"
    EVAL_EVERY="${EVAL_EVERY:-1000}"
    SAVE_EVERY="${SAVE_EVERY:-1000}"
    VAL_MAX_BATCHES=0; TAG="paper${OPTIMIZER_STEPS}"; VALIDATION_ARGS=()
    ;;
  eval_val|eval_test)
    if [[ "${MODE}" == "eval_test" ]]; then
      EVAL_SPLIT="test"; EVAL_LIST="${TEST_LIST}"
    else
      EVAL_SPLIT="val"; EVAL_LIST="${VAL_LIST}"
    fi
    CCU_CKPT="${CCU_CKPT:-}"
    test -n "${CCU_CKPT}" || {
      echo "${MODE} requires CCU_CKPT=/path/to/best_val_miou.pth" >&2
      exit 1
    }
    for required in "${CCU_CKPT}" "${SAR_STATS_JSON}" "${EVAL_LIST}"; do
      test -s "${required}" || {
        echo "Missing CCU-BDA evaluation input: ${required}" >&2
        exit 1
      }
    done
    EVAL_OUTPUT_JSON="${EVAL_OUTPUT_JSON:-${CCU_CKPT%/*}/independent_${EVAL_SPLIT}.json}"
    python -u script/standard_ML/eval_DamageFormer_RGBSAR_CCU_BDA.py \
      --checkpoint "${CCU_CKPT}" \
      --split "${EVAL_SPLIT}" \
      --dataset_path "${DATASET_ROOT}" \
      --data_list_path "${EVAL_LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" \
      --output_json "${EVAL_OUTPUT_JSON}" \
      --device cuda:0 --crop_size 640 \
      --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
      --num_workers "${NUM_WORKERS:-8}" --seed "${SEED}"
    exit 0
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|stats|audit_smoke|audit_val|guard|smoke|pilot|controlled5000|full|paper|eval_val|eval_test" >&2
    exit 2
    ;;
esac

case "${VARIANT}" in c0|c1|c2|c3|c4|e1|e2|e3) ;; *) echo "Unknown CCU variant: ${VARIANT}" >&2; exit 2;; esac
if [[ "${VARIANT}" =~ ^e[123]$ ]]; then
  CCU_SOURCE_CKPT="${CCU_SOURCE_CKPT:-${H6_A2_CKPT}}"
fi
for required in "${SAR_STATS_JSON}" "${TRAIN_LIST}" "${VAL_LIST}"; do
  test -s "${required}" || { echo "Missing required CCU input: ${required}" >&2; exit 1; }
done
if [[ -n "${CCU_SOURCE_CKPT:-}" ]]; then
  test -s "${CCU_SOURCE_CKPT}" || { echo "Missing CCU_SOURCE_CKPT: ${CCU_SOURCE_CKPT}" >&2; exit 1; }
fi
if [[ -n "${REGISTRATION_CKPT:-}" ]]; then
  test -s "${REGISTRATION_CKPT}" || { echo "Missing REGISTRATION_CKPT: ${REGISTRATION_CKPT}" >&2; exit 1; }
fi

if [[ -z "${EXP_NAME}" ]]; then
  EXP_NAME="CCU_${VARIANT^^}_${TAG}_seed${SEED}"
fi
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
if [[ -s "${OUTPUT_DIR}/training_summary.json" && "${FORCE_RERUN}" != "1" ]]; then
  echo "Completed CCU-BDA experiment already exists; skipped: ${OUTPUT_DIR}"
  exit 0
fi
RESUME_ARGS=()
if [[ -s "${OUTPUT_DIR}/latest_resume.pth" && "${FORCE_RERUN}" != "1" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT_DIR}/latest_resume.pth")
fi
MILESTONE_ARGS=()
if [[ "${KEEP_MILESTONE_CHECKPOINTS:-0}" == "1" ]]; then
  MILESTONE_ARGS=(--keep_milestone_checkpoints)
fi
SAFETY_ARGS=()
CONTROLLED_LOSS_ARGS=()
if [[ "${VARIANT}" =~ ^e[123]$ ]]; then
  SAFETY_ARGS=(--select_initial_checkpoint --error_on_nonfinite_grad)
  # Every inherited H6-A2 branch is frozen.  Optimise only the existing
  # damage CE+Lovasz objective; localisation/SAR auxiliary outputs remain
  # evaluation monitors and must not obscure the controlled loss contract.
  CONTROLLED_LOSS_ARGS=(
    --loc_ce_weight 0 --loc_lovasz_weight 0
    --sar_aux_weight 0 --representation_weight 0
  )
fi

echo "================================================================================"
echo "Experiment       : ${EXP_NAME}"
echo "Variant          : ${VARIANT}"
echo "Native SAR       : one channel + train-only statistics"
echo "RGB localization : RGB branch only"
if [[ "${VARIANT}" =~ ^c[234]$ ]]; then
  LOCAL_MATCHING_DESCRIPTION="stage3+stage4"
elif [[ "${VARIANT}" == "e3" ]]; then
  LOCAL_MATCHING_DESCRIPTION="stage4-only-soft"
else
  LOCAL_MATCHING_DESCRIPTION="disabled"
fi
echo "Local matching   : ${LOCAL_MATCHING_DESCRIPTION}"
echo "3D difference    : $([[ ${VARIANT} == e2 ]] && echo stage4-multiscale || echo disabled)"
echo "Change condition : $([[ ${VARIANT} =~ ^c[34]$ ]] && echo enabled || echo disabled)"
echo "Hierarchy        : $([[ ${VARIANT} == c4 ]] && echo enabled || echo disabled)"
echo "Controlled base  : $([[ ${VARIANT} =~ ^e[123]$ ]] && echo frozen-H6-A2 || echo none)"
echo "Peak signal      : $([[ ${VARIANT} =~ ^e[123]$ ]] && echo stage4-only || echo legacy)"
echo "Optimized loss   : $([[ ${VARIANT} =~ ^e[123]$ ]] && echo damage-CE+Lovasz-only || echo standard)"
echo "Controlled source: ${CCU_SOURCE_CKPT:-NONE}"
echo "Registration init: ${REGISTRATION_CKPT:-NONE}"
echo "Teacher/KD       : NONE / NONE"
echo "Test selection   : NEVER"
echo "Output           : ${OUTPUT_DIR}"
echo "Steps/EBS        : ${OPTIMIZER_STEPS}/$(( ${TRAIN_BATCH_SIZE:-4} * ${ACCUMULATION_STEPS:-4} ))"
echo "================================================================================"

python -u script/standard_ML/train_DamageFormer_RGBSAR_CCU_BDA.py \
  --variant "${VARIANT}" \
  --train_dataset_path "${DATASET_ROOT}" \
  --train_data_list_path "${TRAIN_LIST}" \
  --val_dataset_path "${DATASET_ROOT}" \
  --val_data_list_path "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" \
  --output_dir "${OUTPUT_DIR}" \
  --device cuda:0 --crop_size 640 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
  --optimizer_steps "${OPTIMIZER_STEPS}" \
  --eval_every "${EVAL_EVERY}" --save_every "${SAVE_EVERY}" \
  --log_every "${LOG_EVERY:-20}" --num_workers "${NUM_WORKERS:-8}" \
  --val_max_batches "${VAL_MAX_BATCHES}" \
  --rgb_learning_rate "${RGB_LR:-1e-5}" \
  --sar_learning_rate "${SAR_LR:-1e-4}" \
  --fusion_learning_rate "${FUSION_LR:-1e-4}" \
  --weight_decay "${WEIGHT_DECAY:-5e-3}" \
  --amp "${AMP:-bf16}" --seed "${SEED}" \
  "${VALIDATION_ARGS[@]}" "${MILESTONE_ARGS[@]}" \
  "${SAFETY_ARGS[@]}" "${CONTROLLED_LOSS_ARGS[@]}" "${RESUME_ARGS[@]}"
