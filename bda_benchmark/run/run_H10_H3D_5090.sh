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
TEST_LIST="${TEST_LIST:-${DATASET_ROOT}/splitname/standard_ML/test_set.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H10_H3D}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H10_H3D}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
H6_A2_CKPT="${H6_A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
H10_H3_CKPT="${H10_H3_CKPT:-}"
EXP_NAME="${EXP_NAME:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${MODE}" in
  preflight)
    bash -n run/run_H10_H3D_5090.sh
    bash -n run/run_H10_H3D_sequence.sh
    python -m py_compile \
      model/DamageFormer_RGBSAR_H10_H3D.py \
      script/standard_ML/h10_h3d_runtime.py \
      script/standard_ML/train_H10_H3D.py \
      script/standard_ML/eval_H10_H3D.py
    python -m unittest -v tests.test_h10_h3d
    echo "H10-H3D preflight passed."
    exit 0
    ;;
  guard)
    OPTIMIZER_STEPS=0; EVAL_EVERY=1; SAVE_EVERY=1
    VAL_MAX_BATCHES=0; TAG="guard"
    VALIDATION_ARGS=(--validation_only)
    ;;
  smoke)
    case "${VARIANT}" in h2|h3|h4) ;; *) echo "Smoke supports h2|h3|h4" >&2; exit 2;; esac
    OPTIMIZER_STEPS=20; EVAL_EVERY=20; SAVE_EVERY=20
    VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-2}"; TAG="smoke20"
    VALIDATION_ARGS=()
    ;;
  pilot)
    case "${VARIANT}" in h2|h3|h4) ;; *) echo "Pilot supports h2|h3|h4" >&2; exit 2;; esac
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-3000}"
    EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-3000}"
    VAL_MAX_BATCHES=0; TAG="pilot${OPTIMIZER_STEPS}"
    VALIDATION_ARGS=()
    ;;
  full)
    case "${VARIANT}" in h3|h4) ;; *) echo "Full supports selected h3|h4 only" >&2; exit 2;; esac
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-8000}"
    EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-1000}"
    VAL_MAX_BATCHES=0; TAG="full${OPTIMIZER_STEPS}"
    VALIDATION_ARGS=()
    ;;
  eval_val|eval_test)
    if [[ "${MODE}" == "eval_test" ]]; then
      EVAL_SPLIT="test"; EVAL_LIST="${TEST_LIST}"
    else
      EVAL_SPLIT="val"; EVAL_LIST="${VAL_LIST}"
    fi
    H10_CKPT="${H10_CKPT:-}"
    test -n "${H10_CKPT}" || {
      echo "${MODE} requires H10_CKPT=/path/to/best_val_miou.pth" >&2
      exit 1
    }
    for required in "${H10_CKPT}" "${SAR_STATS_JSON}" "${EVAL_LIST}"; do
      test -s "${required}" || { echo "Missing H10 evaluation input: ${required}" >&2; exit 1; }
    done
    EVAL_OUTPUT_JSON="${EVAL_OUTPUT_JSON:-${H10_CKPT%/*}/independent_${EVAL_SPLIT}.json}"
    python -u script/standard_ML/eval_H10_H3D.py \
      --checkpoint "${H10_CKPT}" --split "${EVAL_SPLIT}" \
      --dataset_path "${DATASET_ROOT}" --data_list_path "${EVAL_LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" --output_json "${EVAL_OUTPUT_JSON}" \
      --device cuda:0 --crop_size 640 \
      --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
      --num_workers "${NUM_WORKERS:-8}" --seed "${SEED}"
    exit 0
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|guard|smoke|pilot|full|eval_val|eval_test" >&2
    exit 2
    ;;
esac

case "${VARIANT}" in h0|h1|h2|h3|h4) ;; *) echo "Unknown H10 variant: ${VARIANT}" >&2; exit 2;; esac

SOURCE_CKPT="${H6_A2_CKPT}"
if [[ "${VARIANT}" == "h4" ]]; then
  test -n "${H10_H3_CKPT}" || {
    echo "H10-H4 requires H10_H3_CKPT=/path/to/H10-H3/best_val_miou.pth" >&2
    exit 1
  }
  SOURCE_CKPT="${H10_H3_CKPT}"
fi
for required in "${SOURCE_CKPT}" "${SAR_STATS_JSON}" "${TRAIN_LIST}" "${VAL_LIST}"; do
  test -s "${required}" || { echo "Missing required H10 input: ${required}" >&2; exit 1; }
done

if [[ -z "${EXP_NAME}" ]]; then
  EXP_NAME="H10_${VARIANT^^}_${TAG}_seed${SEED}"
fi
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
if [[ -s "${OUTPUT_DIR}/training_summary.json" && "${FORCE_RERUN}" != "1" ]]; then
  echo "Completed H10 experiment already exists; skipped: ${OUTPUT_DIR}"
  exit 0
fi
RESUME_ARGS=()
if [[ -s "${OUTPUT_DIR}/latest_resume.pth" && "${FORCE_RERUN}" != "1" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT_DIR}/latest_resume.pth")
fi

echo "================================================================================"
echo "Experiment       : ${EXP_NAME}"
echo "Variant          : ${VARIANT}"
echo "Source           : ${SOURCE_CKPT}"
echo "P3 localization  : frozen"
echo "H6-A2 base       : frozen"
case "${VARIANT}" in
  h2) AFFECTED_HEAD="matched-2D" ;;
  h3|h4) AFFECTED_HEAD="Lite-3D-STA" ;;
  *) AFFECTED_HEAD="disabled" ;;
esac
if [[ "${VARIANT}" == "h4" ]]; then
  SEVERITY_HEAD="conditional-2D"
else
  SEVERITY_HEAD="frozen-A2"
fi
echo "Affected head    : ${AFFECTED_HEAD}"
echo "Severity head    : ${SEVERITY_HEAD}"
echo "Objective        : one hierarchical building-damage NLL"
echo "Extra losses     : NONE"
echo "Test selection   : NEVER"
echo "Output           : ${OUTPUT_DIR}"
echo "Steps/EBS        : ${OPTIMIZER_STEPS}/$(( ${TRAIN_BATCH_SIZE:-4} * ${ACCUMULATION_STEPS:-4} ))"
echo "================================================================================"

python -u script/standard_ML/train_H10_H3D.py \
  --variant "${VARIANT}" \
  --source_h3_checkpoint "${SOURCE_CKPT}" \
  --train_dataset_path "${DATASET_ROOT}" --train_data_list_path "${TRAIN_LIST}" \
  --val_dataset_path "${DATASET_ROOT}" --val_data_list_path "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" --dataset_variant h3 \
  --output_dir "${OUTPUT_DIR}" --device cuda:0 --crop_size 640 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
  --optimizer_steps "${OPTIMIZER_STEPS}" --new_module_warmup_steps 0 \
  --eval_every "${EVAL_EVERY}" --save_every "${SAVE_EVERY}" \
  --log_every "${LOG_EVERY:-20}" --num_workers "${NUM_WORKERS:-8}" \
  --val_max_batches "${VAL_MAX_BATCHES}" \
  --new_module_learning_rate "${NEW_LR:-1e-4}" \
  --sar_learning_rate 0 --core_learning_rate 0 \
  --weight_decay "${WEIGHT_DECAY:-5e-3}" \
  --damage_ce_weight 0 --damage_lovasz_weight 0 \
  --sar_aux_weight 0 --representation_weight 0 \
  --flow_regularization_weight 0 --hierarchy_weight 0 \
  --extra_supervised_weight 1 --prior_strength 0 \
  --amp "${AMP:-bf16}" --seed "${SEED}" \
  --save_secondary_best \
  "${VALIDATION_ARGS[@]}" "${RESUME_ARGS[@]}"

if [[ "${MODE}" == "guard" ]]; then
  python - "${SOURCE_CKPT}" "${OUTPUT_DIR}/initial_validation.json" <<'PY'
import json, sys, torch
source = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
expected = source.get("metrics", source.get("best_metrics", {}))
observed = json.load(open(sys.argv[2], encoding="utf-8"))
for key in ("valid_miou", "miou4", "damaged_iou", "destroyed_iou", "loc_f1"):
    delta = abs(float(observed[key]) - float(expected[key]))
    print(f"{key}: observed={observed[key]:.8f}, expected={expected[key]:.8f}, delta={delta:.3g}")
    if delta > 2e-6:
        raise SystemExit(f"H10 exact source guard failed for {key}: {delta}")
print("H10 exact source guard passed.")
PY
fi
