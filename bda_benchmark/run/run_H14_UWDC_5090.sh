#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
VARIANT="${VARIANT:-u1}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
TEST_LIST="${TEST_LIST:-${DATASET_ROOT}/splitname/standard_ML/test_set.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H14_UWDC}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H14_UWDC}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
H6_A2_CKPT="${H6_A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
EXP_NAME="${EXP_NAME:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"

export H14_EVENT_BALANCE="${H14_EVENT_BALANCE:-1}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${MODE}" in
  preflight)
    bash -n run/run_H14_UWDC_5090.sh
    bash -n run/run_H14_UWDC_sequence.sh
    python -m py_compile \
      model/DamageFormer_RGBSAR_H14_UWDC.py \
      script/standard_ML/h14_uwdc_runtime.py \
      script/standard_ML/train_H14_UWDC.py \
      script/standard_ML/eval_H14_UWDC.py \
      script/standard_ML/audit_H14_unique_evidence.py
    python -m unittest -v tests.test_h14_uwdc
    echo "H14-UWDC preflight passed."
    exit 0
    ;;
  audit_smoke|audit_val)
    for required in "${H6_A2_CKPT}" "${SAR_STATS_JSON}" "${VAL_LIST}"; do
      test -s "${required}" || { echo "Missing H14 audit input: ${required}" >&2; exit 1; }
    done
    if [[ "${MODE}" == "audit_smoke" ]]; then
      AUDIT_MAX_BATCHES="${AUDIT_MAX_BATCHES:-4}"; AUDIT_TAG="smoke"
    else
      AUDIT_MAX_BATCHES=0; AUDIT_TAG="val"
    fi
    AUDIT_ROOT="${AUDIT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/H14_UWDC}"
    AUDIT_OUTPUT_JSON="${AUDIT_OUTPUT_JSON:-${AUDIT_ROOT}/H14_unique_evidence_${AUDIT_TAG}.json}"
    mkdir -p "${AUDIT_ROOT}"
    python -u script/standard_ML/audit_H14_unique_evidence.py \
      --checkpoint "${H6_A2_CKPT}" --dataset_path "${DATASET_ROOT}" \
      --data_list_path "${VAL_LIST}" --sar_stats_path "${SAR_STATS_JSON}" \
      --output_json "${AUDIT_OUTPUT_JSON}" --device cuda:0 --crop_size 640 \
      --eval_batch_size "${EVAL_BATCH_SIZE:-1}" --num_workers "${NUM_WORKERS:-8}" \
      --max_batches "${AUDIT_MAX_BATCHES}" --seed "${SEED}"
    exit 0
    ;;
  guard)
    OPTIMIZER_STEPS=0; EVAL_EVERY=1; SAVE_EVERY=1
    VAL_MAX_BATCHES=0; TAG="guard"; VARIANT=u0
    VALIDATION_ARGS=(--validation_only)
    ;;
  smoke)
    case "${VARIANT}" in u1|u2|u3) ;; *) echo "Smoke supports u1|u2|u3" >&2; exit 2;; esac
    OPTIMIZER_STEPS=20; EVAL_EVERY=20; SAVE_EVERY=20
    VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-2}"; TAG="smoke20"
    VALIDATION_ARGS=()
    ;;
  screen|train)
    case "${VARIANT}" in u1|u2|u3) ;; *) echo "Screen/train supports u1|u2|u3" >&2; exit 2;; esac
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-1500}"
    EVAL_EVERY="${EVAL_EVERY:-500}"
    SAVE_EVERY="${SAVE_EVERY:-1500}"
    VAL_MAX_BATCHES=0; TAG="screen${OPTIMIZER_STEPS}"
    VALIDATION_ARGS=()
    ;;
  eval_val|eval_test)
    if [[ "${MODE}" == "eval_test" ]]; then
      EVAL_SPLIT="test"; EVAL_LIST="${TEST_LIST}"
    else
      EVAL_SPLIT="val"; EVAL_LIST="${VAL_LIST}"
    fi
    H14_CKPT="${H14_CKPT:-}"
    test -n "${H14_CKPT}" || { echo "${MODE} requires H14_CKPT" >&2; exit 1; }
    for required in "${H14_CKPT}" "${SAR_STATS_JSON}" "${EVAL_LIST}"; do
      test -s "${required}" || { echo "Missing H14 evaluation input: ${required}" >&2; exit 1; }
    done
    EVAL_OUTPUT_JSON="${EVAL_OUTPUT_JSON:-${H14_CKPT%/*}/independent_${EVAL_SPLIT}.json}"
    BASELINE_ARGS=()
    if [[ -n "${H14_BASELINE_JSON:-}" && -s "${H14_BASELINE_JSON}" ]]; then
      BASELINE_ARGS=(--baseline_metrics_json "${H14_BASELINE_JSON}")
    fi
    python -u script/standard_ML/eval_H14_UWDC.py \
      --checkpoint "${H14_CKPT}" --split "${EVAL_SPLIT}" \
      --dataset_path "${DATASET_ROOT}" --data_list_path "${EVAL_LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" --output_json "${EVAL_OUTPUT_JSON}" \
      --device cuda:0 --crop_size 640 --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
      --num_workers "${NUM_WORKERS:-8}" --seed "${SEED}" "${BASELINE_ARGS[@]}"
    exit 0
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|audit_smoke|audit_val|guard|smoke|screen|train|eval_val|eval_test" >&2
    exit 2
    ;;
esac

for required in "${H6_A2_CKPT}" "${SAR_STATS_JSON}" "${TRAIN_LIST}" "${VAL_LIST}"; do
  test -s "${required}" || { echo "Missing required H14 input: ${required}" >&2; exit 1; }
done
if [[ -z "${EXP_NAME}" ]]; then
  EXP_NAME="H14_${VARIANT^^}_${TAG}_seed${SEED}"
fi
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
if [[ -s "${OUTPUT_DIR}/training_summary.json" && "${FORCE_RERUN}" != "1" ]]; then
  echo "Completed H14 experiment already exists; skipped: ${OUTPUT_DIR}"
  exit 0
fi
RESUME_ARGS=()
if [[ -s "${OUTPUT_DIR}/latest_resume.pth" && "${FORCE_RERUN}" != "1" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT_DIR}/latest_resume.pth")
fi

echo "================================================================================"
echo "Experiment       : ${EXP_NAME}"
echo "Variant          : ${VARIANT}"
echo "Source           : ${H6_A2_CKPT}"
echo "H6-A2 base       : frozen"
echo "Evidence         : common gap + separate RGB/SAR unique"
echo "Superpixel/graph : NONE / NONE"
echo "Selector         : event-worst -> event-mean -> Damaged -> valid"
echo "Test selection   : NEVER"
echo "Output           : ${OUTPUT_DIR}"
echo "Steps/EBS        : ${OPTIMIZER_STEPS}/$(( ${TRAIN_BATCH_SIZE:-4} * ${ACCUMULATION_STEPS:-4} ))"
echo "================================================================================"

python -u script/standard_ML/train_H14_UWDC.py \
  --variant "${VARIANT}" --source_h3_checkpoint "${H6_A2_CKPT}" \
  --train_dataset_path "${DATASET_ROOT}" --train_data_list_path "${TRAIN_LIST}" \
  --val_dataset_path "${DATASET_ROOT}" --val_data_list_path "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" --dataset_variant h3 \
  --output_dir "${OUTPUT_DIR}" --device cuda:0 --crop_size 640 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-4}" --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
  --optimizer_steps "${OPTIMIZER_STEPS}" --new_module_warmup_steps 0 \
  --eval_every "${EVAL_EVERY}" --save_every "${SAVE_EVERY}" \
  --log_every "${LOG_EVERY:-20}" --num_workers "${NUM_WORKERS:-8}" \
  --val_max_batches "${VAL_MAX_BATCHES}" --new_module_learning_rate "${NEW_LR:-1e-4}" \
  --sar_learning_rate 0 --core_learning_rate 0 --weight_decay "${WEIGHT_DECAY:-5e-3}" \
  --focus_probability "${FOCUS_PROBABILITY:-0.5}" --focus_class_weights 0 1 4 2 \
  --damage_ce_weight 0 --damage_lovasz_weight 0 --sar_aux_weight 0 \
  --representation_weight 0 --flow_regularization_weight 0 --hierarchy_weight 0 \
  --extra_supervised_weight 1 --prior_strength 0 --amp "${AMP:-off}" \
  --seed "${SEED}" --no_pretrained_rgb --save_secondary_best \
  "${VALIDATION_ARGS[@]}" "${RESUME_ARGS[@]}"

if [[ "${MODE}" == "guard" ]]; then
  python - "${H6_A2_CKPT}" "${OUTPUT_DIR}/initial_validation.json" <<'PY'
import json, sys, torch
source = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
expected = source.get("metrics", source.get("best_metrics", {}))
observed = json.load(open(sys.argv[2], encoding="utf-8"))
for key in ("valid_miou", "miou4", "damaged_iou", "destroyed_iou", "loc_f1"):
    delta = abs(float(observed[key]) - float(expected[key]))
    print(f"{key}: observed={observed[key]:.8f}, expected={expected[key]:.8f}, delta={delta:.3g}")
    if delta > 2e-6:
        raise SystemExit(f"H14 exact source guard failed for {key}: {delta}")
print("H14-U0 exact H6-A2 guard passed.")
PY
fi
