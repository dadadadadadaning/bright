#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
VARIANT_WAS_SET="${VARIANT+x}"
VARIANT="${VARIANT:-a2}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
TEST_LIST="${TEST_LIST:-${DATASET_ROOT}/splitname/standard_ML/test_set.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H6_SDCR}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
DAMAGE_PRIORS_JSON="${DAMAGE_PRIORS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/H4_RGBSAR/damage_train_priors.json}"
SOURCE_A4_CKPT="${SOURCE_A4_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H5_BIQ/source/H4_A4_step5500_valid602784_locked.pth}"
P3_RGB_CKPT="${P3_RGB_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
H6_A2_CKPT="${H6_A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
EXPECTED_P3_SHA256="${EXPECTED_P3_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"
EXP_NAME="${EXP_NAME:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"
ENABLE_TEST_MONITOR="${ENABLE_TEST_MONITOR:-}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${MODE}" in
  preflight)
    bash -n run/run_H6_SDCR_5090.sh
    bash -n run/run_H6_A2_DG_5090.sh
    bash -n run/run_H6_A2_DG_ablation_sequence.sh
    bash -n run/run_H6_A2_freeze_ablation.sh
    bash -n run/run_H6_SAR_stage_probe.sh
    bash -n run/run_H6_SDCR_sequence.sh
    bash -n run/run_H6B_safe_residual_sequence.sh
    bash -n run/run_H6B1R_sparse_router_sequence.sh
    python -m py_compile \
      dataset/make_data_loader_heterogeneous_rgbsar.py \
      model/DamageFormer_RGBSAR_H6_SDCR.py \
      script/standard_ML/build_RGBSAR_event_holdout_folds.py \
      script/standard_ML/audit_bright_split_domain.py \
      script/standard_ML/audit_H6_SAR_stage_probes.py \
      script/standard_ML/summarize_H6_A2_DG_ablation.py \
      script/standard_ML/train_DamageFormer_RGBSAR_H4.py \
      script/standard_ML/train_DamageFormer_RGBSAR_H6_SDCR.py \
      script/standard_ML/eval_DamageFormer_RGBSAR_H6_SDCR.py \
      script/standard_ML/summarize_H6_SDCR_results.py \
      script/standard_ML/summarize_H6B_rounds.py \
      script/standard_ML/summarize_H6B1R_seeds.py
    python -c \
      "from bda_benchmark.script.standard_ML.eval_DamageFormer_RGBSAR_H6_SDCR import main; print('H6 evaluator import passed.')"
    python -m unittest -v \
      tests.test_damageformer_rgbsar_h6_sdcr \
      tests.test_h6_sar_stage_probe \
      tests.test_h6_a2_dg_protocol \
      tests.test_rgbsar_event_holdout_folds
    echo "H6-SDCR preflight passed."
    exit 0
    ;;
  summary)
    mkdir -p "${OUTPUT_ROOT}"
    python -u script/standard_ML/summarize_H6_SDCR_results.py \
      --output_root "${OUTPUT_ROOT}" \
      --source_a4_checkpoint "${SOURCE_A4_CKPT}" \
      --p3_checkpoint "${P3_RGB_CKPT}" \
      --output_json "${OUTPUT_ROOT}/H6_SDCR_validation_comparison.json"
    exit 0
    ;;
  eval_val|eval_test)
    if [[ "${MODE}" == "eval_test" ]]; then
      EVAL_SPLIT="test"
      EVAL_LIST="${TEST_LIST}"
    else
      EVAL_SPLIT="val"
      EVAL_LIST="${VAL_LIST}"
    fi
    H6_CKPT="${H6_CKPT:-}"
    test -n "${H6_CKPT}" || {
      echo "${MODE} requires H6_CKPT=/path/to/best_val_miou.pth" >&2
      exit 1
    }
    test -s "${H6_CKPT}" || {
      echo "Missing H6 checkpoint: ${H6_CKPT}" >&2
      exit 1
    }
    for required in "${SAR_STATS_JSON}" "${EVAL_LIST}"; do
      test -s "${required}" || {
        echo "Missing validation input: ${required}" >&2
        exit 1
      }
    done
    EVAL_OUTPUT_JSON="${EVAL_OUTPUT_JSON:-${H6_CKPT%/*}/independent_${EVAL_SPLIT}.json}"
    python -u script/standard_ML/eval_DamageFormer_RGBSAR_H6_SDCR.py \
      --checkpoint "${H6_CKPT}" \
      --split "${EVAL_SPLIT}" \
      --dataset_path "${DATASET_ROOT}" \
      --data_list_path "${EVAL_LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" \
      --output_json "${EVAL_OUTPUT_JSON}" \
      --device cuda:0 \
      --crop_size 640 \
      --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
      --num_workers "${NUM_WORKERS:-8}" \
      --seed "${SEED}"
    exit 0
    ;;
  guard)
    if [[ -z "${VARIANT_WAS_SET}" ]]; then VARIANT="a0"; fi
    OPTIMIZER_STEPS=0; EVAL_EVERY=1; SAVE_EVERY=1; VAL_MAX_BATCHES=0
    WARMUP_STEPS=0; TAG="guard"; VALIDATION_ARGS=(--validation_only)
    ;;
  anchor_guard)
    VARIANT="a1"
    OPTIMIZER_STEPS=0; EVAL_EVERY=1; SAVE_EVERY=1; VAL_MAX_BATCHES=0
    WARMUP_STEPS=0; TAG="anchor_guard"; VALIDATION_ARGS=(--validation_only)
    ;;
  safe_guard)
    VARIANT="b0"
    OPTIMIZER_STEPS=0; EVAL_EVERY=1; SAVE_EVERY=1; VAL_MAX_BATCHES=0
    TEST_MAX_BATCHES=0
    WARMUP_STEPS=0; TAG="safe_guard"; VALIDATION_ARGS=(--validation_only)
    ;;
  smoke)
    OPTIMIZER_STEPS=20; EVAL_EVERY=20; SAVE_EVERY=20
    VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-2}"
    WARMUP_STEPS="$([[ ${VARIANT} == a1 || ${VARIANT} == b0 ]] && echo 0 || echo 5)"
    TEST_MAX_BATCHES="${TEST_MAX_BATCHES:-2}"
    TAG="smoke20"; VALIDATION_ARGS=()
    ;;
  pilot)
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-3000}"
    EVAL_EVERY="${EVAL_EVERY:-500}"; SAVE_EVERY="${SAVE_EVERY:-1000}"
    VAL_MAX_BATCHES=0
    WARMUP_STEPS="$([[ ${VARIANT} == a1 || ${VARIANT} == b0 ]] && echo 0 || echo "${WARMUP_STEPS:-1000}")"
    TEST_MAX_BATCHES="${TEST_MAX_BATCHES:-0}"
    TAG="pilot${OPTIMIZER_STEPS}"; VALIDATION_ARGS=()
    ;;
  full)
    OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-8000}"
    EVAL_EVERY="${EVAL_EVERY:-500}"; SAVE_EVERY="${SAVE_EVERY:-1000}"
    VAL_MAX_BATCHES=0
    WARMUP_STEPS="$([[ ${VARIANT} == a1 || ${VARIANT} == b0 ]] && echo 0 || echo "${WARMUP_STEPS:-1000}")"
    TEST_MAX_BATCHES="${TEST_MAX_BATCHES:-0}"
    TAG="full${OPTIMIZER_STEPS}"; VALIDATION_ARGS=()
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|guard|anchor_guard|safe_guard|smoke|pilot|full|eval_val|eval_test|summary" >&2
    exit 2
    ;;
esac

case "${VARIANT}" in a0|a1|a2|a3|b0|b1|b2|b1r) ;; *) echo "Unknown H6 variant: ${VARIANT}" >&2; exit 2;; esac
for required in "${SOURCE_A4_CKPT}" "${SAR_STATS_JSON}" "${DAMAGE_PRIORS_JSON}" "${TRAIN_LIST}" "${VAL_LIST}"; do
  test -s "${required}" || { echo "Missing required H6 input: ${required}" >&2; exit 1; }
done
if [[ "${VARIANT}" != "a0" ]]; then
  test -s "${P3_RGB_CKPT}" || { echo "Missing P3 RGB checkpoint: ${P3_RGB_CKPT}" >&2; exit 1; }
  ACTUAL_P3_SHA256="$(sha256sum "${P3_RGB_CKPT}" | awk '{print $1}')"
  [[ "${ACTUAL_P3_SHA256}" == "${EXPECTED_P3_SHA256}" ]] || {
    echo "P3 SHA256 mismatch: ${ACTUAL_P3_SHA256}" >&2; exit 1;
  }
fi
if [[ "${VARIANT}" == b1 || "${VARIANT}" == b2 || "${VARIANT}" == b1r ]]; then
  test -s "${H6_A2_CKPT}" || {
    echo "Missing locked H6-A2 native expert: ${H6_A2_CKPT}" >&2
    exit 1
  }
fi

if [[ -z "${EXP_NAME}" ]]; then EXP_NAME="H6_${VARIANT^^}_${TAG}_seed${SEED}"; fi
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
if [[ -s "${OUTPUT_DIR}/training_summary.json" && "${FORCE_RERUN}" != "1" ]]; then
  echo "Completed H6 experiment already exists; skipped: ${OUTPUT_DIR}"; exit 0
fi
RESUME_ARGS=()
if [[ -s "${OUTPUT_DIR}/latest_resume.pth" && "${FORCE_RERUN}" != "1" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT_DIR}/latest_resume.pth")
fi
P3_ARGS=()
if [[ "${VARIANT}" != "a0" ]]; then
  P3_ARGS=(--joint_pretrained_checkpoint "${P3_RGB_CKPT}")
fi
A2_ARGS=()
if [[ "${VARIANT}" == b1 || "${VARIANT}" == b2 || "${VARIANT}" == b1r ]]; then
  A2_ARGS=(--sar_pretrained_checkpoint "${H6_A2_CKPT}")
fi
TEST_MONITOR_ARGS=()
if [[ -z "${ENABLE_TEST_MONITOR}" ]]; then
  if [[ "${VARIANT}" == b0 || "${VARIANT}" == b1 || "${VARIANT}" == b2 ]]; then
    ENABLE_TEST_MONITOR=1
  else
    ENABLE_TEST_MONITOR=0
  fi
fi
if [[ "${ENABLE_TEST_MONITOR}" == 1 ]]; then
  test -s "${TEST_LIST}" || {
    echo "Missing test list: ${TEST_LIST}" >&2
    exit 1
  }
  TEST_MONITOR_ARGS=(
    --test_dataset_path "${DATASET_ROOT}"
    --test_data_list_path "${TEST_LIST}"
    --test_every_validation
    --test_max_batches "${TEST_MAX_BATCHES:-0}"
  )
fi
LOSS_ARGS=()
DATASET_VARIANT_ARGS=()
SAMPLING_ARGS=(
  --focus_probability "${FOCUS_PROBABILITY:-0}"
  --focus_class_weights \
    "${FOCUS_BACKGROUND_WEIGHT:-0}" \
    "${FOCUS_INTACT_WEIGHT:-1}" \
    "${FOCUS_DAMAGED_WEIGHT:-2}" \
    "${FOCUS_DESTROYED_WEIGHT:-2}"
)
SELECTION_ARGS=()
if [[ "${SAVE_SECONDARY_BEST:-0}" == "1" ]]; then
  SELECTION_ARGS=(--save_secondary_best)
fi
export H6_EVENT_BALANCE_POWER="${EVENT_BALANCE_POWER:-0}"
export H6_SAR_FREEZE_MODE="${SAR_FREEZE_MODE:-f0_shallow_frozen}"
if [[ "${VARIANT}" == b0 || "${VARIANT}" == b1 || "${VARIANT}" == b2 || "${VARIANT}" == b1r ]]; then
  DATASET_VARIANT_ARGS=(--dataset_variant h4)
  LOSS_ARGS=(
    --sar_aux_weight 0
    --representation_weight 0
    --flow_regularization_weight 0
    --hierarchy_weight 0
  )
fi
case "${VARIANT}" in
  a2|a3|b2) SENSOR_PROTOTYPE_LABEL="stage3+stage4" ;;
  b1|b1r) SENSOR_PROTOTYPE_LABEL="stage4" ;;
  *) SENSOR_PROTOTYPE_LABEL="disabled" ;;
esac

echo "================================================================================"
echo "Experiment       : ${EXP_NAME}"
echo "Variant          : ${VARIANT}"
echo "H4-A4 source     : ${SOURCE_A4_CKPT}"
echo "P3 RGB anchor    : $([[ ${VARIANT} == a0 ]] && echo disabled || echo "${P3_RGB_CKPT}")"
echo "P3 full anchor   : $([[ ${VARIANT} == b0 || ${VARIANT} == b1 || ${VARIANT} == b2 || ${VARIANT} == b1r ]] && echo frozen-exact || echo disabled)"
echo "A2 native expert : $([[ ${VARIANT} == b1 || ${VARIANT} == b2 || ${VARIANT} == b1r ]] && echo "${H6_A2_CKPT}" || echo disabled)"
echo "Sensor prototype : ${SENSOR_PROTOTYPE_LABEL}"
echo "Frequency calib. : $([[ ${VARIANT} == a3 ]] && echo enabled || echo disabled)"
echo "Teacher/KD       : NONE / NONE"
echo "Residual router  : $([[ ${VARIANT} == b1r ]] && echo class-conditional-sparse-router-only || echo legacy)"
echo "Test monitoring  : $([[ ${ENABLE_TEST_MONITOR} == 1 ]] && echo every-validation-log-only || echo disabled-during-development)"
echo "Event balance    : power=${H6_EVENT_BALANCE_POWER}"
echo "SAR freeze mode  : ${H6_SAR_FREEZE_MODE}"
echo "Damage crop focus: p=${FOCUS_PROBABILITY:-0}; weights=${FOCUS_BACKGROUND_WEIGHT:-0}/${FOCUS_INTACT_WEIGHT:-1}/${FOCUS_DAMAGED_WEIGHT:-2}/${FOCUS_DESTROYED_WEIGHT:-2}"
echo "Val selectors    : valid$([[ ${SAVE_SECONDARY_BEST:-0} == 1 ]] && echo '+event+balance' || true)"
echo "RGB/Loc/BN       : FROZEN / FROZEN / FROZEN"
echo "Output           : ${OUTPUT_DIR}"
echo "Steps/EBS        : ${OPTIMIZER_STEPS}/$(( ${TRAIN_BATCH_SIZE:-4} * ${ACCUMULATION_STEPS:-4} ))"
echo "================================================================================"

python -u script/standard_ML/train_DamageFormer_RGBSAR_H6_SDCR.py \
  --variant "${VARIANT}" \
  --source_h3_checkpoint "${SOURCE_A4_CKPT}" \
  --damage_priors_json "${DAMAGE_PRIORS_JSON}" \
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
  --new_module_warmup_steps "${WARMUP_STEPS}" \
  --eval_every "${EVAL_EVERY}" --save_every "${SAVE_EVERY}" \
  --log_every "${LOG_EVERY:-20}" --num_workers "${NUM_WORKERS:-8}" \
  --val_max_batches "${VAL_MAX_BATCHES}" \
  --new_module_learning_rate "${NEW_LR:-1e-4}" \
  --sar_learning_rate "${SAR_LR:-1e-5}" \
  --core_learning_rate "${CORE_LR:-1e-6}" \
  --weight_decay "${WEIGHT_DECAY:-5e-3}" --amp "${AMP:-bf16}" --seed "${SEED}" \
  "${P3_ARGS[@]}" "${A2_ARGS[@]}" "${DATASET_VARIANT_ARGS[@]}" \
  "${LOSS_ARGS[@]}" "${SAMPLING_ARGS[@]}" "${SELECTION_ARGS[@]}" \
  "${TEST_MONITOR_ARGS[@]}" \
  "${VALIDATION_ARGS[@]}" "${RESUME_ARGS[@]}"

if [[ "${ENABLE_TEST_MONITOR}" == 1 ]]; then
  python -u script/standard_ML/summarize_H6B_rounds.py \
    --experiment_dir "${OUTPUT_DIR}" \
    --output_json "${OUTPUT_DIR}/round_comparison.json"
fi

if [[ "${MODE}" == "guard" ]]; then
  python - "${SOURCE_A4_CKPT}" "${OUTPUT_DIR}/initial_validation.json" <<'PY'
import json, sys, torch
source = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
observed = json.load(open(sys.argv[2], encoding="utf-8"))
expected = source.get("metrics", source.get("best_metrics", {}))
for key in ("valid_miou", "miou4", "damaged_iou", "destroyed_iou", "loc_f1"):
    delta = abs(float(observed[key]) - float(expected[key]))
    print(f"{key}: observed={observed[key]:.8f}, expected={expected[key]:.8f}, delta={delta:.3g}")
    if delta > 5e-5: raise SystemExit(f"H6 A0 guard mismatch for {key}")
print("H6 A0 full-validation guard passed.")
PY
fi

if [[ "${MODE}" == "anchor_guard" ]]; then
  python - "${P3_RGB_CKPT}" "${OUTPUT_DIR}/initial_validation.json" <<'PY'
import json, sys, torch
source = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
observed = json.load(open(sys.argv[2], encoding="utf-8"))
raw = source.get("metrics", source.get("best_metrics", {}))
expected = dict(raw)
localization = raw.get("localization", {})
confusion = localization.get("confusion") if isinstance(localization, dict) else None
if confusion:
    tn, fp = map(float, confusion[0])
    fn, tp = map(float, confusion[1])
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    expected.update({
        "loc_precision": precision,
        "loc_recall": recall,
        "loc_f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "loc_iou": tp / max(tp + fp + fn, 1.0),
    })
checked = 0
for key in ("loc_f1", "loc_iou", "loc_precision", "loc_recall"):
    if key not in expected: continue
    checked += 1
    delta = abs(float(observed[key]) - float(expected[key]))
    print(f"{key}: observed={observed[key]:.8f}, expected={expected[key]:.8f}, delta={delta:.3g}")
    if delta > 5e-5: raise SystemExit(f"H6 A1 P3 anchor mismatch for {key}")
if not checked: raise SystemExit("P3 checkpoint has no usable localization metrics")
print("H6 A1 P3 RGB spatial-anchor guard passed.")
PY
fi

if [[ "${MODE}" == "safe_guard" ]]; then
  python - "${P3_RGB_CKPT}" "${OUTPUT_DIR}/initial_validation.json" <<'PY'
import json, sys, torch

source = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
observed = json.load(open(sys.argv[2], encoding="utf-8"))
raw = source.get("metrics", source.get("best_metrics", {}))
iou = raw.get("iou")
if not isinstance(iou, (list, tuple)) or len(iou) != 4:
    raise SystemExit("P3 checkpoint has no four-class IoU vector")
expected = {
    "miou4": sum(map(float, iou)) / 400.0,
    "valid_miou": sum(map(float, iou[1:])) / 300.0,
    "intact_iou": float(iou[1]) / 100.0,
    "damaged_iou": float(iou[2]) / 100.0,
    "destroyed_iou": float(iou[3]) / 100.0,
}
for key, value in expected.items():
    delta = abs(float(observed[key]) - value)
    print(
        f"{key}: observed={observed[key]:.8f}, "
        f"expected={value:.8f}, delta={delta:.3g}"
    )
    if delta > 5e-5:
        raise SystemExit(f"H6-B0 exact P3 guard mismatch for {key}")
print("H6-B0 exact P3 validation guard passed.")
print("Inspect initial test monitor for the expected P3 test mIoU4=0.672725.")
PY
fi
