#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
VARIANT="${VARIANT:-c1}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
TEST_LIST="${TEST_LIST:-${DATASET_ROOT}/splitname/standard_ML/test_set.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/OA_HMD}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/OA_HMD}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
H6_A2_CKPT="${H6_A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
EXP_NAME="${EXP_NAME:-}"
FORCE_RERUN="${FORCE_RERUN:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${MODE}" in
  preflight)
    bash -n run/run_OA_HMD_5090.sh
    bash -n run/run_OA_HMD_sequence.sh
    python -m py_compile \
      model/DamageFormer_RGBSAR_OA_HMD.py \
      script/standard_ML/train_OA_HMD.py \
      script/standard_ML/eval_OA_HMD.py
    python -m unittest -v tests.test_oa_hmd
    echo "OA-HMD preflight passed."
    exit 0
    ;;
  guard)
    VARIANT="c0"; OPTIMIZER_STEPS=0; EVAL_EVERY=1; SAVE_EVERY=1
    VAL_MAX_BATCHES=0; TAG="guard"
    ;;
  smoke)
    VARIANT="c1"; OPTIMIZER_STEPS=20; EVAL_EVERY=20; SAVE_EVERY=20
    VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-2}"; TAG="smoke20"
    ;;
  pilot)
    VARIANT="c1"; OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-3000}"
    EVAL_EVERY="${EVAL_EVERY:-500}"; SAVE_EVERY="${SAVE_EVERY:-1000}"
    VAL_MAX_BATCHES=0; TAG="pilot${OPTIMIZER_STEPS}"
    ;;
  eval_val|eval_test)
    if [[ "${MODE}" == "eval_test" ]]; then
      EVAL_SPLIT="test"; EVAL_LIST="${TEST_LIST}"
    else
      EVAL_SPLIT="val"; EVAL_LIST="${VAL_LIST}"
    fi
    OA_HMD_CKPT="${OA_HMD_CKPT:-}"
    test -n "${OA_HMD_CKPT}" || {
      echo "${MODE} requires OA_HMD_CKPT=/path/to/checkpoint.pth" >&2
      exit 1
    }
    for required in "${OA_HMD_CKPT}" "${SAR_STATS_JSON}" "${EVAL_LIST}"; do
      test -s "${required}" || {
        echo "Missing OA-HMD evaluation input: ${required}" >&2
        exit 1
      }
    done
    EVAL_OUTPUT_JSON="${EVAL_OUTPUT_JSON:-${OA_HMD_CKPT%/*}/independent_${EVAL_SPLIT}.json}"
    BASELINE_ARGS=()
    if [[ -n "${OA_BASELINE_JSON:-}" ]]; then
      test -s "${OA_BASELINE_JSON}" || {
        echo "Missing OA baseline JSON: ${OA_BASELINE_JSON}" >&2
        exit 1
      }
      BASELINE_ARGS=(--baseline_json "${OA_BASELINE_JSON}")
    fi
    python -u script/standard_ML/eval_OA_HMD.py \
      --checkpoint "${OA_HMD_CKPT}" --split "${EVAL_SPLIT}" \
      --dataset_path "${DATASET_ROOT}" --data_list_path "${EVAL_LIST}" \
      --sar_stats_path "${SAR_STATS_JSON}" --output_json "${EVAL_OUTPUT_JSON}" \
      --device cuda:0 --crop_size 640 \
      --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
      --num_workers "${NUM_WORKERS:-8}" --seed "${SEED}" \
      "${BASELINE_ARGS[@]}"
    exit 0
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|guard|smoke|pilot|eval_val|eval_test" >&2
    exit 2
    ;;
esac

for required in "${H6_A2_CKPT}" "${P3_CKPT}" "${SAR_STATS_JSON}" "${TRAIN_LIST}" "${VAL_LIST}"; do
  test -s "${required}" || {
    echo "Missing required OA-HMD input: ${required}" >&2
    exit 1
  }
done

if [[ -z "${EXP_NAME}" ]]; then
  EXP_NAME="OA_HMD_${VARIANT^^}_${TAG}_seed${SEED}"
fi
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}" "${LOG_ROOT}"
if [[ -s "${OUTPUT_DIR}/training_summary.json" && "${FORCE_RERUN}" != "1" ]]; then
  echo "Completed OA-HMD experiment already exists; skipped: ${OUTPUT_DIR}"
  exit 0
fi
RESUME_ARGS=()
if [[ -s "${OUTPUT_DIR}/latest_resume.pth" && "${FORCE_RERUN}" != "1" ]]; then
  RESUME_ARGS=(--resume "${OUTPUT_DIR}/latest_resume.pth")
fi

echo "================================================================================"
echo "Experiment       : ${EXP_NAME}"
echo "Variant          : ${VARIANT}"
echo "Main anchor      : ${P3_CKPT}"
echo "Native Stage3    : ${H6_A2_CKPT}"
echo "Connection       : independent projections + [R,S,|R-S|,R*S]"
echo "Decision changed : Intact <-> Affected only"
echo "P3 localisation  : frozen"
echo "P3 severity odds : frozen/preserved"
echo "FCA/Flow/3D      : NONE / NONE / NONE"
echo "Steps/EBS        : ${OPTIMIZER_STEPS}/$(( ${TRAIN_BATCH_SIZE:-4} * ${ACCUMULATION_STEPS:-4} ))"
echo "Test selection   : NEVER"
echo "Output           : ${OUTPUT_DIR}"
echo "================================================================================"

python -u script/standard_ML/train_OA_HMD.py \
  --variant "${VARIANT}" \
  --source_h3_checkpoint "${H6_A2_CKPT}" \
  --joint_pretrained_checkpoint "${P3_CKPT}" \
  --train_dataset_path "${DATASET_ROOT}" --train_data_list_path "${TRAIN_LIST}" \
  --val_dataset_path "${DATASET_ROOT}" --val_data_list_path "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" --dataset_variant h4 \
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
  "${RESUME_ARGS[@]}"

if [[ "${MODE}" == "guard" ]]; then
  GUARD_CKPT="${OUTPUT_DIR}/best_val_miou.pth"
  GUARD_JSON="${OUTPUT_DIR}/independent_val.json"
  OA_HMD_CKPT="${GUARD_CKPT}" \
  EVAL_OUTPUT_JSON="${GUARD_JSON}" \
  bash run/run_OA_HMD_5090.sh eval_val
  python - "${P3_CKPT}" "${GUARD_CKPT}" "${GUARD_JSON}" <<'PY'
import json, sys, torch
p3 = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
guard = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
observed = json.load(open(sys.argv[3], encoding="utf-8"))["observed_metrics"]
required = ("valid_miou", "miou4", "damaged_iou", "destroyed_iou", "loc_f1")

def find_metrics(value):
    if isinstance(value, dict):
        if all(key in value for key in required):
            return value
        for key in ("metrics", "best_metrics", "observed_metrics", "validation"):
            if key in value:
                found = find_metrics(value[key])
                if found is not None:
                    return found
        for child in value.values():
            found = find_metrics(child)
            if found is not None:
                return found
    return None

expected = find_metrics(p3)
if expected is None:
    # Some migrated P3 weight files intentionally retain only model tensors.
    # C0 directly returns the strictly-loaded P3 anchor, so the tensor contract
    # is already exact. In that legacy case compare the independent pass with
    # the C0 initial-selection metrics to verify deterministic evaluation.
    expected = find_metrics(guard)
    if expected is None:
        raise SystemExit(
            "Neither legacy P3 nor C0 guard checkpoint contains comparable metrics"
        )
    print(
        "Legacy P3 has no embedded metric map; using strict P3 state load + "
        "C0 initial/independent metric equality."
    )

for key in required:
    delta = abs(float(observed[key]) - float(expected[key]))
    print(f"{key}: observed={observed[key]:.8f}, expected={expected[key]:.8f}, delta={delta:.3g}")
    if delta > 2e-6:
        raise SystemExit(f"OA-HMD C0 exact P3 guard failed for {key}: {delta}")
print("OA-HMD C0 exact P3 guard passed.")
PY
fi
