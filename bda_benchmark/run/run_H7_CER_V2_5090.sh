#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
case "${MODE}" in
  preflight|guard|smoke|pilot|eval_val) ;;
  *) echo "Unknown mode '${MODE}'; choose preflight|guard|smoke|pilot|eval_val" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_ROOT}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_ROOT}/splitname/standard_ML/val_set.txt}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
A2_CKPT="${A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
SOURCE_A4_CKPT="${SOURCE_A4_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H5_BIQ/source/H4_A4_step5500_valid602784_locked.pth}"
SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/HeterogeneousDamageFormer/sar_train_stats.json}"
DAMAGE_PRIORS_JSON="${DAMAGE_PRIORS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/H4_RGBSAR/damage_train_priors.json}"
ORACLE_JSON="${ORACLE_JSON:-/root/shared-nvme/BRIGHT-eval-results/H7_P3_A2_oracle/P3_A2_oracle_validation_full.json}"
PAIRWISE_JSON="${PAIRWISE_JSON:-/root/shared-nvme/BRIGHT-eval-results/H7_P3_A2_oracle/P3_A2_pairwise_transitions.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H7_CER_V2}"
EXPECTED_P3_SHA256="${EXPECTED_P3_SHA256:-45fb0e85a20ab7a36da98dbdcd1462d5aa3c95c6e8a09dd8d99488194992f5f7}"
EXP_NAME="${EXP_NAME:-}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ "${MODE}" == preflight ]]; then
  bash -n run/run_H7_CER_V2_5090.sh
  python -m py_compile \
    model/DamageFormer_RGBSAR_H7_CER_V2.py \
    script/standard_ML/train_DamageFormer_RGBSAR_H7_CER_V2.py \
    script/standard_ML/eval_H7_CER_V2_eventwise.py
  python -m unittest -v \
    tests.test_damageformer_rgbsar_h6_sdcr \
    tests.test_h7_cer \
    tests.test_h7_cer_v2
  echo "H7-CER-V2 preflight passed."
  exit 0
fi

if [[ "${MODE}" == eval_val ]]; then
  H7_V2_CKPT="${H7_V2_CKPT:-}"
  test -s "${H7_V2_CKPT}" || {
    echo "eval_val requires H7_V2_CKPT=/path/to/best_val_miou.pth" >&2; exit 1;
  }
  for required in "${SAR_STATS_JSON}" "${VAL_LIST}" "${ORACLE_JSON}"; do
    test -s "${required}" || { echo "Missing V2 evaluation input: ${required}" >&2; exit 1; }
  done
  EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$(dirname "${H7_V2_CKPT}")/eventwise_val}"
  mkdir -p "${EVAL_OUTPUT_ROOT}"
  python -u script/standard_ML/eval_H7_CER_V2_eventwise.py \
    --checkpoint "${H7_V2_CKPT}" \
    --oracle_json "${ORACLE_JSON}" \
    --dataset_path "${DATASET_ROOT}" \
    --val_data_list_path "${VAL_LIST}" \
    --sar_stats_path "${SAR_STATS_JSON}" \
    --output_json "${EVAL_OUTPUT_ROOT}/H7_CER_V2_eventwise_val.json" \
    --output_event_tsv "${EVAL_OUTPUT_ROOT}/H7_CER_V2_eventwise_val.tsv" \
    --device cuda:0 --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
    --num_workers "${NUM_WORKERS:-8}" --seed "${SEED}"
  exit 0
fi

for required in \
  "${P3_CKPT}" "${A2_CKPT}" "${SOURCE_A4_CKPT}" \
  "${SAR_STATS_JSON}" "${DAMAGE_PRIORS_JSON}" "${TRAIN_LIST}" "${VAL_LIST}" \
  "${ORACLE_JSON}" "${PAIRWISE_JSON}"
do
  test -s "${required}" || { echo "Missing H7-CER-V2 input: ${required}" >&2; exit 1; }
done

python - "${PAIRWISE_JSON}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
rows = {
    (str(row["p3_class"]).lower(), str(row["a2_class"]).lower()): row
    for row in payload.get("overall", [])
}
required = (("damaged", "background"), ("destroyed", "background"))
for pair in required:
    row = rows.get(pair)
    if not row:
        raise SystemExit(f"pairwise report lacks required pair {pair}")
    print(
        f"Pair {pair[0]}->{pair[1]} | net={row['net_a2_correct']} | "
        f"win={100*float(row['a2_win_rate_when_one_expert_correct']):.2f}% | "
        f"safe={row['safe_candidate']}"
    )
    if int(row["net_a2_correct"]) <= 0 or not row.get("safe_candidate"):
        raise SystemExit(f"V2-A pair failed the locked validation audit: {pair}")
PY

actual_p3_sha256="$(sha256sum "${P3_CKPT}" | awk '{print $1}')"
[[ "${actual_p3_sha256}" == "${EXPECTED_P3_SHA256}" ]] || {
  echo "P3 SHA256 mismatch: ${actual_p3_sha256}" >&2; exit 1;
}

case "${MODE}" in
  guard) STEPS=0; EVAL_EVERY=1; VAL_MAX_BATCHES=0; TAG=guard; EXTRA=(--validation_only) ;;
  smoke) STEPS=20; EVAL_EVERY=20; VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-2}"; TAG=smoke20; EXTRA=() ;;
  pilot) STEPS="${OPTIMIZER_STEPS:-1500}"; EVAL_EVERY="${EVAL_EVERY:-250}"; VAL_MAX_BATCHES=0; TAG="pilot${STEPS}"; EXTRA=() ;;
esac

if [[ -z "${EXP_NAME}" ]]; then EXP_NAME="H7_CER_V2A_${TAG}_seed${SEED}"; fi
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${OUTPUT_DIR}"
RESUME_ARGS=()
if [[ "${MODE}" == pilot && -n "${RESUME_CKPT:-}" ]]; then
  test -s "${RESUME_CKPT}" || { echo "Missing V2 resume: ${RESUME_CKPT}" >&2; exit 1; }
  RESUME_ARGS+=(--resume "${RESUME_CKPT}")
elif [[ "${MODE}" == pilot && "${FORCE_RERUN:-0}" != 1 && -s "${OUTPUT_DIR}/latest_resume.pth" ]]; then
  RESUME_ARGS+=(--resume "${OUTPUT_DIR}/latest_resume.pth")
fi

echo "=============================================================================="
echo "Experiment       : ${EXP_NAME}"
echo "Experts          : frozen P3 + frozen H6-A2"
echo "Allowed pairs    : damaged->background; destroyed->background"
echo "Selector output  : KEEP_P3 / USE_A2 / REJECT_BOTH"
echo "Optimized loss   : ONE event/pair-balanced selector CE"
echo "Other losses     : disabled (monitor values may still be printed)"
echo "Test             : NOT ACCESSED"
echo "Steps/EBS        : ${STEPS}/16"
echo "Output           : ${OUTPUT_DIR}"
echo "=============================================================================="

python -u script/standard_ML/train_DamageFormer_RGBSAR_H7_CER_V2.py \
  --variant cer_v2a \
  --source_h3_checkpoint "${SOURCE_A4_CKPT}" \
  --joint_pretrained_checkpoint "${P3_CKPT}" \
  --sar_pretrained_checkpoint "${A2_CKPT}" \
  --damage_priors_json "${DAMAGE_PRIORS_JSON}" \
  --train_dataset_path "${DATASET_ROOT}" --train_data_list_path "${TRAIN_LIST}" \
  --val_dataset_path "${DATASET_ROOT}" --val_data_list_path "${VAL_LIST}" \
  --sar_stats_path "${SAR_STATS_JSON}" --dataset_variant h4 \
  --output_dir "${OUTPUT_DIR}" --device cuda:0 --crop_size 640 \
  --train_batch_size "${TRAIN_BATCH_SIZE:-4}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --accumulation_steps "${ACCUMULATION_STEPS:-4}" \
  --optimizer_steps "${STEPS}" --new_module_warmup_steps 0 \
  --eval_every "${EVAL_EVERY}" --save_every "${SAVE_EVERY:-500}" \
  --log_every "${LOG_EVERY:-20}" --num_workers "${NUM_WORKERS:-8}" \
  --val_max_batches "${VAL_MAX_BATCHES}" \
  --new_module_learning_rate "${SELECTOR_LR:-1e-4}" \
  --sar_learning_rate 0 --core_learning_rate 0 \
  --weight_decay "${WEIGHT_DECAY:-5e-3}" \
  --damage_ce_weight 0 --damage_lovasz_weight 0 \
  --sar_aux_weight 0 --representation_weight 0 \
  --flow_regularization_weight 0 --hierarchy_weight 0 \
  --extra_supervised_weight 1.0 \
  --amp "${AMP:-bf16}" --seed "${SEED}" \
  "${RESUME_ARGS[@]}" "${EXTRA[@]}"

if [[ "${MODE}" == guard ]]; then
  python - "${OUTPUT_DIR}/initial_validation.json" <<'PY'
import json
import sys

metrics = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {"valid_miou": 0.605019, "miou4": 0.695772}
for key, target in expected.items():
    observed = float(metrics[key])
    delta = abs(observed - target)
    print(f"{key}: observed={observed:.8f}, expected={target:.8f}, delta={delta:.3g}")
    if delta > 5e-5:
        raise SystemExit(f"H7-CER-V2 exact-P3 guard failed: {key}")
print("H7-CER-V2 exact-P3 guard passed; initial route rate is zero by construction.")
PY
fi
