#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/BRIGHT-master

export PYTHONPATH=/root/shared-nvme/BRIGHT-master:/root/shared-nvme/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

DATASET_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset"
TEST_LIST="/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/test_set.txt"
SAVE_ROOT="/root/shared-nvme/BRIGHT-master/saved_weights/BGCA_final"
DATASET_NAME="A4_DG_dualScaleGate_dilatedROI_crop640_iter120k_bs2"
EXP_PARENT="${SAVE_ROOT}/${DATASET_NAME}"
EXP_DIR="${EXP_DIR:-}"
OUT_ROOT="/root/shared-nvme/BRIGHT-master/eval_results/unified/A4_DG_120k"

if [[ -z "${EXP_DIR}" ]]; then
  EXP_DIR="$(ls -td "${EXP_PARENT}"/DamageFormer_BGCA_* 2>/dev/null | head -n 1)"
fi

if [[ -z "${EXP_DIR}" || ! -d "${EXP_DIR}" ]]; then
  echo "Cannot find experiment dir. Set EXP_DIR=/path/to/experiment and rerun." >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}"
echo "Evaluating EXP_DIR=${EXP_DIR}"

CKPTS=(
  best_balanced_score.pth
  best_valid_miou.pth
  best_damaged_iou.pth
  best_destroyed_iou.pth
)

for CKPT in "${CKPTS[@]}"; do
  CKPT_PATH="${EXP_DIR}/${CKPT}"
  if [[ ! -f "${CKPT_PATH}" ]]; then
    echo "Skip missing checkpoint: ${CKPT_PATH}"
    continue
  fi

  TAG="${CKPT%.pth}"
  SAVE_DIR="${OUT_ROOT}/${TAG}"
  CUDA_VISIBLE_DEVICES=1 python full_evaluate_unified.py \
    --eval_protocol research \
    --model_type DamageFormer_BGCA \
    --model_path "${CKPT_PATH}" \
    --dataset_path "${DATASET_ROOT}" \
    --data_list_path "${TEST_LIST}" \
    --image_size 640 \
    --batch_size 1 \
    --num_workers 4 \
    --save_dir "${SAVE_DIR}" \
    --roi_mode dilated_soft \
    --roi_fusion multiplicative \
    --roi_beta 1.0 \
    --roi_alpha 0.1 \
    --roi_dilate 5 \
    --use_mddm \
    --mddm_variant dual_scale_gate \
    --mddm_gamma_init 0.0 \
    --mddm_use_roi \
    --dg_gate_reduction 4 \
    --dg_gamma_init 0.0 \
    --dg_gamma_mode sigmoid01
done

python - <<'PY'
import csv
import json
import os

out_root = "/root/shared-nvme/BRIGHT-master/eval_results/unified/A4_DG_120k"
rows = []
for name in [
    "best_balanced_score",
    "best_valid_miou",
    "best_damaged_iou",
    "best_destroyed_iou",
]:
    path = os.path.join(out_root, name, "summary.json")
    if not os.path.exists(path):
        continue
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    per_class = summary.get("per_class", {})
    destroyed = per_class.get("3", {})
    damaged = per_class.get("2", {})
    intact = per_class.get("1", {})
    rows.append({
        "checkpoint": name,
        "loc_iou": summary.get("loc_iou", 0.0),
        "intact_iou": intact.get("iou", 0.0),
        "damaged_iou": damaged.get("iou", 0.0),
        "destroyed_iou": destroyed.get("iou", 0.0),
        "valid_miou": summary.get("valid_miou", 0.0),
        "final_miou": summary.get("final_miou", 0.0),
        "destroyed_to_intact": summary.get("destroyed_to_intact_rate", 0.0),
        "destroyed_recall": destroyed.get("recall", 0.0),
    })

summary_csv = os.path.join(out_root, "checkpoint_summary.csv")
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "checkpoint",
        "loc_iou",
        "intact_iou",
        "damaged_iou",
        "destroyed_iou",
        "valid_miou",
        "final_miou",
        "destroyed_to_intact",
        "destroyed_recall",
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote summary: {summary_csv}")
for row in rows:
    print(row)
PY
