#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
VARIANT="${2:-a2_lca3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../.." && pwd)"
cd "${PROJECT_ROOT}"

case "${VARIANT}" in a2_lca3|a3_conf3|a4_orient3|a5_conf5) ;; *) echo "variant must be a2_lca3|a3_conf3|a4_orient3|a5_conf5" >&2; exit 2 ;; esac

STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_PATH}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_PATH}/splitname/standard_ML/val_set.txt}"
D0_CKPT="${D0_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_PARENT_V1/D0_original_damageformer_final50000_seed3407/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_ROI_LCA_V1}"
AUDIT_ROOT="${AUDIT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/DAMAGEFORMER_ROI_LCA_V1}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/BRIGHT-train-logs/DAMAGEFORMER_ROI_LCA_V1}"
SEED="${SEED:-3407}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

preflight() {
  python -m py_compile \
    model/DamageFormer_ROI_LCA.py \
    experiments/damageformer_roi_lca_v1/train.py \
    experiments/damageformer_roi_lca_v1/audit.py \
    experiments/damageformer_roi_lca_v1/summarize.py
  python -m unittest experiments.damageformer_roi_lca_v1.test_contract -v
  test -s "${D0_CKPT}" || { echo "Missing D0 checkpoint: ${D0_CKPT}" >&2; exit 3; }
  test -s "${TRAIN_LIST}"; test -s "${VAL_LIST}"
  echo "DamageFormer ROI-LCA V1 preflight passed. Test access: FORBIDDEN."
}

stage() {
  case "$1" in
    smoke) echo "smoke20 20 10 8" ;;
    probe) echo "probe5000 5000 500 0" ;;
    confirm) echo "confirm20000 20000 1000 0" ;;
    *) echo "unknown training stage $1" >&2; exit 2 ;;
  esac
}

train_one() {
  local phase="$1" tag steps eval_every val_max out resume=()
  read -r tag steps eval_every val_max <<<"$(stage "${phase}")"
  out="${OUTPUT_ROOT}/${VARIANT}_${tag}_seed${SEED}"
  if [[ -s "${out}/training_summary.json" ]]; then
    echo "Completed experiment exists; skipped: ${out}"
    return
  fi
  mkdir -p "${out}"
  if [[ -s "${out}/latest_resume.pth" ]]; then
    resume+=(--resume "${out}/latest_resume.pth" --skip_initial_eval)
  elif [[ "${phase}" == "confirm" && -s "${OUTPUT_ROOT}/${VARIANT}_probe5000_seed${SEED}/latest_resume.pth" ]]; then
    resume+=(--resume "${OUTPUT_ROOT}/${VARIANT}_probe5000_seed${SEED}/latest_resume.pth" --skip_initial_eval)
    [[ -s "${OUTPUT_ROOT}/${VARIANT}_probe5000_seed${SEED}/best_val_miou.pth" ]] && \
      cp "${OUTPUT_ROOT}/${VARIANT}_probe5000_seed${SEED}/best_val_miou.pth" "${out}/best_val_miou.pth"
  fi
  python -u experiments/damageformer_roi_lca_v1/train.py \
    --variant "${VARIANT}" --d0_checkpoint "${D0_CKPT}" \
    --dataset_path "${DATASET_PATH}" --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}" \
    --output_dir "${out}" --optimizer_steps "${steps}" --eval_every "${eval_every}" \
    --save_every "${eval_every}" --val_max_batches "${val_max}" \
    --batch_size "${BATCH_SIZE:-4}" --eval_batch_size "${EVAL_BATCH_SIZE:-2}" \
    --accumulation_steps "${ACCUMULATION_STEPS:-4}" --num_workers "${NUM_WORKERS:-8}" \
    --learning_rate "${LR:-1e-4}" --weight_decay "${WEIGHT_DECAY:-5e-3}" \
    --shift_probability "${SHIFT_PROBABILITY:-0.5}" \
    --offset_loss_weight "${OFFSET_LOSS_WEIGHT:-0.05}" \
    --device "${DEVICE:-cuda:0}" --amp "${AMP:-bf16}" --seed "${SEED}" "${resume[@]}"
}

audit_variant() {
  local target="$1" tag="${TRAIN_TAG:-probe5000}" checkpoint directory
  if [[ "${target}" == "d0" ]]; then
    checkpoint="${D0_CKPT}"; directory="${AUDIT_ROOT}/d0_seed${SEED}"
  else
    checkpoint="${OUTPUT_ROOT}/${target}_${tag}_seed${SEED}/best_val_miou.pth"
    directory="${AUDIT_ROOT}/${target}_${tag}_seed${SEED}"
  fi
  test -s "${checkpoint}" || { echo "Missing checkpoint: ${checkpoint}" >&2; exit 4; }
  mkdir -p "${directory}"
  for perturbation in none shift_x4 shift_y4 shift_x8 shift_y8 shift_x16 shift_y16 pair_shuffle zero; do
    [[ -s "${directory}/${perturbation}.json" && "${FORCE_AUDIT:-0}" != "1" ]] && continue
    python -u experiments/damageformer_roi_lca_v1/audit.py \
      --variant "${target}" --checkpoint "${checkpoint}" --dataset_path "${DATASET_PATH}" \
      --val_list "${VAL_LIST}" --perturbation "${perturbation}" \
      --batch_size "${AUDIT_BATCH_SIZE:-2}" --num_workers "${NUM_WORKERS:-8}" \
      --device "${DEVICE:-cuda:0}" --output_json "${directory}/${perturbation}.json"
  done
}

case "${MODE}" in
  preflight) preflight ;;
  smoke|probe|confirm) preflight; train_one "${MODE}" ;;
  audit) preflight; audit_variant "${VARIANT}" ;;
  audit_d0) preflight; audit_variant d0 ;;
  audit_all)
    preflight; audit_variant d0
    for value in a2_lca3 a3_conf3 a4_orient3 a5_conf5; do audit_variant "${value}"; done
    ;;
  summarize)
    python experiments/damageformer_roi_lca_v1/summarize.py \
      --audit_root "${AUDIT_ROOT}" --tag "${TRAIN_TAG:-probe5000}" --seed "${SEED}" \
      --output_json "${AUDIT_ROOT}/summary_${TRAIN_TAG:-probe5000}_seed${SEED}.json"
    ;;
  status)
    pgrep -af 'damageformer_roi_lca_v1/train.py' || true
    find "${OUTPUT_ROOT}" -maxdepth 2 -name training_summary.json -print 2>/dev/null || true
    ;;
  *) echo "mode must be preflight|smoke|probe|confirm|audit|audit_d0|audit_all|summarize|status" >&2; exit 2 ;;
esac
