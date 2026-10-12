#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/BRIGHT-master

export PYTHONPATH=/root/shared-nvme/BRIGHT-master:/root/shared-nvme/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

DATASET_ROOT="${DATASET_ROOT:-/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset}"
TEST_LIST="${TEST_LIST:-/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/test_set.txt}"
SAVE_ROOT="${SAVE_ROOT:-/root/shared-nvme/BRIGHT-master/saved_weights/BGCA_final}"
BASELINE_ROOT="${BASELINE_ROOT:-/root/shared-nvme/BRIGHT-master/saved_weights/BRIGHT}"
OUT_ROOT="${OUT_ROOT:-/root/shared-nvme/BRIGHT-master/eval_results_official}"
IMAGE_SIZE="${IMAGE_SIZE:-640}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

CKPTS=(
  best_balanced_score.pth
  best_damaged_iou.pth
  best_destroyed_iou.pth
  best_valid_miou.pth
)

mkdir -p "${OUT_ROOT}"

latest_dir() {
  local pattern="$1"
  ls -td ${pattern} 2>/dev/null | head -n 1 || true
}

latest_exp_under_parent() {
  local parent="$1"
  latest_dir "${parent}/DamageFormer_BGCA_*"
}

resolve_bgca_exp() {
  local explicit="$1"
  shift
  if [[ -n "${explicit}" && -d "${explicit}" ]]; then
    echo "${explicit}"
    return 0
  fi

  local candidate
  local exp_dir
  for candidate in "$@"; do
    if [[ -d "${candidate}" ]]; then
      exp_dir="$(latest_exp_under_parent "${candidate}")"
      if [[ -n "${exp_dir}" ]]; then
        echo "${exp_dir}"
        return 0
      fi
    fi
  done
}

resolve_baseline_exp() {
  if [[ -n "${DAMAGEFORMER_EXP_DIR:-}" && -d "${DAMAGEFORMER_EXP_DIR}" ]]; then
    echo "${DAMAGEFORMER_EXP_DIR}"
    return 0
  fi
  latest_dir "${BASELINE_ROOT}/DamageFormer_*"
}

run_ckpts() {
  local model_name="$1"
  local exp_dir="$2"
  shift 2

  if [[ -z "${exp_dir}" || ! -d "${exp_dir}" ]]; then
    echo "Skip ${model_name}: experiment dir not found. Set ${model_name}_EXP_DIR if needed."
    return 0
  fi

  echo "==== ${model_name} ===="
  echo "EXP_DIR=${exp_dir}"

  local ckpt ckpt_path tag save_dir
  for ckpt in "${CKPTS[@]}"; do
    ckpt_path="${exp_dir}/${ckpt}"
    if [[ ! -f "${ckpt_path}" ]]; then
      echo "Skip missing checkpoint: ${ckpt_path}"
      continue
    fi

    tag="${ckpt%.pth}"
    save_dir="${OUT_ROOT}/${model_name}/${tag}"
    echo "Evaluating ${model_name}/${tag}"
    python full_evaluate_unified.py \
      --eval_protocol official \
      --model_path "${ckpt_path}" \
      --dataset_path "${DATASET_ROOT}" \
      --data_list_path "${TEST_LIST}" \
      --image_size "${IMAGE_SIZE}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --save_dir "${save_dir}" \
      "$@"
  done
}

DAMAGEFORMER_DIR="$(resolve_baseline_exp)"
A4_DIR="$(resolve_bgca_exp "${A4_EXP_DIR:-}" \
  "${SAVE_ROOT}/A4_MDDM_dilatedROI_crop640_iter200k_bs2" \
  "${SAVE_ROOT}/A4_dilatedROI_crop640_iter200k_bs2" \
  "${SAVE_ROOT}/A4_base_dilatedROI_crop640_iter200k_bs2")"
A4D_DIR="$(resolve_bgca_exp "${A4D_EXP_DIR:-}" \
  "${SAVE_ROOT}/A4_D_MDDM_largeScale_dilatedROI_crop640_iter200k_bs2" \
  "${SAVE_ROOT}/A4_D_largeScale_dilatedROI_crop640_iter200k_bs2")"
A4IRC_DIR="$(resolve_bgca_exp "${A4IRC_EXP_DIR:-}" \
  "${SAVE_ROOT}/A4_IRC_innerRing_dilatedROI_crop640_iter200k_bs2")"
A4DG_DIR="$(resolve_bgca_exp "${A4DG_EXP_DIR:-}" \
  "${SAVE_ROOT}/A4_DG_dualScaleGate_dilatedROI_crop640_iter200k_bs2" \
  "${SAVE_ROOT}/A4_DG_dualScaleGate_dilatedROI_crop640_iter120k_bs2")"
A4_EXIST_DIR="$(resolve_bgca_exp "${A4_EXIST_EXP_DIR:-}" \
  "${SAVE_ROOT}/A4_ExistHead_dilatedROI_crop640_iter200k_bs2" \
  "${SAVE_ROOT}/A4_ExistHead_dilatedROI_crop640_iter120k_bs2")"
A4DG_EXIST_DIR="$(resolve_bgca_exp "${A4DG_EXIST_EXP_DIR:-}" \
  "${SAVE_ROOT}/A4_DG_ExistHead_dilatedROI_crop640_iter200k_bs2" \
  "${SAVE_ROOT}/A4_DG_ExistHead_dilatedROI_crop640_iter120k_bs2")"

run_ckpts "DamageFormer_baseline" "${DAMAGEFORMER_DIR}" \
  --model_type DamageFormer

run_ckpts "A4" "${A4_DIR}" \
  --model_type DamageFormer_BGCA \
  --roi_mode dilated_soft \
  --roi_fusion multiplicative \
  --roi_beta 1.0 \
  --roi_alpha 0.1 \
  --roi_dilate 5 \
  --use_mddm \
  --mddm_variant base \
  --mddm_gamma_init 0.0 \
  --mddm_use_roi

run_ckpts "A4_D" "${A4D_DIR}" \
  --model_type DamageFormer_BGCA \
  --roi_mode dilated_soft \
  --roi_fusion multiplicative \
  --roi_beta 1.0 \
  --roi_alpha 0.1 \
  --roi_dilate 5 \
  --use_mddm \
  --mddm_variant large_scale \
  --mddm_gamma_init 0.0 \
  --mddm_use_roi

run_ckpts "A4_IRC" "${A4IRC_DIR}" \
  --model_type DamageFormer_BGCA \
  --roi_mode dilated_soft \
  --roi_fusion multiplicative \
  --roi_beta 1.0 \
  --roi_alpha 0.1 \
  --roi_dilate 5 \
  --use_mddm \
  --mddm_variant base \
  --mddm_gamma_init 0.0 \
  --mddm_use_roi \
  --use_inner_ring_context \
  --ring_kernel_size 15 \
  --irc_gamma_init 0.0

run_ckpts "A4_DG" "${A4DG_DIR}" \
  --model_type DamageFormer_BGCA \
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

run_ckpts "A4_ExistHead" "${A4_EXIST_DIR}" \
  --model_type DamageFormer_BGCA \
  --roi_mode dilated_soft \
  --roi_fusion multiplicative \
  --roi_beta 1.0 \
  --roi_alpha 0.1 \
  --roi_dilate 5 \
  --use_mddm \
  --mddm_variant base \
  --mddm_gamma_init 0.0 \
  --mddm_use_roi \
  --use_damage_exist_head \
  --lambda_exist 0.1

run_ckpts "A4_DG_ExistHead" "${A4DG_EXIST_DIR}" \
  --model_type DamageFormer_BGCA \
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
  --dg_gamma_mode sigmoid01 \
  --use_damage_exist_head \
  --lambda_exist 0.1

python scripts/eval/summarize_all_official_results.py --root "${OUT_ROOT}"
