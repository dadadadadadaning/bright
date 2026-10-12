#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/BRIGHT-master

export PYTHONPATH=/root/shared-nvme/BRIGHT-master:/root/shared-nvme/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

DATASET_ROOT="/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset"
TEST_LIST="/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/test_set.txt"
OUT_ROOT="/root/shared-nvme/BRIGHT-master/eval_results_official"

DAMAGEFORMER_CKPT="${DAMAGEFORMER_CKPT:-}"
A4_CKPT="${A4_CKPT:-}"
A4D_CKPT="${A4D_CKPT:-}"
A4IRC_CKPT="${A4IRC_CKPT:-}"
A4DG_CKPT="${A4DG_CKPT:-}"

mkdir -p "${OUT_ROOT}"

run_eval() {
  local name="$1"
  local ckpt="$2"
  shift 2

  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "Skip ${name}: checkpoint not found (${ckpt})"
    return 0
  fi

  echo "Evaluating ${name}: ${ckpt}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" python full_evaluate_unified.py \
    --eval_protocol official \
    --model_path "${ckpt}" \
    --dataset_path "${DATASET_ROOT}" \
    --data_list_path "${TEST_LIST}" \
    --image_size 640 \
    --batch_size 1 \
    --num_workers 2 \
    --save_dir "${OUT_ROOT}/${name}" \
    "$@"
}

run_eval "DamageFormer" "${DAMAGEFORMER_CKPT}" \
  --model_type DamageFormer

run_eval "A4" "${A4_CKPT}" \
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

run_eval "A4_D" "${A4D_CKPT}" \
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

run_eval "A4_IRC" "${A4IRC_CKPT}" \
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

run_eval "A4_DG" "${A4DG_CKPT}" \
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

python scripts/eval/summarize_official_results.py --root "${OUT_ROOT}"
