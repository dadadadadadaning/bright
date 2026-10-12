#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-preflight}"
HERE="$(cd "$(dirname "$0")" && pwd)"; PROJECT_ROOT="$(cd "${HERE}/../.." && pwd)"; REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"; cd "${PROJECT_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"; DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_PATH}/splitname/standard_ML/train_set.txt}"; VAL_LIST="${VAL_LIST:-${DATASET_PATH}/splitname/standard_ML/val_set.txt}"
INSTANCE_DIR="${INSTANCE_DIR:-${DATASET_PATH}/target_instance_level}"
A1_CKPT="${A1_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a1_long35000_seed3407/best_val_miou.pth}"
JOINT_ROOT="${JOINT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_A1_JOINT_SOFT_PARTITION_V1/a1_joint_partition_joint20000_seed3407}"
JOINT_BEST_CKPT="${JOINT_BEST_CKPT:-${JOINT_ROOT}/best_val_miou.pth}"; JOINT_LATE_CKPT="${JOINT_LATE_CKPT:-${JOINT_ROOT}/latest.pth}"
AUDIT_ROOT="${AUDIT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/REGION_AUDIT_FROZEN_V1}"
FROZEN_OUTPUT_ROOT="${FROZEN_OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_FROZEN_A1_PARTITION_V1}"
FROZEN_OUTPUT_DIR="${FROZEN_OUTPUT_DIR:-${FROZEN_OUTPUT_ROOT}/c5_frozenA1_10000_seed3407}"
SEED="${SEED:-3407}"; REGION_STRIDE="${REGION_STRIDE:-8}"; DEVICE="${DEVICE:-cuda:0}"
preflight(){
 python -m py_compile experiments/damageformer_region_audit_frozen_v1/*.py
 python -m unittest experiments.damageformer_region_audit_frozen_v1.test_contract -v
 python - <<'PY'
from bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.model import DamageFormerA1CrossModalSuperpixel
print('old C5 import: OK')
PY
 test -s "${A1_CKPT}" || { echo "Missing A1: ${A1_CKPT}" >&2; exit 3; }; test -s "${VAL_LIST}" || exit 3
 [[ "$(basename "${VAL_LIST}")" != *test* ]] || { echo 'TEST list forbidden' >&2; exit 4; }
 if [[ -d "${INSTANCE_DIR}" ]]; then python -m bda_benchmark.experiments.damageformer_region_audit_frozen_v1.inspect_instances --instance_dir "${INSTANCE_DIR}" --limit 10; else echo "WARN missing ${INSTANCE_DIR}"; fi
 python - <<'PY'
import torch
print('torch=',torch.__version__,'cuda=',torch.cuda.is_available(),'gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY
 echo 'PREFLIGHT PASSED'
}
audit_checkpoint(){
 local ckpt="$1" tag="$2"; test -s "$ckpt" || { echo "Missing checkpoint: $ckpt" >&2; exit 5; }; mkdir -p "${AUDIT_ROOT}/${tag}"; local -a fallback=(); [[ "${ALLOW_SEMANTIC_CC_FALLBACK:-0}" == 1 ]] && fallback+=(--allow_semantic_cc_fallback)
 python -u -m bda_benchmark.experiments.damageformer_region_audit_frozen_v1.audit_region_structure --checkpoint "$ckpt" --dataset_path "${DATASET_PATH}" --val_list "${VAL_LIST}" --instance_dir "${INSTANCE_DIR}" --output_dir "${AUDIT_ROOT}/${tag}" --region_stride "${REGION_STRIDE}" --batch_size "${AUDIT_BATCH_SIZE:-1}" --num_workers "${NUM_WORKERS:-4}" --device "${DEVICE}" --effective_region_threshold "${EFFECTIVE_REGION_THRESHOLD:-0.05}" --multi_building_fraction "${MULTI_BUILDING_FRACTION:-0.10}" "${fallback[@]}"
}
exact_replay(){ mkdir -p "${AUDIT_ROOT}/exact_replay"; python -u -m bda_benchmark.experiments.damageformer_region_audit_frozen_v1.exact_replay --a1_ckpt "${A1_CKPT}" --dataset_path "${DATASET_PATH}" --val_list "${VAL_LIST}" --region_stride "${REGION_STRIDE}" --batch_size "${EVAL_BATCH_SIZE:-2}" --num_workers "${NUM_WORKERS:-8}" --device "${DEVICE}" --output_json "${AUDIT_ROOT}/exact_replay/a1_zero_partition.json"; }
frozen_train(){
 local steps="$1" ev="$2" vmax="$3"; mkdir -p "${FROZEN_OUTPUT_DIR}"; local -a resume=(); [[ -s "${FROZEN_OUTPUT_DIR}/latest.pth" ]] && resume+=(--resume "${FROZEN_OUTPUT_DIR}/latest.pth")
 python -u -m bda_benchmark.experiments.damageformer_a1_cross_modal_superpixel_v1.train --variant c5_aligned_assoc --parent_ckpt "${A1_CKPT}" --dataset_path "${DATASET_PATH}" --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}" --output_dir "${FROZEN_OUTPUT_DIR}" --optimizer_steps "$steps" --eval_every "$ev" --save_every "${SAVE_EVERY:-5000}" --val_max_batches "$vmax" --batch_size "${BATCH_SIZE:-2}" --eval_batch_size "${EVAL_BATCH_SIZE:-2}" --accumulation_steps "${ACCUMULATION_STEPS:-8}" --num_workers "${NUM_WORKERS:-8}" --plugin_lr "${PLUGIN_LR:-1e-4}" --weight_decay "${WEIGHT_DECAY:-5e-3}" --region_stride "${REGION_STRIDE}" --device "${DEVICE}" --amp_dtype "${AMP_DTYPE:-bf16}" --seed "${SEED}" "${resume[@]}"
}
case "$MODE" in
 preflight) preflight;;
 inspect_instances) python -m bda_benchmark.experiments.damageformer_region_audit_frozen_v1.inspect_instances --instance_dir "${INSTANCE_DIR}" --limit 100;;
 exact_replay) preflight; exact_replay;;
 audit_best) preflight; audit_checkpoint "${JOINT_BEST_CKPT}" joint_best;;
 audit_late) preflight; audit_checkpoint "${JOINT_LATE_CKPT}" joint_late;;
 audit_custom) preflight; test -n "${CHECKPOINT:-}" || { echo 'Set CHECKPOINT=/path/file.pth' >&2; exit 2; }; audit_checkpoint "${CHECKPOINT}" "${AUDIT_TAG:-custom}";;
 compare) python -m bda_benchmark.experiments.damageformer_region_audit_frozen_v1.compare_audits "${AUDIT_ROOT}/joint_best/summary.json" "${AUDIT_ROOT}/joint_late/summary.json";;
 frozen_smoke) preflight; exact_replay; frozen_train 20 10 8;;
 frozen10000) preflight; exact_replay; frozen_train 10000 500 0;;
 *) echo "Usage: $0 {preflight|inspect_instances|exact_replay|audit_best|audit_late|audit_custom|compare|frozen_smoke|frozen10000}" >&2; exit 2;;
esac
