#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
VARIANT="${2:-w3_id_mass}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${VARIANT}" in w0_full_residual|w1_id_calibration|w2_id_logit|w3_id_mass|w5_logit4_kd) ;; *) echo "Unknown variant: ${VARIANT}" >&2; exit 2;; esac
STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${DATASET_PATH}/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${DATASET_PATH}/splitname/standard_ML/val_set.txt}"
A1_CKPT="${A1_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a1_long35000_seed3407/best_val_miou.pth}"
STAGE1_CKPT="${STAGE1_CKPT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_BUILDING_INTERNAL_CHANGE_PARTITION_V2/p_sc_residual_seed3407/stage1_change/latest.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_REGION_EVIDENCE_READOUT_V1}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/BRIGHT-train-logs/DAMAGEFORMER_REGION_EVIDENCE_READOUT_V1}"
AUDIT_ROOT="${AUDIT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/DAMAGEFORMER_REGION_EVIDENCE_READOUT_V1}"
SEED="${SEED:-3407}"; DEVICE="${DEVICE:-cuda:0}"; STEPS="${STEPS:-2000}"
OUT="${OUTPUT_ROOT}/${VARIANT}_step${STEPS}_seed${SEED}"
CKPT="${CHECKPOINT:-${OUT}/best_val_miou.pth}"
CALIBRATION="${AUDIT_ROOT}/${VARIANT}_train_coverage.json"

preflight() {
  python -m py_compile experiments/damageformer_region_evidence_readout_v1/*.py
  python -m unittest experiments.damageformer_region_evidence_readout_v1.test_contract -v
  for path in "${A1_CKPT}" "${STAGE1_CKPT}" "${TRAIN_LIST}" "${VAL_LIST}"; do test -s "$path" || { echo "Missing: $path" >&2; exit 3; }; done
  [[ "$(basename "${TRAIN_LIST}")" != *test* && "$(basename "${VAL_LIST}")" != *test* ]] || { echo "TEST ACCESS IS FORBIDDEN" >&2; exit 4; }
  echo "Region Evidence Readout V1 preflight passed. Test access: FORBIDDEN."
}

train() {
  mkdir -p "${OUT}" "${LOG_ROOT}"
  local -a resume=(); [[ -s "${OUT}/latest.pth" ]] && resume=(--resume "${OUT}/latest.pth")
  python -u -m bda_benchmark.experiments.damageformer_region_evidence_readout_v1.train \
    --variant "${VARIANT}" --dataset_path "${DATASET_PATH}" --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}" \
    --a1_ckpt "${A1_CKPT}" --stage1_ckpt "${STAGE1_CKPT}" --output_dir "${OUT}" \
    --optimizer_steps "${STEPS}" --eval_every "${EVAL_EVERY:-250}" --log_every "${LOG_EVERY:-20}" \
    --batch_size "${BATCH_SIZE:-2}" --eval_batch_size "${EVAL_BATCH_SIZE:-2}" \
    --accumulation_steps "${ACCUMULATION_STEPS:-8}" --num_workers "${NUM_WORKERS:-8}" \
    --lr "${LR:-2e-5}" --device "${DEVICE}" --seed "${SEED}" --amp_dtype "${AMP_DTYPE:-bf16}" \
    --val_max_batches "${VAL_MAX_BATCHES:-0}" --expected_stage1_sha256 "${EXPECTED_STAGE1_SHA256:-}" "${resume[@]}"
}

calibrate() {
  test -s "${CKPT}" || { echo "Missing checkpoint: ${CKPT}" >&2; exit 5; }; mkdir -p "${AUDIT_ROOT}"
  python -u -m bda_benchmark.experiments.damageformer_region_evidence_readout_v1.calibrate \
    --variant "${VARIANT}" --dataset_path "${DATASET_PATH}" --train_list "${TRAIN_LIST}" \
    --a1_ckpt "${A1_CKPT}" --stage1_ckpt "${STAGE1_CKPT}" --checkpoint "${CKPT}" \
    --output_json "${CALIBRATION}" --batch_size "${EVAL_BATCH_SIZE:-2}" --num_workers "${NUM_WORKERS:-8}" --device "${DEVICE}"
}

audit() {
  test -s "${CKPT}" || { echo "Missing checkpoint: ${CKPT}" >&2; exit 5; }; mkdir -p "${AUDIT_ROOT}"
  local -a cal=(); [[ -s "${CALIBRATION}" ]] && cal=(--calibration_json "${CALIBRATION}")
  python -u -m bda_benchmark.experiments.damageformer_region_evidence_readout_v1.audit \
    --variant "${VARIANT}" --dataset_path "${DATASET_PATH}" --val_list "${VAL_LIST}" \
    --a1_ckpt "${A1_CKPT}" --stage1_ckpt "${STAGE1_CKPT}" --checkpoint "${CKPT}" \
    --output_json "${AUDIT_ROOT}/${VARIANT}_audit.json" --batch_size "${EVAL_BATCH_SIZE:-2}" \
    --num_workers "${NUM_WORKERS:-8}" --device "${DEVICE}" "${cal[@]}"
}

case "${MODE}" in
  preflight) preflight;;
  smoke) preflight; STEPS=20; OUT="${OUTPUT_ROOT}/_smoke/${VARIANT}_seed${SEED}"; VAL_MAX_BATCHES=8; EVAL_EVERY=10; train;;
  train) preflight; train;;
  calibrate) preflight; calibrate;;
  audit) preflight; audit;;
  stage) preflight; train; if [[ "${VARIANT}" == w2_id_logit || "${VARIANT}" == w3_id_mass ]]; then calibrate; fi; audit;;
  sequence)
    preflight
    for v in w0_full_residual w1_id_calibration w2_id_logit w3_id_mass w5_logit4_kd; do bash "$0" stage "$v"; done
    python -m bda_benchmark.experiments.damageformer_region_evidence_readout_v1.summarize --audit_root "${AUDIT_ROOT}" --output_json "${AUDIT_ROOT}/summary.json"
    ;;
  summarize) python -m bda_benchmark.experiments.damageformer_region_evidence_readout_v1.summarize --audit_root "${AUDIT_ROOT}" --output_json "${AUDIT_ROOT}/summary.json";;
  status) pgrep -af 'damageformer_region_evidence_readout_v1.(train|calibrate|audit)' || true; find "${OUTPUT_ROOT}" -name training_summary.json -print 2>/dev/null || true;;
  *) echo "Usage: $0 {preflight|smoke|train|calibrate|audit|stage|sequence|summarize|status} [variant]" >&2; exit 2;;
esac
