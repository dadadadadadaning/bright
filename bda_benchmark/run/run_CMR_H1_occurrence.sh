#!/usr/bin/env bash
set -euo pipefail

# CMR-H1: Train-only Memory supervision.  Validation uses plain P3 and this
# script has deliberately no Test mode.
MODE="${1:-replay}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

SEED="${SEED:-3407}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/dataset}"
TRAIN_LIST="${TRAIN_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/train_set.txt}"
VAL_LIST="${VAL_LIST:-${PROJECT_ROOT}/dataset/splitname/standard_ML/val_set.txt}"
P3_CKPT="${P3_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}"
P0_ROOT="${P0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_POOLING}"
H0_ROOT="${H0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_H0_TRAIN_ONLY}"
H0B_ROOT="${H0B_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_H0B_EVENT_CAP}"
H1_ROOT="${H1_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/CMR_H1_OCCURRENCE}"
H1_RESULT_ROOT="${H1_RESULT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_H1_OCCURRENCE}"
TRAIN_CACHE="${TRAIN_CACHE:-${P0_ROOT}/CMR_P0_train_pooling_seed${SEED}.pth}"
H0_PAIRS="${H0_PAIRS:-${H0_ROOT}/CMR_H0_train_hard_pairs_seed${SEED}.pth}"
H0B_MANIFEST="${H0B_MANIFEST:-${H0B_ROOT}/CMR_H0B_event_capped_manifest_seed${SEED}.pth}"
H1S0_AUDIT="${H1S0_AUDIT:-${H1_RESULT_ROOT}/CMR_H1S0_replay_seed${SEED}.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${H1_ROOT}/CMR_H1A_screen500_seed${SEED}}"

preflight() {
  python -m py_compile \
    model/cmr/occurrence.py \
    script/standard_ML/p3_hard_occurrence_runtime.py \
    script/standard_ML/audit_CMR_H1S0_replay.py \
    script/standard_ML/train_CMR_H1_occurrence.py
  python -m unittest \
    tests.test_cmr_h0_train_only \
    tests.test_cmr_h0b_event_cap \
    tests.test_cmr_h1_occurrence -v
  echo "CMR-H1 Train-only occurrence preflight passed."
}

ensure_inputs() {
  for path in "${TRAIN_CACHE}" "${H0_PAIRS}" "${H0B_MANIFEST}" "${P3_CKPT}" "${TRAIN_LIST}" "${VAL_LIST}"; do
    test -s "${path}" || { echo "Missing H1 input: ${path}" >&2; exit 3; }
  done
}

replay_guard() {
  ensure_inputs
  mkdir -p "${H1_RESULT_ROOT}"
  python -u script/standard_ML/audit_CMR_H1S0_replay.py \
    --dataset_path "${DATASET_ROOT}" \
    --train_cache "${TRAIN_CACHE}" \
    --h0_pairs "${H0_PAIRS}" \
    --h0b_manifest "${H0B_MANIFEST}" \
    --p3_checkpoint "${P3_CKPT}" \
    --output "${H1S0_AUDIT}" \
    --device cuda:0 \
    --sample_count "${REPLAY_SAMPLE_COUNT:-256}" \
    --num_workers "${NUM_WORKERS:-8}"
}

require_replay() {
  test -s "${H1S0_AUDIT}" || { echo "Missing H1S0 audit: ${H1S0_AUDIT}" >&2; exit 4; }
  python -c 'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get("passed") is True and r.get("test_accessed") is False else 5)' "${H1S0_AUDIT}" || {
    echo "H1S0 replay guard failed; training is blocked." >&2
    exit 5
  }
}

screen() {
  ensure_inputs
  require_replay
  mkdir -p "${OUTPUT_DIR}"
  python -u script/standard_ML/train_CMR_H1_occurrence.py \
    --train_dataset_path "${DATASET_ROOT}" \
    --train_data_list_path "${TRAIN_LIST}" \
    --val_dataset_path "${DATASET_ROOT}" \
    --val_data_list_path "${VAL_LIST}" \
    --train_cache "${TRAIN_CACHE}" \
    --h0_pairs "${H0_PAIRS}" \
    --h0b_manifest "${H0B_MANIFEST}" \
    --h1s0_audit "${H1S0_AUDIT}" \
    --p3_checkpoint "${P3_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --device cuda:0 \
    --train_batch_size "${TRAIN_BATCH_SIZE:-2}" \
    --hard_batch_size "${HARD_BATCH_SIZE:-2}" \
    --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
    --num_workers "${NUM_WORKERS:-8}" \
    --optimizer_steps 500 \
    --eval_every 250 \
    --log_every "${LOG_EVERY:-20}" \
    --core_learning_rate "${CORE_LR:-0.000001}" \
    --projector_learning_rate "${PROJECTOR_LR:-0.0001}" \
    --hard_weight "${HARD_WEIGHT:-0.05}" \
    --distill_weight "${DISTILL_WEIGHT:-0.5}" \
    --temperature "${TEMPERATURE:-0.10}" \
    --negative_count "${NEGATIVE_COUNT:-4}" \
    --seed "${SEED}"
}

case "${MODE}" in
  preflight) preflight ;;
  replay) replay_guard ;;
  screen) screen ;;
  sequence)
    preflight
    replay_guard
    screen
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|replay|screen|sequence" >&2
    exit 2
    ;;
esac

