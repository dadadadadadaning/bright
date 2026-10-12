#!/usr/bin/env bash
set -euo pipefail

# Single-GPU, restart-safe overnight schedule:
#   1) wait for or resume the formal 50k original DamageFormer parent;
#   2) verify the parent identity;
#   3) run A0-A3 controlled confirmations from the same parent;
#   4) continue selected SALT controls/candidates from their 8k checkpoints.
# No official test split is accessed.

HERE="$(cd "$(dirname "$0")" && pwd)"
BDA_ROOT="$(cd "${HERE}/.." && pwd)"
cd "${BDA_ROOT}"

STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
SEED="${SEED:-3407}"
GPU_ID="${GPU_ID:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RUN_A_CONFIRM="${RUN_A_CONFIRM:-1}"
RUN_SALT_SHORT="${RUN_SALT_SHORT:-1}"
A_STAGE="${A_STAGE:-long}"
SALT_STAGE="${SALT_STAGE:-long}"

case "${A_STAGE}" in confirm|long|full) ;; *) echo "A_STAGE must be confirm|long|full" >&2; exit 2 ;; esac
case "${SALT_STAGE}" in short|long|final) ;; *) echo "SALT_STAGE must be short|long|final" >&2; exit 2 ;; esac

PARENT_DIR="${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_PARENT_V1/D0_original_damageformer_final50000_seed${SEED}"
PARENT_CKPT="${PARENT_DIR}/best_val_miou.pth"
PARENT_SUMMARY="${PARENT_DIR}/training_summary.json"
MASTER_LOG_ROOT="${STORAGE_ROOT}/BRIGHT-train-logs/OVERNIGHT_A_SALT_V1"
mkdir -p "${MASTER_LOG_ROOT}"

# Prevent two launchers from scheduling duplicate GPU work.
exec 9>"${MASTER_LOG_ROOT}/launcher_seed${SEED}.lock"
if ! flock -n 9; then
  echo "Another overnight A/SALT launcher already holds the lock." >&2
  exit 9
fi

timestamp() { date --iso-8601=seconds; }
stage() { echo "===== $* | $(timestamp) ====="; }

parent_complete() {
  [[ -s "${PARENT_CKPT}" && -s "${PARENT_SUMMARY}" ]] || return 1
  python -c "import json; p=json.load(open('${PARENT_SUMMARY}')); raise SystemExit(0 if int(p.get('completed_step',-1)) == 50000 else 1)"
}

parent_running() {
  pgrep -af 'train_SALT_CDE.py' 2>/dev/null | grep -F -- "${PARENT_DIR}" >/dev/null
}

stage "configuration"
echo "BDA root          : ${BDA_ROOT}"
echo "Storage root      : ${STORAGE_ROOT}"
echo "GPU/seed          : ${GPU_ID}/${SEED}"
echo "Parent            : ${PARENT_CKPT}"
echo "A enabled/stage   : ${RUN_A_CONFIRM}/${A_STAGE}"
echo "SALT enabled/stage: ${RUN_SALT_SHORT}/${SALT_STAGE}"
echo "Official test     : FORBIDDEN"

stage "preflight"
GPU_ID="${GPU_ID}" SEED="${SEED}" STORAGE_ROOT="${STORAGE_ROOT}" \
  bash experiments/p3_damageformer_ablation_v1/prepare_parent.sh preflight

if ! parent_complete; then
  if parent_running; then
    stage "formal D0 is already running; wait for completion"
    poll_count=0
    while parent_running; do
      sleep "${POLL_SECONDS}"
      poll_count=$((poll_count + 1))
      if (( poll_count % 10 == 0 )); then
        echo "Still waiting for formal D0 at $(timestamp)"
      fi
    done
  fi

  # If the earlier job stopped before 50k, this resumes from latest_resume.pth.
  if ! parent_complete; then
    stage "formal D0 incomplete; start or resume to 50000"
    GPU_ID="${GPU_ID}" SEED="${SEED}" STORAGE_ROOT="${STORAGE_ROOT}" \
      bash experiments/p3_damageformer_ablation_v1/prepare_parent.sh train
  fi
fi

parent_complete || { echo "Formal D0 did not complete 50000 steps." >&2; exit 10; }

stage "verify formal D0 parent"
GPU_ID="${GPU_ID}" SEED="${SEED}" STORAGE_ROOT="${STORAGE_ROOT}" \
  bash experiments/p3_damageformer_ablation_v1/prepare_parent.sh verify

if [[ "${RUN_A_CONFIRM}" == "1" ]]; then
  stage "A0-A3 controlled stage=${A_STAGE}"
  PARENT_CKPT="${PARENT_CKPT}" PARENT_KIND=damageformer \
  GPU_ID="${GPU_ID}" SEED="${SEED}" STORAGE_ROOT="${STORAGE_ROOT}" \
    bash experiments/p3_damageformer_ablation_v1/run_sequential.sh "${A_STAGE}" a0 a1 a2 a3
  stage "A0-A3 stage=${A_STAGE} complete"
else
  stage "A0-A3 confirm skipped by RUN_A_CONFIRM=${RUN_A_CONFIRM}"
fi

if [[ "${RUN_SALT_SHORT}" == "1" ]]; then
  stage "SALT preflight"
  SEED="${SEED}" bash experiments/salt_cde_validation_v1/run.sh preflight
  stage "SALT selected variants: continue pilot8000 to stage=${SALT_STAGE}"
  # Do not set FRESH_STAGE: run.sh should inherit each pilot8000 optimizer state.
  SEED="${SEED}" \
    bash experiments/salt_cde_validation_v1/run_sequential.sh "${SALT_STAGE}" d0 d1 d2 d2r
  stage "SALT stage=${SALT_STAGE} complete"
else
  stage "SALT short stage skipped by RUN_SALT_SHORT=${RUN_SALT_SHORT}"
fi

stage "overnight schedule complete"
echo "Review A summary under: ${STORAGE_ROOT}/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1"
echo "Review SALT checkpoints under: ${STORAGE_ROOT}/BRIGHT-checkpoints/SALT_CDE_VALIDATION_V1"
echo "Next manual decision: review adjacent A gains and select the final architecture; do not access test yet."
