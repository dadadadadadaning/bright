#!/usr/bin/env bash
set -euo pipefail

# Safely continue one SPI T1 seed without launching duplicate GPU jobs.
# It can be started while `run_SPI_T1_isolated.sh full` is still running:
# while the original full pipeline is alive this script only monitors it;
# if that pipeline exits after completing T0, this script runs only the
# missing T1 and/or dual-ground-truth audit stages.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-3407}"
STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_T1_ISOLATED}"
AUDIT_ROOT="${AUDIT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_T1_ISOLATED}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/BRIGHT-train-logs}"
OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-3000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
POLL_SECONDS="${POLL_SECONDS:-60}"

T0_CKPT="${OUTPUT_ROOT}/seed${SEED}/t0_original_objective/last.pth"
T1_CKPT="${OUTPUT_ROOT}/seed${SEED}/t1_component_balanced/last.pth"
AUDIT_JSON="${AUDIT_ROOT}/SPI_T1_validation_full_dual_gt_seed${SEED}.json"

mkdir -p "${LOG_ROOT}" "${AUDIT_ROOT}"

checkpoint_complete() {
  local path="$1"
  local expected_variant="$2"
  python - "$path" "$expected_variant" "$SEED" "$OPTIMIZER_STEPS" <<'PY'
import os
import sys
import torch

path, expected_variant, expected_seed, expected_step = sys.argv[1:]
if not os.path.isfile(path) or os.path.getsize(path) == 0:
    raise SystemExit(1)
try:
    payload = torch.load(path, map_location="cpu", weights_only=False)
except Exception:
    raise SystemExit(1)
ok = (
    payload.get("variant") == expected_variant
    and int(payload.get("seed", -1)) == int(expected_seed)
    and int(payload.get("step", -1)) == int(expected_step)
    and payload.get("test_accessed") is False
)
raise SystemExit(0 if ok else 1)
PY
}

full_pipeline_running() {
  pgrep -af '[r]un_SPI_T1_isolated\.sh full' >/dev/null 2>&1
}

variant_running() {
  local variant="$1"
  pgrep -af "[s]pi_t1\.train.*--variant ${variant}" >/dev/null 2>&1
}

audit_running() {
  pgrep -af '[s]pi_t1\.audit' >/dev/null 2>&1
}

wait_for_checkpoint_or_owner_exit() {
  local path="$1"
  local variant="$2"
  while ! checkpoint_complete "$path" "$variant"; do
    if ! full_pipeline_running && ! variant_running "$variant"; then
      return 1
    fi
    printf 'Waiting for %s step=%s ...\n' "$variant" "$OPTIMIZER_STEPS"
    sleep "$POLL_SECONDS"
  done
  printf 'Verified %s: step=%s seed=%s test=False\n' \
    "$variant" "$OPTIMIZER_STEPS" "$SEED"
}

echo "SPI T1 safe continuation | seed=${SEED} | target steps=${OPTIMIZER_STEPS}"

if ! checkpoint_complete "$T0_CKPT" t0_original_objective; then
  if full_pipeline_running || variant_running t0_original_objective; then
    wait_for_checkpoint_or_owner_exit "$T0_CKPT" t0_original_objective || {
      echo "T0 stopped before a complete step=${OPTIMIZER_STEPS} checkpoint was saved." >&2
      echo "This trainer cannot resume exactly; archive the incomplete T0 output and restart T0." >&2
      exit 3
    }
  else
    echo "No complete T0 checkpoint and no active T0/full process were found." >&2
    echo "Do not continue to T1; restart the paired experiment from T0." >&2
    exit 3
  fi
else
  echo "T0 checkpoint is already complete."
fi

if ! checkpoint_complete "$T1_CKPT" t1_component_balanced; then
  # Give an existing full pipeline time to enter T1 before deciding ownership.
  sleep 15
  if full_pipeline_running || variant_running t1_component_balanced; then
    wait_for_checkpoint_or_owner_exit "$T1_CKPT" t1_component_balanced || {
      echo "T1 stopped before a complete step=${OPTIMIZER_STEPS} checkpoint was saved." >&2
      echo "Archive the incomplete T1 output and rerun this script." >&2
      exit 4
    }
  else
    echo "Original pipeline is no longer active; starting the missing T1 stage."
    SEED="$SEED" NUM_WORKERS="$NUM_WORKERS" OPTIMIZER_STEPS="$OPTIMIZER_STEPS" \
      bash run/run_SPI_T1_isolated.sh train_t1 2>&1 | \
      tee "${LOG_ROOT}/SPI_T1_seed${SEED}_t1_continue.log"
  fi
fi

checkpoint_complete "$T1_CKPT" t1_component_balanced || {
  echo "T1 completion verification failed; audit will not run." >&2
  exit 4
}

if [[ -s "$AUDIT_JSON" ]]; then
  echo "Dual-GT audit already exists: ${AUDIT_JSON}"
  exit 0
fi

sleep 15
if full_pipeline_running || audit_running; then
  echo "An existing pipeline owns the audit; waiting for its result."
  while [[ ! -s "$AUDIT_JSON" ]]; do
    if ! full_pipeline_running && ! audit_running; then
      echo "Existing audit exited without producing ${AUDIT_JSON}" >&2
      exit 5
    fi
    sleep "$POLL_SECONDS"
  done
else
  echo "Starting the missing dual-GT audit."
  SEED="$SEED" NUM_WORKERS="$NUM_WORKERS" \
    bash run/run_SPI_T1_isolated.sh audit 2>&1 | \
    tee "${LOG_ROOT}/SPI_T1_seed${SEED}_dual_gt_audit.log"
fi

test -s "$AUDIT_JSON"
echo "Completed: ${AUDIT_JSON}"
