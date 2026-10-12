#!/usr/bin/env bash
set -euo pipefail

# Gate-controlled continuation after the first complete 3407 T0/T1 pair.
# Run this inside tmux only after run_SPI_T1_isolated.sh full has finished.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -z "${TMUX:-}" ]]; then
  echo "Refusing a multi-hour continuation outside tmux." >&2
  echo "Start with: tmux new -s spi_t1_followup" >&2
  exit 2
fi

if [[ -d /root/data ]]; then DEFAULT_STORAGE_ROOT=/root/data; else DEFAULT_STORAGE_ROOT=/root/shared-nvme; fi
STORAGE_ROOT="${STORAGE_ROOT:-${DEFAULT_STORAGE_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/SPI_T1_ISOLATED}"
AUDIT_ROOT="${AUDIT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/SPI_T1_ISOLATED}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/BRIGHT-train-logs}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OPTIMIZER_STEPS="${OPTIMIZER_STEPS:-3000}"

mkdir -p "${LOG_ROOT}" "${AUDIT_ROOT}"

checkpoint_step() {
  python -c "import torch; p=torch.load('$1', map_location='cpu', weights_only=False); print(int(p.get('step', -1)))"
}

require_complete_pair() {
  local seed="$1"
  local variant checkpoint observed
  for variant in t0_original_objective t1_component_balanced; do
    checkpoint="${OUTPUT_ROOT}/seed${seed}/${variant}/last.pth"
    test -s "${checkpoint}" || {
      echo "Missing seed ${seed} checkpoint: ${checkpoint}" >&2
      exit 1
    }
    observed="$(checkpoint_step "${checkpoint}")"
    [[ "${observed}" == "${OPTIMIZER_STEPS}" ]] || {
      echo "Incomplete seed ${seed}/${variant}: step=${observed}, expected=${OPTIMIZER_STEPS}" >&2
      exit 1
    }
  done
}

decision_for_seed() {
  local seed="$1"
  local result="${AUDIT_ROOT}/SPI_T1_validation_full_dual_gt_seed${seed}.json"
  python -c "import json; print(json.load(open('${result}', encoding='utf-8'))['decision']['decision'])"
}

assert_fresh_seed() {
  local seed="$1"
  local path
  for path in \
    "${OUTPUT_ROOT}/seed${seed}/t0_original_objective" \
    "${OUTPUT_ROOT}/seed${seed}/t1_component_balanced"; do
    if [[ -e "${path}" ]]; then
      echo "Refusing to overwrite an existing seed directory: ${path}" >&2
      echo "Inspect and archive it before retrying." >&2
      exit 1
    fi
  done
}

echo "[1/5] Preflight updated dual-GT audit"
bash run/run_SPI_T1_isolated.sh preflight

echo "[2/5] Verify and re-audit the completed seed 3407 pair"
require_complete_pair 3407
SEED=3407 NUM_WORKERS="${NUM_WORKERS}" \
  bash run/run_SPI_T1_isolated.sh audit 2>&1 | \
  tee "${LOG_ROOT}/SPI_T1_seed3407_dual_gt_audit.log"

seed3407_decision="$(decision_for_seed 3407)"
echo "Seed 3407 decision: ${seed3407_decision}"
if [[ "${seed3407_decision}" != "PROCEED_TO_T2_PSEUDO_GEOMETRY_LEARNING" ]]; then
  echo "Seed 3407 did not pass the preregistered gate; stopping before extra seeds."
  exit 0
fi

echo "[3/5] Train and audit independent seed 4407"
assert_fresh_seed 4407
SEED=4407 NUM_WORKERS="${NUM_WORKERS}" OPTIMIZER_STEPS="${OPTIMIZER_STEPS}" \
  bash run/run_SPI_T1_isolated.sh full 2>&1 | \
  tee "${LOG_ROOT}/SPI_T1_seed4407_full_dual_gt.log"
require_complete_pair 4407

echo "[4/5] Train and audit independent seed 5407"
assert_fresh_seed 5407
SEED=5407 NUM_WORKERS="${NUM_WORKERS}" OPTIMIZER_STEPS="${OPTIMIZER_STEPS}" \
  bash run/run_SPI_T1_isolated.sh full 2>&1 | \
  tee "${LOG_ROOT}/SPI_T1_seed5407_full_dual_gt.log"
require_complete_pair 5407

echo "[5/5] Cross-seed confirmation"
bash run/run_SPI_T1_isolated.sh summarize 2>&1 | \
  tee "${LOG_ROOT}/SPI_T1_cross_seed_summary.log"

echo "Finished. Inspect: ${AUDIT_ROOT}/SPI_T1_cross_seed_summary.json"
