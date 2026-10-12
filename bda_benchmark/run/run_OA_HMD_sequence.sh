#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-screen}"
if [[ "${MODE}" != "screen" ]]; then
  echo "Usage: bash run/run_OA_HMD_sequence.sh screen" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-3407}"
STEPS="${OPTIMIZER_STEPS:-3000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/OA_HMD}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/OA_HMD}"
mkdir -p "${LOG_ROOT}"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(timestamp)] OA-HMD preflight starts"
bash run/run_OA_HMD_5090.sh preflight

echo "[$(timestamp)] C0 exact P3 guard starts"
EXP_NAME="OA_HMD_C0_guard_seed${SEED}" \
SEED="${SEED}" bash run/run_OA_HMD_5090.sh guard
BASELINE_JSON="${OUTPUT_ROOT}/OA_HMD_C0_guard_seed${SEED}/independent_val.json"

echo "[$(timestamp)] C1 smoke starts"
EXP_NAME="OA_HMD_C1_smoke20_seed${SEED}" \
SEED="${SEED}" bash run/run_OA_HMD_5090.sh smoke 2>&1 \
  | tee "${LOG_ROOT}/OA_HMD_C1_smoke20_seed${SEED}.log"
echo "[$(timestamp)] C1 smoke passed"

echo "[$(timestamp)] C1 ${STEPS}-step controlled screen starts"
EXP_NAME="OA_HMD_C1_pilot${STEPS}_seed${SEED}" \
SEED="${SEED}" OPTIMIZER_STEPS="${STEPS}" \
EVAL_EVERY="${EVAL_EVERY:-500}" SAVE_EVERY="${SAVE_EVERY:-1000}" \
bash run/run_OA_HMD_5090.sh pilot \
  2>&1 | tee "${LOG_ROOT}/OA_HMD_C1_pilot${STEPS}_seed${SEED}.log"

EXP="${OUTPUT_ROOT}/OA_HMD_C1_pilot${STEPS}_seed${SEED}"
CKPT="${EXP}/best_event_worst.pth"
if [[ ! -s "${CKPT}" ]]; then
  CKPT="${EXP}/best_val_miou.pth"
fi
echo "[$(timestamp)] C1 strict independent validation starts | checkpoint=${CKPT}"
OA_HMD_CKPT="${CKPT}" \
OA_BASELINE_JSON="${BASELINE_JSON}" \
EVAL_OUTPUT_JSON="${EXP}/independent_val.json" \
bash run/run_OA_HMD_5090.sh eval_val

python - "${EXP}/independent_val.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
g = r.get("advancement_gate") or {}
print(json.dumps(g, indent=2))
if not g.get("passed", False):
    print("Decision: STOP_C1_DO_NOT_TEST_OR_ADD_STAGE2")
else:
    print("Decision: C1_PASSED_LOCK_CONFIGURATION_THEN_OPTIONALLY_TEST")
PY

GATE_PASSED="$(python - "${EXP}/independent_val.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
print("1" if (r.get("advancement_gate") or {}).get("passed", False) else "0")
PY
)"
if [[ "${GATE_PASSED}" == "1" && "${EVAL_TEST_ON_PASS:-0}" == "1" ]]; then
  echo "[$(timestamp)] C1 gate passed; one independent diagnostic test starts"
  OA_HMD_CKPT="${CKPT}" \
  EVAL_OUTPUT_JSON="${EXP}/independent_test.json" \
  bash run/run_OA_HMD_5090.sh eval_test
fi

echo "[$(timestamp)] OA-HMD controlled sequence finished"
echo "Test was not accessed. To run one diagnostic test only after the gate passes:"
echo "OA_HMD_CKPT=${CKPT} bash run/run_OA_HMD_5090.sh eval_test"
