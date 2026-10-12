#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-3407}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/CCU_BDA}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/CCU_BDA}"
AUDIT_ROOT="${AUDIT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CCU_BDA}"
H6_A2_CKPT="${H6_A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
RUN_TEST="${RUN_TEST:-1}"
mkdir -p "${LOG_ROOT}" "${AUDIT_ROOT}"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(timestamp)] E0/E2 preflight starts"
bash run/run_CCU_BDA_5090.sh preflight
bash run/run_CCU_BDA_5090.sh stats

E0_JSON="${AUDIT_ROOT}/E0_H6A2_local_correspondence_val_rerun_seed${SEED}.json"
E0_LOG="${LOG_ROOT}/E0_H6A2_local_correspondence_val_rerun_seed${SEED}.log"
echo "[$(timestamp)] E0 validation-only audit starts"
env H6_A2_CKPT="${H6_A2_CKPT}" \
  AUDIT_OUTPUT_JSON="${E0_JSON}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}" \
  NUM_WORKERS="${NUM_WORKERS:-8}" \
  CORRELATION_WINDOW="${CORRELATION_WINDOW:-5}" \
  CORRELATION_TEMPERATURE="${CORRELATION_TEMPERATURE:-0.10}" \
  AUDIT_MIN_CLASS_PURITY="${AUDIT_MIN_CLASS_PURITY:-0.75}" \
  bash run/run_CCU_BDA_5090.sh audit_val >"${E0_LOG}" 2>&1
echo "[$(timestamp)] E0 audit completed"

python - "${E0_JSON}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert payload.get("test_accessed") is False
assert payload.get("training_performed") is False
if payload.get("decision") != "PROCEED_TO_CONTROLLED_E1_E3":
    raise SystemExit("E2 blocked by E0 decision: " + repr(payload.get("decision")))
print("E0 gate passed:", payload.get("best_signal"))
PY

GUARD_NAME="CCU_E2_H6A2_guard_seed${SEED}"
GUARD_LOG="${LOG_ROOT}/${GUARD_NAME}.log"
echo "[$(timestamp)] E2 exact H6-A2 guard starts"
env VARIANT=e2 SEED="${SEED}" EXP_NAME="${GUARD_NAME}" \
  CCU_SOURCE_CKPT="${H6_A2_CKPT}" FORCE_RERUN=1 \
  bash run/run_CCU_BDA_5090.sh guard >"${GUARD_LOG}" 2>&1

python - "${H6_A2_CKPT}" "${OUTPUT_ROOT}/${GUARD_NAME}/validation_metrics.json" <<'PY'
import json, sys, torch
source = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
expected = source.get("best_metrics", source.get("metrics", {}))
observed = json.load(open(sys.argv[2], "r", encoding="utf-8"))
for key in ("valid_miou", "miou4", "damaged_iou", "destroyed_iou", "loc_f1"):
    delta = abs(float(observed[key]) - float(expected[key]))
    print(f"{key}: observed={observed[key]:.8f}, expected={expected[key]:.8f}, delta={delta:.3g}")
    if delta > 2e-6:
        raise SystemExit(f"E2 H6-A2 guard failed for {key}: {delta}")
print("E2 exact H6-A2 guard passed.")
PY

SMOKE_NAME="CCU_E2_smoke20_seed${SEED}"
SMOKE_LOG="${LOG_ROOT}/${SMOKE_NAME}.log"
echo "[$(timestamp)] E2 smoke starts"
env VARIANT=e2 SEED="${SEED}" EXP_NAME="${SMOKE_NAME}" \
  CCU_SOURCE_CKPT="${H6_A2_CKPT}" FORCE_RERUN=1 \
  bash run/run_CCU_BDA_5090.sh smoke >"${SMOKE_LOG}" 2>&1
echo "[$(timestamp)] E2 smoke passed"

EXP_NAME="CCU_E2_controlled5000_seed${SEED}"
EXP_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
TRAIN_LOG="${LOG_ROOT}/${EXP_NAME}.log"
echo "[$(timestamp)] E2 controlled 5000-step training starts"
env VARIANT=e2 SEED="${SEED}" EXP_NAME="${EXP_NAME}" \
  CCU_SOURCE_CKPT="${H6_A2_CKPT}" FORCE_RERUN="${FORCE_RERUN:-0}" \
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}" \
  ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}" \
  KEEP_MILESTONE_CHECKPOINTS=0 \
  bash run/run_CCU_BDA_5090.sh controlled5000 >"${TRAIN_LOG}" 2>&1
echo "[$(timestamp)] E2 training completed"

CKPT="${EXP_DIR}/best_val_miou.pth"
VAL_JSON="${EXP_DIR}/independent_val.json"
echo "[$(timestamp)] E2 strict independent validation starts"
env CCU_CKPT="${CKPT}" EVAL_OUTPUT_JSON="${VAL_JSON}" \
  bash run/run_CCU_BDA_5090.sh eval_val >>"${TRAIN_LOG}" 2>&1
echo "[$(timestamp)] E2 independent validation completed"

TEST_JSON="${EXP_DIR}/independent_test.json"
if [[ "${RUN_TEST}" == "1" ]]; then
  echo "[$(timestamp)] E2 locked-checkpoint independent test starts"
  env CCU_CKPT="${CKPT}" EVAL_OUTPUT_JSON="${TEST_JSON}" \
    bash run/run_CCU_BDA_5090.sh eval_test >>"${TRAIN_LOG}" 2>&1
  echo "[$(timestamp)] E2 independent test completed"
fi

SUMMARY="${LOG_ROOT}/CCU_E0_E2_controlled5000_seed${SEED}.json"
python - "${E0_JSON}" "${VAL_JSON}" "${TEST_JSON}" "${SUMMARY}" <<'PY'
import json, sys
from pathlib import Path
e0_path, val_path, test_path, output_path = map(Path, sys.argv[1:])
e0 = json.loads(e0_path.read_text(encoding="utf-8"))
val = json.loads(val_path.read_text(encoding="utf-8"))
test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.is_file() else None
payload = {
    "protocol": "CCU_E0_E2_controlled5000_summary_v1",
    "e0": {
        "decision": e0.get("decision"),
        "best_signal": e0.get("best_signal"),
        "test_accessed": e0.get("test_accessed"),
    },
    "e2_validation": val.get("observed_metrics"),
    "e2_validation_checkpoint": val.get("checkpoint"),
    "e2_test": test.get("observed_metrics") if test else None,
    "e2_test_checkpoint": test.get("checkpoint") if test else None,
}
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
print("Saved:", output_path)
PY

echo "[$(timestamp)] E0/E2 sequence finished"
