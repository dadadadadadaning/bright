#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-screen}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-3407}"
VARIANTS="${VARIANTS:-d1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H12_DGB}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H12_DGB}"
AUDIT_ROOT="${AUDIT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/H12_DGB}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
SKIP_AUDIT="${SKIP_AUDIT:-0}"
mkdir -p "${LOG_ROOT}" "${AUDIT_ROOT}"
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(timestamp)] H12 preflight starts"
bash run/run_H12_DGB_5090.sh preflight
echo "[$(timestamp)] preflight passed"

if [[ "${SKIP_AUDIT}" != "1" ]]; then
  audit_log="${LOG_ROOT}/H12_shallow_audit_val_seed${SEED}.log"
  echo "[$(timestamp)] shallow evidence audit starts"
  if bash run/run_H12_DGB_5090.sh audit_val >"${audit_log}" 2>&1; then
    echo "[$(timestamp)] shallow evidence audit completed; see ${AUDIT_ROOT}"
  else
    echo "[$(timestamp)] shallow evidence audit failed; see ${audit_log}"
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  fi
fi

name="H12_D0_guard_seed${SEED}"
log="${LOG_ROOT}/${name}.log"
echo "[$(timestamp)] d0: exact H6-A2 guard starts"
env VARIANT=d0 SEED="${SEED}" EXP_NAME="${name}" FORCE_RERUN=1 \
  bash run/run_H12_DGB_5090.sh guard >"${log}" 2>&1
echo "[$(timestamp)] d0: guard passed"
D0_JSON="${OUTPUT_ROOT}/${name}/initial_validation.json"

failures=0
for variant in ${VARIANTS}; do
  case "${variant}" in d1|d2|d3) ;; *) echo "Sequence supports d1 d2 d3 only" >&2; exit 2;; esac
  smoke_name="H12_${variant^^}_smoke20_seed${SEED}"
  smoke_log="${LOG_ROOT}/${smoke_name}.log"
  echo "[$(timestamp)] ${variant}: smoke starts"
  if env VARIANT="${variant}" SEED="${SEED}" EXP_NAME="${smoke_name}" FORCE_RERUN=1 \
      bash run/run_H12_DGB_5090.sh smoke >"${smoke_log}" 2>&1; then
    echo "[$(timestamp)] ${variant}: smoke passed"
  else
    echo "[$(timestamp)] ${variant}: smoke failed; see ${smoke_log}"
    failures=$((failures + 1))
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] && continue
    exit 1
  fi

  steps="${OPTIMIZER_STEPS:-1500}"
  experiment="H12_${variant^^}_screen${steps}_seed${SEED}"
  train_log="${LOG_ROOT}/${experiment}.log"
  echo "[$(timestamp)] ${variant}: controlled ${steps}-step screen starts"
  if env VARIANT="${variant}" SEED="${SEED}" EXP_NAME="${experiment}" \
      OPTIMIZER_STEPS="${steps}" FORCE_RERUN="${FORCE_RERUN:-0}" \
      bash run/run_H12_DGB_5090.sh "${MODE}" >"${train_log}" 2>&1; then
    echo "[$(timestamp)] ${variant}: training completed"
    checkpoint="${OUTPUT_ROOT}/${experiment}/best_val_miou.pth"
    output="${OUTPUT_ROOT}/${experiment}/independent_val.json"
    env H12_CKPT="${checkpoint}" EVAL_OUTPUT_JSON="${output}" \
      H12_BASELINE_JSON="${D0_JSON}" \
      bash run/run_H12_DGB_5090.sh eval_val >>"${train_log}" 2>&1
    echo "[$(timestamp)] ${variant}: independent validation completed"
  else
    echo "[$(timestamp)] ${variant}: training failed; see ${train_log}"
    failures=$((failures + 1))
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  fi
done

REPORT="${OUTPUT_ROOT}/H12_screen_comparison_seed${SEED}.json"
python - "${OUTPUT_ROOT}" "${REPORT}" "${SEED}" "${OPTIMIZER_STEPS:-1500}" "${D0_JSON}" <<'PY'
import json, sys
from pathlib import Path
root, output, seed, steps, d0_path = sys.argv[1:]
records = []
d0 = json.loads(Path(d0_path).read_text(encoding="utf-8")) if Path(d0_path).is_file() else {}
for variant in ("d1", "d2", "d3"):
    directory = Path(root) / f"H12_{variant.upper()}_screen{steps}_seed{seed}"
    path = directory / "independent_val.json"
    if not path.is_file():
        continue
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = result["observed_metrics"]
    records.append({
        "variant": variant,
        "path": str(path),
        "valid_miou": metrics["valid_miou"],
        "damaged_iou": metrics["damaged_iou"],
        "affected_recall": metrics["affected_recall"],
        "damaged_gate_miss": metrics["damaged_gate_miss"],
        "event_mean": metrics["event_mean_official_style_4class_miou"],
        "event_worst": metrics["event_worst_official_style_4class_miou"],
        "advancement_gate": result.get("advancement_gate"),
    })
payload = {
    "protocol": "H12_DGB_screen_comparison_v1",
    "seed": int(seed),
    "steps": int(steps),
    "test_accessed": False,
    "d0_baseline": d0,
    "records": records,
    "next_step": (
        "Run D2 only if D1 improved Damaged IoU, Affected Recall and gate-miss "
        "without lowering event mean/worst. Run D3 only on the winning bypass. "
        "Do not run eval_test until the advancement gate passes."
    ),
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "[$(timestamp)] H12 sequence finished with ${failures} training failures"
exit ${failures}
