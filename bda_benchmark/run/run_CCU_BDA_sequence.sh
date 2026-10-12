#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pilot}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-3407}"
# The revised protocol does not batch-run the full prototype chain by
# default.  C2-C4 require an explicit, passing E0 audit.
VARIANTS="${VARIANTS:-c0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/CCU_BDA}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/CCU_BDA}"
CCU_E0_AUDIT_JSON="${CCU_E0_AUDIT_JSON:-/root/shared-nvme/BRIGHT-eval-results/CCU_BDA/E0_H6A2_local_correspondence_val.json}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
mkdir -p "${LOG_ROOT}"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(timestamp)] shared preflight starts"
bash run/run_CCU_BDA_5090.sh preflight
echo "[$(timestamp)] shared preflight passed"
echo "[$(timestamp)] train-only SAR stats check starts"
bash run/run_CCU_BDA_5090.sh stats
echo "[$(timestamp)] train-only SAR stats passed"

if [[ " ${VARIANTS} " =~ " c2 " || " ${VARIANTS} " =~ " c3 " || " ${VARIANTS} " =~ " c4 " ]]; then
  test -s "${CCU_E0_AUDIT_JSON}" || {
    echo "[$(timestamp)] C2-C4 blocked: missing E0 audit ${CCU_E0_AUDIT_JSON}"
    echo "Run: bash run/run_CCU_BDA_5090.sh audit_val"
    exit 1
  }
  python - "${CCU_E0_AUDIT_JSON}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
decision = payload.get("decision")
if decision != "PROCEED_TO_E2_LOCAL_CORRELATION":
    raise SystemExit(
        "C2-C4 blocked by E0 decision: " + repr(decision)
    )
print("E0 audit gate passed:", sys.argv[1])
PY
fi

failures=0
for variant in ${VARIANTS}; do
  source_checkpoint=""
  case "${variant}" in
    c3)
      source_checkpoint="${OUTPUT_ROOT}/CCU_C2_pilot3000_seed${SEED}/best_val_miou.pth"
      ;;
    c4)
      source_checkpoint="${OUTPUT_ROOT}/CCU_C3_pilot3000_seed${SEED}/best_val_miou.pth"
      ;;
  esac
  if [[ -n "${source_checkpoint}" && ! -s "${source_checkpoint}" ]]; then
    echo "[$(timestamp)] ${variant}: missing controlled source ${source_checkpoint}"
    failures=$((failures + 1))
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] && continue
    exit 1
  fi

  smoke_name="CCU_${variant^^}_smoke20_seed${SEED}"
  smoke_log="${LOG_ROOT}/${smoke_name}.log"
  echo "[$(timestamp)] ${variant}: smoke starts"
  if env VARIANT="${variant}" SEED="${SEED}" EXP_NAME="${smoke_name}" \
      CCU_SOURCE_CKPT="${source_checkpoint}" \
      bash run/run_CCU_BDA_5090.sh smoke >"${smoke_log}" 2>&1; then
    echo "[$(timestamp)] ${variant}: smoke passed"
  else
    echo "[$(timestamp)] ${variant}: smoke failed; see ${smoke_log}"
    failures=$((failures + 1))
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] && continue
    exit 1
  fi

  pilot_name="CCU_${variant^^}_pilot3000_seed${SEED}"
  pilot_log="${LOG_ROOT}/${pilot_name}.log"
  echo "[$(timestamp)] ${variant}: ${MODE} starts"
  if env VARIANT="${variant}" SEED="${SEED}" EXP_NAME="${pilot_name}" \
      CCU_SOURCE_CKPT="${source_checkpoint}" \
      bash run/run_CCU_BDA_5090.sh "${MODE}" >"${pilot_log}" 2>&1; then
    echo "[$(timestamp)] ${variant}: ${MODE} completed"
  else
    echo "[$(timestamp)] ${variant}: ${MODE} failed; see ${pilot_log}"
    failures=$((failures + 1))
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  fi
done

SUMMARY="${LOG_ROOT}/CCU_sequence_${MODE}_$(date '+%Y%m%d_%H%M%S').json"
python - "${OUTPUT_ROOT}" "${SUMMARY}" "${SEED}" ${VARIANTS} <<'PY'
import json, sys
from pathlib import Path

root, output, seed, *variants = sys.argv[1:]
records = []
for variant in variants:
    directory = Path(root) / f"CCU_{variant.upper()}_pilot3000_seed{seed}"
    path = directory / "training_summary.json"
    if path.is_file():
        summary = json.loads(path.read_text(encoding="utf-8"))
        metrics = summary.get("best_metrics", {})
        records.append({
            "variant": variant,
            "directory": str(directory),
            "completed_step": summary.get("completed_step"),
            "valid_miou": metrics.get("valid_miou"),
            "miou4": metrics.get("miou4"),
            "damaged_iou": metrics.get("damaged_iou"),
            "destroyed_iou": metrics.get("destroyed_iou"),
            "event_mean": metrics.get("event_mean_official_style_4class_miou"),
        })
Path(output).write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
print("Saved:", output)
PY
echo "[$(timestamp)] CCU-BDA sequence finished; failures=${failures}"
exit "${failures}"
