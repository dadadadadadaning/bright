#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-pilot}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-3407}"
VARIANTS="${VARIANTS:-h2 h3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H10_H3D}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H10_H3D}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
mkdir -p "${LOG_ROOT}"
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(timestamp)] H10 preflight starts"
bash run/run_H10_H3D_5090.sh preflight
echo "[$(timestamp)] preflight passed"

for guard_variant in h0 h1; do
  name="H10_${guard_variant^^}_guard_seed${SEED}"
  log="${LOG_ROOT}/${name}.log"
  echo "[$(timestamp)] ${guard_variant}: exact H6-A2 guard starts"
  env VARIANT="${guard_variant}" SEED="${SEED}" EXP_NAME="${name}" FORCE_RERUN=1 \
    bash run/run_H10_H3D_5090.sh guard >"${log}" 2>&1
  echo "[$(timestamp)] ${guard_variant}: guard passed"
done

failures=0
for variant in ${VARIANTS}; do
  case "${variant}" in h2|h3) ;; *) echo "Sequence supports h2 h3 only" >&2; exit 2;; esac
  smoke_name="H10_${variant^^}_smoke20_seed${SEED}"
  smoke_log="${LOG_ROOT}/${smoke_name}.log"
  echo "[$(timestamp)] ${variant}: smoke starts"
  if env VARIANT="${variant}" SEED="${SEED}" EXP_NAME="${smoke_name}" FORCE_RERUN=1 \
      bash run/run_H10_H3D_5090.sh smoke >"${smoke_log}" 2>&1; then
    echo "[$(timestamp)] ${variant}: smoke passed"
  else
    echo "[$(timestamp)] ${variant}: smoke failed; see ${smoke_log}"
    failures=$((failures + 1))
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] && continue
    exit 1
  fi

  steps="${OPTIMIZER_STEPS:-3000}"
  experiment="H10_${variant^^}_pilot${steps}_seed${SEED}"
  train_log="${LOG_ROOT}/${experiment}.log"
  echo "[$(timestamp)] ${variant}: controlled ${steps}-step training starts"
  if env VARIANT="${variant}" SEED="${SEED}" EXP_NAME="${experiment}" \
      OPTIMIZER_STEPS="${steps}" FORCE_RERUN="${FORCE_RERUN:-0}" \
      bash run/run_H10_H3D_5090.sh "${MODE}" >"${train_log}" 2>&1; then
    echo "[$(timestamp)] ${variant}: training completed"
    checkpoint="${OUTPUT_ROOT}/${experiment}/best_val_miou.pth"
    output="${OUTPUT_ROOT}/${experiment}/independent_val.json"
    env H10_CKPT="${checkpoint}" EVAL_OUTPUT_JSON="${output}" \
      bash run/run_H10_H3D_5090.sh eval_val >>"${train_log}" 2>&1
    echo "[$(timestamp)] ${variant}: independent validation completed"
  else
    echo "[$(timestamp)] ${variant}: training failed; see ${train_log}"
    failures=$((failures + 1))
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  fi
done

REPORT="${OUTPUT_ROOT}/H10_H2_H3_comparison_seed${SEED}.json"
python - "${OUTPUT_ROOT}" "${REPORT}" "${SEED}" "${OPTIMIZER_STEPS:-3000}" <<'PY'
import json, sys
from pathlib import Path
root, output, seed, steps = sys.argv[1:]
records = []
events = {}
for variant in ("h2", "h3"):
    directory = Path(root) / f"H10_{variant.upper()}_pilot{steps}_seed{seed}"
    path = directory / "independent_val.json"
    if not path.is_file():
        continue
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = result["observed_metrics"]
    records.append({
        "variant": variant,
        "directory": str(directory),
        "selected_step": result["checkpoint"]["step"],
        **{key: metrics.get(key) for key in (
            "valid_miou", "miou4", "damaged_iou", "destroyed_iou",
            "affected_iou", "affected_f1", "affected_recall",
            "damaged_gate_miss", "damaged_severity_to_destroyed",
            "event_mean_official_style_4class_miou",
            "event_worst_official_style_4class_miou",
            "h10_occurrence_transition",
        )},
    })
    events[variant] = result.get("event_metrics", {})
comparison = {}
if len(records) == 2 and {record["variant"] for record in records} == {"h2", "h3"}:
    h2 = next(record for record in records if record["variant"] == "h2")
    h3 = next(record for record in records if record["variant"] == "h3")
    keys = (
        "valid_miou", "miou4", "damaged_iou", "destroyed_iou",
        "affected_f1", "affected_recall", "damaged_gate_miss",
        "event_mean_official_style_4class_miou",
        "event_worst_official_style_4class_miou",
    )
    comparison["h3_minus_h2_pp"] = {
        key: 100.0 * (float(h3[key]) - float(h2[key]))
        for key in keys if h2.get(key) is not None and h3.get(key) is not None
    }
    event_delta = {
        event: 100.0 * (
            float(events["h3"][event]["miou4"])
            - float(events["h2"][event]["miou4"])
        )
        for event in sorted(set(events["h2"]) & set(events["h3"]))
    }
    comparison["h3_minus_h2_event_miou4_pp"] = event_delta
    comparison["h3_nonnegative_events"] = sum(value >= 0 for value in event_delta.values())
    comparison["comparable_events"] = len(event_delta)
    comparison["proceed_to_h4"] = bool(
        comparison["h3_minus_h2_pp"].get("affected_f1", -1) > 0
        and comparison["h3_minus_h2_pp"].get("affected_recall", -1) > 0
        and comparison["h3_minus_h2_pp"].get("damaged_gate_miss", 1) < 0
        and comparison["h3_minus_h2_pp"].get("miou4", -1) >= 0
        and comparison["h3_nonnegative_events"] >= 10
    )
payload = {
    "protocol": "H10_H2_H3_controlled_comparison_v1",
    "test_accessed": False,
    "records": records,
    "comparison": comparison,
}
Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
print("Saved:", output)
PY

echo "[$(timestamp)] H10 H2/H3 sequence finished; failures=${failures}"
exit "${failures}"
