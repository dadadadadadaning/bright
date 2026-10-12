#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-3407}"
VARIANTS="${VARIANTS:-e1 e3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/CCU_BDA}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/CCU_BDA}"
H6_A2_CKPT="${H6_A2_CKPT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
E0_JSON="${CCU_E0_AUDIT_JSON:-/root/shared-nvme/BRIGHT-eval-results/CCU_BDA/E0_H6A2_local_correspondence_val.json}"
CONTINUE_ON_FAILURE="${CONTINUE_ON_FAILURE:-0}"
mkdir -p "${LOG_ROOT}"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(timestamp)] controlled E1/E3 preflight starts"
bash run/run_CCU_BDA_5090.sh preflight
echo "[$(timestamp)] preflight passed"
bash run/run_CCU_BDA_5090.sh stats

for required in "${H6_A2_CKPT}" "${E0_JSON}"; do
  test -s "${required}" || {
    echo "Missing controlled E1/E3 input: ${required}" >&2
    exit 1
  }
done
python - "${E0_JSON}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
assert payload.get("test_accessed") is False
assert payload.get("training_performed") is False
if payload.get("decision") != "PROCEED_TO_CONTROLLED_E1_E3":
    raise SystemExit("E1/E3 blocked by E0 decision: " + repr(payload.get("decision")))
print("E0 scientific gate passed:", sys.argv[1])
PY

failures=0
for variant in ${VARIANTS}; do
  case "${variant}" in e1|e3) ;; *) echo "Unsupported controlled variant: ${variant}" >&2; exit 2;; esac

  guard_name="CCU_${variant^^}_H6A2_guard_seed${SEED}"
  guard_log="${LOG_ROOT}/${guard_name}.log"
  echo "[$(timestamp)] ${variant}: exact H6-A2 guard starts"
  if env VARIANT="${variant}" SEED="${SEED}" EXP_NAME="${guard_name}" \
      CCU_SOURCE_CKPT="${H6_A2_CKPT}" FORCE_RERUN=1 \
      bash run/run_CCU_BDA_5090.sh guard >"${guard_log}" 2>&1; then
    python - "${H6_A2_CKPT}" "${OUTPUT_ROOT}/${guard_name}/validation_metrics.json" <<'PY'
import json, sys, torch
source = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
expected = source.get("best_metrics", source.get("metrics", {}))
observed = json.load(open(sys.argv[2], "r", encoding="utf-8"))
for key in ("valid_miou", "miou4", "damaged_iou", "destroyed_iou", "loc_f1"):
    delta = abs(float(observed[key]) - float(expected[key]))
    print(f"{key}: observed={observed[key]:.8f}, expected={expected[key]:.8f}, delta={delta:.3g}")
    if delta > 2e-6:
        raise SystemExit(f"controlled H6-A2 guard failed for {key}: {delta}")
print("Controlled H6-A2 exact guard passed.")
PY
    echo "[$(timestamp)] ${variant}: exact guard passed"
  else
    echo "[$(timestamp)] ${variant}: guard failed; see ${guard_log}"
    exit 1
  fi

  smoke_name="CCU_${variant^^}_smoke20_seed${SEED}"
  smoke_log="${LOG_ROOT}/${smoke_name}.log"
  echo "[$(timestamp)] ${variant}: smoke starts"
  if env VARIANT="${variant}" SEED="${SEED}" EXP_NAME="${smoke_name}" \
      CCU_SOURCE_CKPT="${H6_A2_CKPT}" FORCE_RERUN=1 \
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}" \
      ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}" \
      bash run/run_CCU_BDA_5090.sh smoke >"${smoke_log}" 2>&1; then
    echo "[$(timestamp)] ${variant}: smoke passed"
  else
    echo "[$(timestamp)] ${variant}: smoke failed; see ${smoke_log}"
    failures=$((failures + 1))
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] && continue
    exit 1
  fi

  experiment="CCU_${variant^^}_controlled5000_seed${SEED}"
  train_log="${LOG_ROOT}/${experiment}.log"
  echo "[$(timestamp)] ${variant}: controlled 5000-step training starts"
  if env VARIANT="${variant}" SEED="${SEED}" EXP_NAME="${experiment}" \
      CCU_SOURCE_CKPT="${H6_A2_CKPT}" FORCE_RERUN="${FORCE_RERUN:-0}" \
      TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}" \
      ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-4}" \
      bash run/run_CCU_BDA_5090.sh controlled5000 >"${train_log}" 2>&1; then
    echo "[$(timestamp)] ${variant}: controlled training completed"
    best_checkpoint="${OUTPUT_ROOT}/${experiment}/best_val_miou.pth"
    independent_val="${OUTPUT_ROOT}/${experiment}/independent_val.json"
    echo "[$(timestamp)] ${variant}: strict independent validation starts"
    if env CCU_CKPT="${best_checkpoint}" \
        EVAL_OUTPUT_JSON="${independent_val}" \
        bash run/run_CCU_BDA_5090.sh eval_val >>"${train_log}" 2>&1; then
      echo "[$(timestamp)] ${variant}: independent validation completed"
    else
      echo "[$(timestamp)] ${variant}: independent validation failed; see ${train_log}"
      failures=$((failures + 1))
      [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
    fi
  else
    echo "[$(timestamp)] ${variant}: training failed; see ${train_log}"
    failures=$((failures + 1))
    [[ "${CONTINUE_ON_FAILURE}" == "1" ]] || exit 1
  fi
done

SUMMARY="${LOG_ROOT}/CCU_E1_E3_controlled5000_seed${SEED}.json"
python - "${OUTPUT_ROOT}" "${SUMMARY}" "${SEED}" ${VARIANTS} <<'PY'
import json, sys
from pathlib import Path
root, output, seed, *variants = sys.argv[1:]
records = []
evaluations = {}
for variant in variants:
    directory = Path(root) / f"CCU_{variant.upper()}_controlled5000_seed{seed}"
    path = directory / "training_summary.json"
    if not path.is_file():
        continue
    summary = json.loads(path.read_text(encoding="utf-8"))
    evaluation_path = directory / "independent_val.json"
    evaluation = (
        json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation_path.is_file() else {}
    )
    evaluations[variant] = evaluation
    metrics = evaluation.get("observed_metrics", summary.get("best_metrics", {}))
    checkpoint = evaluation.get("checkpoint", {})
    records.append({
        "variant": variant,
        "directory": str(directory),
        "completed_step": summary.get("completed_step"),
        "selected_step": checkpoint.get("step"),
        "strict_independent_validation": checkpoint.get("strict_load", False),
        "valid_miou": metrics.get("valid_miou"),
        "miou4": metrics.get("miou4"),
        "damaged_iou": metrics.get("damaged_iou"),
        "destroyed_iou": metrics.get("destroyed_iou"),
        "event_mean": metrics.get("event_mean_official_style_4class_miou"),
        "event_worst": metrics.get("event_worst_official_style_4class_miou"),
        "transition": metrics.get("ccu_controlled_transition"),
    })
comparison = {}
if "e1" in evaluations and "e3" in evaluations:
    e1_metrics = evaluations["e1"].get("observed_metrics", {})
    e3_metrics = evaluations["e3"].get("observed_metrics", {})
    metric_keys = (
        "valid_miou", "miou4", "damaged_iou", "destroyed_iou",
        "event_mean_official_style_4class_miou",
        "event_worst_official_style_4class_miou",
    )
    comparison["e3_minus_e1_pp"] = {
        key: 100.0 * (float(e3_metrics[key]) - float(e1_metrics[key]))
        for key in metric_keys if key in e1_metrics and key in e3_metrics
    }
    e1_events = evaluations["e1"].get("event_metrics", {})
    e3_events = evaluations["e3"].get("event_metrics", {})
    event_deltas = {}
    for event in sorted(set(e1_events) & set(e3_events)):
        delta = 100.0 * (
            float(e3_events[event]["miou4"])
            - float(e1_events[event]["miou4"])
        )
        event_deltas[event] = delta
    comparison["e3_minus_e1_event_miou4_pp"] = event_deltas
    comparison["e3_nonnegative_events"] = sum(
        delta >= 0.0 for delta in event_deltas.values()
    )
    comparison["comparable_events"] = len(event_deltas)
payload = {
    "protocol": "CCU_E1_E3_controlled5000_comparison_v1",
    "test_accessed": False,
    "records": records,
    "comparison": comparison,
}
Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
print("Saved:", output)
PY
echo "[$(timestamp)] controlled E1/E3 sequence finished; failures=${failures}"
exit "${failures}"
