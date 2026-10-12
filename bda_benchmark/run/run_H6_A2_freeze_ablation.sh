#!/usr/bin/env bash
set -euo pipefail

# Controlled diagnostic: does the native-SAR expert fail because its shallow
# representation is frozen?  This sequence never evaluates on test.
#
# F0: historical H6-A2 policy (stem + stages 1/2 frozen)
# F1: stem + stage 1 frozen; stages 2-4 train
# F2: complete SAR hierarchy trains; RGB/localisation remain frozen

MODE="${1:-screen}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

case "${MODE}" in
  preflight)
    bash run/run_H6_SDCR_5090.sh preflight
    exit 0
    ;;
  smoke) STEPS="${OPTIMIZER_STEPS:-20}" ;;
  screen|evaluate) STEPS="${OPTIMIZER_STEPS:-1500}" ;;
  *) echo "Expected preflight|smoke|screen|evaluate" >&2; exit 2 ;;
esac

SEED="${SEED:-3407}"
EVAL_EVERY="${EVAL_EVERY:-500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/H6_SDCR}"
mkdir -p "${LOG_ROOT}"

declare -a LABELS=(F0 F1 F2)
declare -a MODES=(f0_shallow_frozen f1_stage2_to_4 f2_all_sar)

for index in 0 1 2; do
  label="${LABELS[$index]}"
  freeze_mode="${MODES[$index]}"
  exp_name="H6_A2_${label}_freeze${STEPS}_seed${SEED}"
  log="${LOG_ROOT}/${exp_name}.log"
  if [[ "${MODE}" == evaluate ]]; then
    checkpoint="${OUTPUT_ROOT}/${exp_name}/best_val_miou.pth"
    output_json="${OUTPUT_ROOT}/${exp_name}/independent_val.json"
    test -s "${checkpoint}" || {
      echo "Missing ${label} best validation checkpoint: ${checkpoint}" >&2
      exit 1
    }
    echo "[$(date '+%F %T')] ${label}: strict independent validation"
    env \
      SAR_FREEZE_MODE="${freeze_mode}" \
      H6_CKPT="${checkpoint}" \
      EVAL_OUTPUT_JSON="${output_json}" \
      bash run/run_H6_SDCR_5090.sh eval_val \
      2>&1 | tee "${LOG_ROOT}/${exp_name}_independent_val.log"
    continue
  fi
  echo "[$(date '+%F %T')] ${label}: ${freeze_mode}, ${STEPS} steps"
  env \
    VARIANT=a2 \
    SAR_FREEZE_MODE="${freeze_mode}" \
    EXP_NAME="${exp_name}" \
    SEED="${SEED}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    LOG_ROOT="${LOG_ROOT}" \
    OPTIMIZER_STEPS="${STEPS}" \
    EVAL_EVERY="${EVAL_EVERY}" \
    SAVE_EVERY="${STEPS}" \
    ENABLE_TEST_MONITOR=0 \
    FORCE_RERUN="${FORCE_RERUN:-0}" \
    bash run/run_H6_SDCR_5090.sh "$([[ ${MODE} == smoke ]] && echo smoke || echo pilot)" \
    2>&1 | tee "${log}"
done

python - "${OUTPUT_ROOT}" "${STEPS}" "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
steps = int(sys.argv[2])
seed = int(sys.argv[3])
print(
    f"{'Run':<4} {'Mode':<20} {'Step':>6} {'Valid':>9} {'Dmg':>9} "
    f"{'Des':>9} {'Recall':>9} {'GateMiss':>10} {'Event':>9} {'Worst':>9}"
)
print("-" * 112)
for label, mode in (
    ("F0", "f0_shallow_frozen"),
    ("F1", "f1_stage2_to_4"),
    ("F2", "f2_all_sar"),
):
    directory = root / f"H6_A2_{label}_freeze{steps}_seed{seed}"
    path = directory / "independent_val.json"
    independent = path.is_file()
    if not independent:
        path = directory / "training_summary.json"
    if not path.is_file():
        print(f"{label:<4} MISSING: {path}")
        continue
    result = json.loads(path.read_text(encoding="utf-8"))
    metrics = result["observed_metrics"] if independent else result["best_metrics"]
    selected = (
        result.get("checkpoint", {})
        if independent
        else result.get("selection_records", {}).get("valid", {})
    )
    print(
        f"{label:<4} {mode:<20} {int(selected.get('step', 0)):6d} "
        f"{100*metrics['valid_miou']:9.4f} {100*metrics['damaged_iou']:9.4f} "
        f"{100*metrics['destroyed_iou']:9.4f} "
        f"{100*metrics.get('affected_recall', 0):9.4f} "
        f"{100*metrics.get('damaged_gate_miss', 0):10.4f} "
        f"{100*metrics['event_mean_official_style_4class_miou']:9.4f} "
        f"{100*metrics['event_worst_official_style_4class_miou']:9.4f}"
    )
PY
