#!/usr/bin/env bash
set -euo pipefail

# A gated, sequential launcher for the building-guided SEC experiments.
# It deliberately writes to a new root and never overwrites SEC_ALIGNMENT_V1.
STAGE="${1:-full}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
RUNNER="${SCRIPT_DIR}/run.sh"

export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data/BRIGHT-checkpoints/SEC_BUILDING_CFSA_V1}"
export AUDIT_ROOT="${AUDIT_ROOT:-/root/shared-nvme/BRIGHT-eval-results/SEC_BUILDING_CFSA_V1}"
export TRAIN_TAG="probe5000"
export AUDIT_WEIGHTS="latest"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${AUDIT_ROOT}"

admission_file() {
  local candidate="$1" reference="${2:-d1}"
  echo "${AUDIT_ROOT}/${candidate}_vs_${reference}_probe5000_latest_admission.json"
}

show_admission() {
  local path="$1"
  python - "${path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
print("=" * 88)
print(f"Admission: {path}")
print(f"reference={data.get('reference')} candidate={data.get('candidate')}")
print(f"clean_gain_pp={data.get('clean_gain_pp')}")
print(f"damaged_f1_gain_pp={data.get('damaged_f1_gain_pp')}")
print(f"events_nonnegative={data.get('events', {}).get('nonnegative')}")
print(f"x8_shift={data.get('x8_shift_robustness')}")
print(f"screen_pass={data.get('screen_pass')}")
print("=" * 88)
PY
}

require_pass() {
  local path="$1"
  show_admission "${path}"
  python - "${path}" <<'PY'
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if data.get("screen_pass") is True else 12)
PY
}

train_and_audit() {
  local variant="$1"
  echo "===== ${variant} SMOKE START $(date --iso-8601=seconds) ====="
  bash "${RUNNER}" smoke "${variant}"
  echo "===== ${variant} PROBE5000 START $(date --iso-8601=seconds) ====="
  bash "${RUNNER}" probe "${variant}"
  echo "===== ${variant} AUDIT LATEST START $(date --iso-8601=seconds) ====="
  bash "${RUNNER}" audit "${variant}"
  bash "${RUNNER}" compare "${variant}"
  show_admission "$(admission_file "${variant}")"
  echo "===== ${variant} COMPLETE $(date --iso-8601=seconds) ====="
}

case "${STAGE}" in
  screen|support|full) ;;
  *) echo "stage must be screen|support|full" >&2; exit 2 ;;
esac

echo "Pipeline stage : ${STAGE}"
echo "Output root    : ${OUTPUT_ROOT}"
echo "Audit root     : ${AUDIT_ROOT}"
echo "Policy         : sequential, latest-checkpoint audit, gated continuation"

bash "${RUNNER}" preflight
bash "${RUNNER}" audit_reference

if [[ "${STAGE}" == "screen" || "${STAGE}" == "full" ]]; then
  train_and_audit s2r
  if ! require_pass "$(admission_file s2r)"; then
    echo "STOP: S2R failed its screen gate. Support and coarse-to-fine stages were not run." >&2
    exit 12
  fi
fi

if [[ "${STAGE}" == "support" || "${STAGE}" == "full" ]]; then
  if [[ "${STAGE}" == "support" ]]; then
    test -s "$(admission_file s2r)" || {
      echo "Missing passed S2R screen in this AUDIT_ROOT; run stage=screen first." >&2
      exit 12
    }
    require_pass "$(admission_file s2r)"
  fi
  # All support variants are needed for direct attribution, so run them
  # sequentially after S2R has passed (or when support is explicitly requested).
  for variant in s2r0 s2rp s2rr; do
    train_and_audit "${variant}"
  done
  bash "${RUNNER}" compare_pair s2r s2r0
  bash "${RUNNER}" compare_pair s2rp s2r0
  bash "${RUNNER}" compare_pair s2rr s2rp
  show_admission "$(admission_file s2r s2r0)"
  show_admission "$(admission_file s2rp s2r0)"
  show_admission "$(admission_file s2rr s2rp)"

  # Coarse-to-fine uses predicted building+ring, so its fair reference is s2rr.
  if ! require_pass "$(admission_file s2rr)"; then
    echo "STOP: S2RR failed against D1. Coarse-to-fine was not run." >&2
    exit 12
  fi
  train_and_audit s2rcf
  bash "${RUNNER}" compare_pair s2rcf s2rr
  show_admission "$(admission_file s2rcf s2rr)"
fi

echo "Pipeline finished successfully at $(date --iso-8601=seconds)"
