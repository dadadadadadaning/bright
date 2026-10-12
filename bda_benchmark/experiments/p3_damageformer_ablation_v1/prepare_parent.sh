#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-preflight}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
PARENT_ROOT="${PARENT_ROOT:-${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_PARENT_V1}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/BRIGHT-train-logs/DAMAGEFORMER_PARENT_V1}"
RESULT_ROOT="${RESULT_ROOT:-${STORAGE_ROOT}/BRIGHT-eval-results/DAMAGEFORMER_PARENT_V1}"
SEED="${SEED:-3407}"
EXP_DIR="${PARENT_ROOT}/D0_original_damageformer_final50000_seed${SEED}"
CHECKPOINT="${EXP_DIR}/best_val_miou.pth"

cd "${PROJECT_ROOT}"

case "${MODE}" in
  preflight)
    python -m py_compile \
      experiments/p3_damageformer_ablation_v1/inspect_parent.py \
      experiments/p3_damageformer_ablation_v1/core.py
    OUTPUT_ROOT="${PARENT_ROOT}" \
      bash experiments/salt_cde_validation_v1/run.sh preflight
    echo "Formal parent output: ${EXP_DIR}"
    ;;
  train)
    mkdir -p "${PARENT_ROOT}" "${LOG_ROOT}"
    # Dedicated output root means this cannot accidentally reuse the old
    # probe2000/pilot8000 directories. A partial final run resumes itself.
    OUTPUT_ROOT="${PARENT_ROOT}" FRESH_STAGE=1 SEED="${SEED}" \
      bash experiments/salt_cde_validation_v1/run.sh final d0
    ;;
  verify)
    test -s "${CHECKPOINT}" || { echo "Missing formal checkpoint: ${CHECKPOINT}" >&2; exit 4; }
    mkdir -p "${RESULT_ROOT}"
    python -u experiments/p3_damageformer_ablation_v1/inspect_parent.py \
      --checkpoint "${CHECKPOINT}" \
      --training_summary "${EXP_DIR}/training_summary.json" \
      --expected_step 50000 \
      --output_json "${RESULT_ROOT}/parent_identity_seed${SEED}.json"
    echo ""
    echo "Use for controlled ablation:"
    echo "export PARENT_CKPT=${CHECKPOINT}"
    echo "export PARENT_KIND=damageformer"
    ;;
  status)
    pgrep -af 'train_SALT_CDE.py.*--variant d0' || true
    if [[ -s "${EXP_DIR}/training_summary.json" ]]; then
      python -m json.tool "${EXP_DIR}/training_summary.json"
    elif [[ -s "${EXP_DIR}/latest_resume.pth" ]]; then
      python - "${EXP_DIR}/latest_resume.pth" <<'PY'
import sys, torch
p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("saved_step:", p.get("step"))
print("best_valid_miou:", p.get("best_metrics", {}).get("valid_miou"))
PY
    else
      echo "No formal parent run found in ${EXP_DIR}"
    fi
    ;;
  *) echo "mode must be preflight|train|verify|status" >&2; exit 2 ;;
esac
