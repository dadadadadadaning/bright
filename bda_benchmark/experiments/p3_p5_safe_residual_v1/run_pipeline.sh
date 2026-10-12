#!/usr/bin/env bash
set -euo pipefail

# One-command sequence: checks -> 20-step smoke -> 5000-step probe -> audit -> compare.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

echo "===== P3+E3 SAFE RESIDUAL: PREFLIGHT ====="
bash experiments/p3_p5_safe_residual_v1/run.sh preflight

echo "===== P3+E3 SAFE RESIDUAL: SMOKE20 ====="
bash experiments/p3_p5_safe_residual_v1/run.sh smoke

echo "===== P3+E3 SAFE RESIDUAL: PROBE5000 ====="
bash experiments/p3_p5_safe_residual_v1/run.sh probe

echo "===== P3 REFERENCE AUDIT ====="
REFERENCE_AUDIT="${REFERENCE_AUDIT:-/root/shared-nvme/BRIGHT-eval-results/P5_ABLATION_VALIDATION_V1/E0_P3_frozen/none.json}"
if [[ ! -s "${REFERENCE_AUDIT}" ]]; then
  BASE_CKPT="${P3_CKPT:-/root/data/BRIGHT-checkpoints/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth}" \
  SAR_STATS_JSON="${SAR_STATS_JSON:-/root/shared-nvme/BRIGHT-checkpoints/S8_P4_NSMICB/sar_train_stats.json}" \
  bash experiments/p5_ablation_validation_v1/run.sh audit e0
fi

echo "===== P3+E3 SAFE RESIDUAL: FROZEN AUDIT ====="
TRAIN_TAG=probe5000 AUDIT_WEIGHTS=best_val_miou \
bash experiments/p3_p5_safe_residual_v1/run.sh audit

echo "===== P3+E3 SAFE RESIDUAL: ADMISSION ====="
TRAIN_TAG=probe5000 AUDIT_WEIGHTS=best_val_miou \
bash experiments/p3_p5_safe_residual_v1/run.sh compare

echo "===== PIPELINE COMPLETE; OFFICIAL TEST WAS NOT ACCESSED ====="
