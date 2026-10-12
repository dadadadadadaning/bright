#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-n0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-3407}"
H6_INIT="${H6_INIT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/OA_HSDNET}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/OA_HSDNET}"
mkdir -p "${LOG_ROOT}"

test -s "${H6_INIT}" || {
  echo "Missing H6-A2 encoder initialisation: ${H6_INIT}" >&2
  exit 1
}

case "${TARGET}" in
  n0|n1_shared|n1_stage|n1_cond|n2) VARIANTS=("${TARGET}") ;;
  *) echo "Usage: $0 n0|n1_shared|n1_stage|n1_cond|n2" >&2; exit 2 ;;
esac

bash run/run_OABH_5090.sh preflight

for variant in "${VARIANTS[@]}"; do
  upper="${variant^^}"
  echo "[$(date '+%F %T')] ${variant}: smoke"
  OABH_H6_INIT_CKPT="${H6_INIT}" \
  VARIANT="${variant}" SEED="${SEED}" \
  EXP_NAME="OA_HSDNET_${upper}_smoke20_seed${SEED}" \
    bash run/run_OABH_5090.sh smoke \
    > "${LOG_ROOT}/OA_HSDNET_${upper}_smoke20_seed${SEED}.log" 2>&1

  heads_exp="${OUTPUT_ROOT}/OA_HSDNET_${upper}_heads500_seed${SEED}"
  echo "[$(date '+%F %T')] ${variant}: frozen-encoder heads"
  OABH_H6_INIT_CKPT="${H6_INIT}" \
  VARIANT="${variant}" SEED="${SEED}" \
  EXP_NAME="OA_HSDNET_${upper}_heads500_seed${SEED}" \
    bash run/run_OABH_5090.sh heads \
    > "${LOG_ROOT}/OA_HSDNET_${upper}_heads500_seed${SEED}.log" 2>&1
  heads_ckpt="${heads_exp}/best_event_worst.pth"
  test -s "${heads_ckpt}" || { echo "Missing heads checkpoint: ${heads_ckpt}" >&2; exit 1; }

  deep_exp="${OUTPUT_ROOT}/OA_HSDNET_${upper}_deep1500_seed${SEED}"
  echo "[$(date '+%F %T')] ${variant}: deep adaptation"
  OABH_FULL_INIT_CKPT="${heads_ckpt}" \
  VARIANT="${variant}" SEED="${SEED}" \
  EXP_NAME="OA_HSDNET_${upper}_deep1500_seed${SEED}" \
    bash run/run_OABH_5090.sh deep \
    > "${LOG_ROOT}/OA_HSDNET_${upper}_deep1500_seed${SEED}.log" 2>&1
  deep_ckpt="${deep_exp}/best_event_worst.pth"
  test -s "${deep_ckpt}" || { echo "Missing deep checkpoint: ${deep_ckpt}" >&2; exit 1; }

  echo "[$(date '+%F %T')] ${variant}: joint Stage2-4 SAR adaptation"
  OABH_FULL_INIT_CKPT="${deep_ckpt}" \
  VARIANT="${variant}" SEED="${SEED}" \
  EXP_NAME="OA_HSDNET_${upper}_joint3000_seed${SEED}" \
    bash run/run_OABH_5090.sh joint \
    > "${LOG_ROOT}/OA_HSDNET_${upper}_joint3000_seed${SEED}.log" 2>&1
done

echo "OA-HSDNet controlled sequence completed. Test was never accessed."
