#!/usr/bin/env bash
set -euo pipefail

# Controlled OA-HSDNet experiment: keep N1-shared architecture fixed, balance
# training images by disaster event, adapt only heads then SAR Stage3/4, and
# deliberately stop before the unstable Stage2 joint phase.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SEED="${SEED:-3407}"
H6_INIT="${H6_INIT:-/root/shared-nvme/BRIGHT-checkpoints/H6_SDCR/H6_A2_pilot3000_seed3407/best_val_miou.pth}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints/OA_HSDNET}"
LOG_ROOT="${LOG_ROOT:-/root/shared-nvme/BRIGHT-train-logs/OA_HSDNET}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-${OUTPUT_ROOT}/OA_HSDNET_N1_SHARED_deep1500_seed${SEED}/training_summary.json}"
mkdir -p "${LOG_ROOT}"

test -s "${H6_INIT}" || {
  echo "Missing H6-A2 encoder initialisation: ${H6_INIT}" >&2
  exit 1
}

bash run/run_OABH_5090.sh preflight

echo "[$(date '+%F %T')] n1_shared_eb: smoke"
EVENT_BALANCED_SAMPLING=1 \
OABH_H6_INIT_CKPT="${H6_INIT}" \
VARIANT=n1_shared SEED="${SEED}" \
EXP_NAME="OA_HSDNET_N1_SHARED_EB_smoke20_seed${SEED}" \
  bash run/run_OABH_5090.sh smoke \
  > "${LOG_ROOT}/OA_HSDNET_N1_SHARED_EB_smoke20_seed${SEED}.log" 2>&1

heads_exp="${OUTPUT_ROOT}/OA_HSDNET_N1_SHARED_EB_heads500_seed${SEED}"
echo "[$(date '+%F %T')] n1_shared_eb: event-balanced frozen-encoder heads"
EVENT_BALANCED_SAMPLING=1 \
OABH_H6_INIT_CKPT="${H6_INIT}" \
VARIANT=n1_shared SEED="${SEED}" \
EXP_NAME="OA_HSDNET_N1_SHARED_EB_heads500_seed${SEED}" \
  bash run/run_OABH_5090.sh heads \
  > "${LOG_ROOT}/OA_HSDNET_N1_SHARED_EB_heads500_seed${SEED}.log" 2>&1
heads_ckpt="${heads_exp}/best_event_worst.pth"
test -s "${heads_ckpt}" || {
  echo "Missing event-balanced heads checkpoint: ${heads_ckpt}" >&2
  exit 1
}

deep_exp="${OUTPUT_ROOT}/OA_HSDNET_N1_SHARED_EB_deep1500_seed${SEED}"
echo "[$(date '+%F %T')] n1_shared_eb: event-balanced Stage3/4 adaptation"
EVENT_BALANCED_SAMPLING=1 \
OABH_FULL_INIT_CKPT="${heads_ckpt}" \
VARIANT=n1_shared SEED="${SEED}" \
EXP_NAME="OA_HSDNET_N1_SHARED_EB_deep1500_seed${SEED}" \
  bash run/run_OABH_5090.sh deep \
  > "${LOG_ROOT}/OA_HSDNET_N1_SHARED_EB_deep1500_seed${SEED}.log" 2>&1

candidate_summary="${deep_exp}/training_summary.json"
test -s "${candidate_summary}" || {
  echo "Missing event-balanced summary: ${candidate_summary}" >&2
  exit 1
}

if [[ -s "${BASELINE_SUMMARY}" ]]; then
  report="${deep_exp}/event_balanced_advancement_gate.json"
  python script/standard_ML/summarize_OABH.py \
    --baseline "${BASELINE_SUMMARY}" \
    --candidate "${candidate_summary}" \
    --output "${report}"
  echo "Saved comparison: ${report}"
else
  echo "Baseline summary not found; training completed without comparison: ${BASELINE_SUMMARY}" >&2
fi

echo "OA-HSDNet N1-shared event-balanced Deep-only sequence completed."
echo "Joint Stage2 adaptation and test evaluation were intentionally not run."
