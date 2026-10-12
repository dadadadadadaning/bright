#!/usr/bin/env bash
set -euo pipefail

# H0B changes only the Train anchor schedule.  It does not train P3, load
# Validation/Test features, or grant Memory any inference permission.
MODE="${1:-audit}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="$(cd "${PROJECT_ROOT}/.." && pwd):${PROJECT_ROOT}:${PYTHONPATH:-}"

SEED="${SEED:-3407}"
P0_ROOT="${P0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_POOLING}"
H0_ROOT="${H0_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_H0_TRAIN_ONLY}"
H0B_ROOT="${H0B_ROOT:-/root/shared-nvme/BRIGHT-eval-results/CMR_H0B_EVENT_CAP}"
TRAIN_CACHE="${TRAIN_CACHE:-${P0_ROOT}/CMR_P0_train_pooling_seed${SEED}.pth}"
H0_PAIRS="${H0_PAIRS:-${H0_ROOT}/CMR_H0_train_hard_pairs_seed${SEED}.pth}"
MANIFEST="${MANIFEST:-${H0B_ROOT}/CMR_H0B_event_capped_manifest_seed${SEED}.pth}"
AUDIT_JSON="${AUDIT_JSON:-${H0B_ROOT}/CMR_H0B_event_capped_audit_seed${SEED}.json}"

preflight() {
  python -m py_compile \
    script/standard_ML/cmr_event_capped_sampler.py \
    script/standard_ML/audit_CMR_H0B_event_cap.py
  python -m unittest \
    tests.test_cmr_h0_train_only \
    tests.test_cmr_h0b_event_cap -v
  echo "CMR-H0B event-capped Train sampler preflight passed."
}

ensure_h0() {
  if [[ ! -s "${TRAIN_CACHE}" || ! -s "${H0_PAIRS}" ]]; then
    echo "H0 inputs missing; rebuilding the Train-only H0 audit." >&2
    bash run/run_CMR_H0_train_only.sh audit
  fi
  test -s "${TRAIN_CACHE}" || { echo "Missing Train cache: ${TRAIN_CACHE}" >&2; exit 3; }
  test -s "${H0_PAIRS}" || { echo "Missing H0 pairs: ${H0_PAIRS}" >&2; exit 3; }
}

audit_sampler() {
  ensure_h0
  mkdir -p "${H0B_ROOT}"
  python -u script/standard_ML/audit_CMR_H0B_event_cap.py \
    --train_cache "${TRAIN_CACHE}" \
    --h0_pairs "${H0_PAIRS}" \
    --manifest_output "${MANIFEST}" \
    --audit_output "${AUDIT_JSON}" \
    --cap_per_event "${CAP_PER_EVENT:-64}" \
    --audit_epochs "${AUDIT_EPOCHS:-8}" \
    --seed "${SEED}"
}

case "${MODE}" in
  preflight) preflight ;;
  audit) audit_sampler ;;
  all)
    preflight
    audit_sampler
    ;;
  *)
    echo "Unknown mode '${MODE}'; choose preflight|audit|all" >&2
    exit 2
    ;;
esac

