#!/usr/bin/env bash
set -euo pipefail

# Comprehensive launcher:
# 1) overall Building Context model FIRST
# 2) all original region ablations C1-C9
# 3) Building Context controlled variants
#
# C0/A1 is evaluation-only and can be audited separately.

PHASE="${1:-probe}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${HERE}/../.." && pwd)"
REPO_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "============================================================"
echo "STEP 1: OVERALL MODEL FIRST: full_joint ${PHASE}"
echo "============================================================"
bash "${HERE}/run.sh" "${PHASE}" full_joint

echo "============================================================"
echo "STEP 2: ORIGINAL REGION ABLATIONS C1-C9 ${PHASE}"
echo "============================================================"
bash experiments/damageformer_a1_cross_modal_superpixel_v1/run_sequential.sh \
  "${PHASE}" \
  c1_lca \
  c2_slico \
  c3_rgb_assoc \
  c4_raw_sar_assoc \
  c5_aligned_assoc \
  c6_no_confidence \
  c7_shuffled_sar \
  c8_damage_hint \
  c9_boundary

echo "============================================================"
echo "STEP 3: BUILDING CONTEXT CONTROLLED VARIANTS ${PHASE}"
echo "============================================================"

# full_joint already ran above, so do the remaining ones.
for variant in \
  bir_on_frozen_c5 \
  bir_shuffle_on_frozen_c5 \
  rgb_only_building \
  raw_sar_building \
  no_conf_building \
  shifted_sar_building \
  wrong_building_joint
do
  bash "${HERE}/run.sh" "${PHASE}" "${variant}"
done

echo "============================================================"
echo "ALL ${PHASE} EXPERIMENTS COMPLETE"
echo "============================================================"
