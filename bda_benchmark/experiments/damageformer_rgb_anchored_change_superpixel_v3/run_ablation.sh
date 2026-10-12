#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-quick}"
HERE="$(cd "$(dirname "$0")" && pwd)"

case "${PHASE}" in
  quick)
    export STAGE0_STEPS="${STAGE0_STEPS:-500}"
    export STAGE1_STEPS="${STAGE1_STEPS:-1000}"
    export STAGE0_EVAL_EVERY="${STAGE0_EVAL_EVERY:-250}"
    export STAGE1_EVAL_EVERY="${STAGE1_EVAL_EVERY:-500}"
    export VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-64}"
    LEGACY_PHASE="${LEGACY_PHASE:-probe}"
    ;;
  full)
    export STAGE0_STEPS="${STAGE0_STEPS:-1000}"
    export STAGE1_STEPS="${STAGE1_STEPS:-3000}"
    export VAL_MAX_BATCHES="${VAL_MAX_BATCHES:-0}"
    LEGACY_PHASE="${LEGACY_PHASE:-confirm}"
    ;;
  *) echo "Usage: $0 {quick|full}" >&2; exit 2 ;;
esac

# P0/P1: fixed partition-only controls.
bash "${HERE}/run.sh" audit_grid p2_rgb_structure
bash "${HERE}/run.sh" audit_slico p2_rgb_structure

# P3/P4: already isolated one-step joint-partition controls.  They are kept
# under their original protocol because their architecture predates V3.
if [[ "${RUN_LEGACY_JOINT_CONTROLS:-1}" == "1" ]]; then
  bash "${HERE}/../damageformer_a1_cross_modal_superpixel_v1/run.sh" "${LEGACY_PHASE}" c4_raw_sar_assoc
  bash "${HERE}/../damageformer_a1_cross_modal_superpixel_v1/run.sh" "${LEGACY_PHASE}" c5_aligned_assoc
fi

# P2 and P5-P8: matched V3 trainable partition family.
for variant in \
  p2_rgb_structure \
  p5_anchored_change \
  p6_shuffled_sar \
  p7_no_confidence \
  p8_no_damage_boundary
do
  echo "===== ${variant} START $(date -Iseconds) ====="
  bash "${HERE}/run.sh" pipeline "${variant}"
  bash "${HERE}/run.sh" audit_learned "${variant}"
  echo "===== ${variant} COMPLETE $(date -Iseconds) ====="
done

python -m bda_benchmark.experiments.damageformer_rgb_anchored_change_superpixel_v3.summarize \
  --audit_root "${AUDIT_ROOT:-/root/data/BRIGHT-eval-results/DAMAGEFORMER_RGB_ANCHORED_CHANGE_SUPERPIXEL_V3}"
