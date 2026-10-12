#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/root/data/BRIGHT-master/bda_benchmark}"
RUN="${PROJECT_ROOT}/experiments/damageformer_building_internal_change_partition_v2/run.sh"
STORAGE_ROOT="${STORAGE_ROOT:-/root/data}"
SEED="${SEED:-3407}"
export STORAGE_ROOT SEED
export DEVICE="${DEVICE:-cuda:0}"
export AMP_DTYPE="${AMP_DTYPE:-bf16}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
export ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-8}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export STAGE0_STEPS="${STAGE0_STEPS:-1000}"
export STAGE0_EVAL_EVERY="${STAGE0_EVAL_EVERY:-500}"
export STAGE1_STEPS="${STAGE1_STEPS:-3000}"
export STAGE1_EVAL_EVERY="${STAGE1_EVAL_EVERY:-500}"
export STAGE2_STEPS="${STAGE2_STEPS:-5000}"
export STAGE2_EVAL_EVERY="${STAGE2_EVAL_EVERY:-500}"
export SAVE_EVERY="${SAVE_EVERY:-5000}"
export VAL_MAX_BATCHES=0

OUT_ROOT="${STORAGE_ROOT}/BRIGHT-checkpoints/DAMAGEFORMER_BUILDING_INTERNAL_CHANGE_PARTITION_V2"
LOG_ROOT="${STORAGE_ROOT}/BRIGHT-train-logs/DAMAGEFORMER_BUILDING_INTERNAL_CHANGE_PARTITION_V2"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
cd "$PROJECT_ROOT"

bash "$RUN" preflight p_sc_residual
bash "$RUN" exact_replay p_sc_residual
bash "$RUN" stage0 p_sc_residual
bash "$RUN" audit0 p_sc_residual
MAIN0="$OUT_ROOT/p_sc_residual_seed${SEED}/stage0_structure/latest.pth"

STAGE0_INIT_CKPT="$MAIN0" bash "$RUN" stage1 p_sc_residual
bash "$RUN" audit1 p_sc_residual
MAIN1="$OUT_ROOT/p_sc_residual_seed${SEED}/stage1_change/latest.pth"
STAGE1_INIT_CKPT="$MAIN1" bash "$RUN" stage2 p_sc_residual
bash "$RUN" audit2 p_sc_residual

for V in p_s_struct_only p_c_change_only p_sc_no_peak p_sc_mean_only p_sc_no_bag p_sc_shuffled_sar; do
  STAGE0_INIT_CKPT="$MAIN0" bash "$RUN" stage1 "$V"
  CHECKPOINT="$OUT_ROOT/${V}_seed${SEED}/stage1_change/latest.pth" AUDIT_TAG=stage1_latest bash "$RUN" audit_custom "$V"
  STAGE1_INIT_CKPT="$OUT_ROOT/${V}_seed${SEED}/stage1_change/latest.pth" bash "$RUN" stage2 "$V"
  bash "$RUN" audit2 "$V"
done

STAGE1_INIT_CKPT="$MAIN1" bash "$RUN" stage2 p_sc_no_uncertainty
bash "$RUN" audit2 p_sc_no_uncertainty

echo "Stage-A complete. Main steps=1000+3000+5000; each Stage1/2 ablation=3000+5000; no-uncertainty=5000 only."
