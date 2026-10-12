#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN="$HERE/run.sh"

# This script deliberately assumes Stage-A P-SC has completed.  Set RUN_STAGE_A=1
# only if you really want to launch the expensive V2 ablation tree first.
if [[ "${RUN_STAGE_A:-0}" == "1" ]]; then
  bash "$HERE/run_stage_a_all.sh"
fi

bash "$RUN" preflight
bash "$RUN" cache_stage1
bash "$RUN" audit_region

# Stage-A feedback utility is evaluated only if Stage-2 exists.
if [[ -s "${STAGE2_CKPT:-/root/data/BRIGHT-checkpoints/DAMAGEFORMER_BUILDING_INTERNAL_CHANGE_PARTITION_V2/p_sc_residual_seed${SEED:-3407}/stage2_feedback/latest.pth}" ]]; then
  bash "$RUN" cache_stage2_val
  bash "$RUN" compare_feedback
fi

# B0: frozen representation diagnostics. 2000 cheap cached-feature steps each.
for M in ce balanced_ce blv seesaw logit_adjust; do
  bash "$RUN" tail_probe "$M"
done

# B1: event-balanced multi-prototype separation. 3000 cached-feature steps each.
for K in 1 2 4; do
  bash "$RUN" multi_proto "$K"
done
python -u -m bda_benchmark.experiments.damageformer_causal_research_chain_v1.select_best_tail \
  --root "${CHAIN_OUT_ROOT:-/root/data/BRIGHT-checkpoints/DAMAGEFORMER_CAUSAL_RESEARCH_CHAIN_V1}/B_multi_proto" --seed "${SEED:-3407}"

# C: first headroom, then learned specialist. 3000 steps.
bash "$RUN" expert_oracle
bash "$RUN" expert_train

# D: class-level historical knowledge; no network training.
for K in 1 2 4; do
  bash "$RUN" proto_oracle "$K"
done

# E: instance memory oracle + mandatory negative controls; no verifier training yet.
for C in none shuffle_labels shuffle_events wrong_regions; do
  bash "$RUN" memory_oracle "$C"
done

# E-final: train-only calibration verifier, 3000 steps; Validation is read-only.
bash "$RUN" memory_verifier

echo "ALL DOWNSTREAM CAUSAL-CHAIN EXPERIMENTS COMPLETED"
