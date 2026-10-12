# SRAC Memory metric V2: Train-only retrieval-induced hard queries

This experiment changes **only the search projection**. It keeps the original
Train bank membership, support values, frozen front, Q, J1/J2/J3 and Reader.
It does not train a gate or decide when to accept a reference.

Two identity-initialized `RetrievalProjector(128)` arms use identical reliable
supports (`bank distribution max >= 0.8`), normalized soft-overlap loss,
optimizer, seed, batch size and exactly 2,000 updates:

| Arm | Which Train queries fill the difficult half of each class? |
| --- | --- |
| `j1_hard` | The original J1-hardness upper quartile in all four classes. |
| `mislead` | For Intact: J1-correct Train regions whose **frozen old J3 Top-8 soft-label vote** leans Damaged. For Damaged: J1-missed Train regions. Background/Destroyed retain the J1-hardness upper quartile. |

Both arms also sample ordinary regions. Mixed Train queries remain in the
ordinary pool with their complete four-way soft labels. The Top-8 vote is used
**only for Train mining**; it is not the J3 Reader output or an accuracy bound.
The old full bank and original metric produce this vote using the original
global projected-cosine ranking with same-event and same-image exclusion.
Only selected Train query tokens are read in batches from the verified
memory-mapped Train cache. Formal training fails if either target risk cohort
is absent; it does not tune thresholds on Val.

## Requirements

Use a passed 450-image `legacy/native` E0 from
`damageformer_srac_hardquery_memory_v1`. The Train-only query cache must be
completed using that package's `queries` stage and the **same** E0, front,
bank, metric and data lists. The run refuses smoke/formal mismatches, changed
query shard SHA256 values, source changes, runtime changes or mismatched
checkpoints. Val and Test labels are never used for this fitting.

On the Linux server, from `/root/shared-nvme/BRIGHT-master`:

```bash
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export EXP=bda_benchmark/experiments/damageformer_srac_memory_metric_v2
export JOINT_RUN=/root/shared-nvme/BRIGHT-checkpoints/SRAC_SNAPSHOT_JOINT123_V1/run_20261009_191815/joint
export A1_CKPT=/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth
export E0_ROOT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130
export E0_PROTOCOL="$E0_ROOT/protocol"
export QUERY_RUN="$E0_ROOT/queries"
export PROFILE=legacy POOL_MODE=native
unset MATMUL_TF32 CUDNN_TF32 DETERMINISTIC METRIC_V2_SMOKE
bash "$EXP/run.sh" preflight
```

If `"$QUERY_RUN/query_manifest.json"` is absent, first generate Train queries
with the original package and the same E0. It processes all 2,918 Train
images; this is not Val/Test evaluation:

```bash
export RUN_ROOT="$E0_ROOT"
bash bda_benchmark/experiments/damageformer_srac_hardquery_memory_v1/run.sh queries
```

Then fit both matched arms in a new directory:

```bash
export METRIC_V2_OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_METRIC_V2/fit_$(date +%Y%m%d_%H%M%S)
bash "$EXP/run.sh" train
cat "$METRIC_V2_OUTPUT/summary.json"
```

The outputs are `metric_j1_hard_last.pth`, `metric_mislead_last.pth`,
`summary.json`, `manifest.json` and per-arm training logs. `complete=true`
means fitting and immutable-input checks passed; it **does not** mean the new
projector improved Val. A separate fixed-Reader Val450 evaluator must compare
the two projectors and original J3 under one numerical protocol. Compare
four-class IoU, difficult Damaged rescue, Intact→Damaged harm, Top-8 Train-label
compatibility and per-event results. Use the companion
[`damageformer_srac_memory_metric_eval_v2`](../damageformer_srac_memory_metric_eval_v2/README.md)
for that comparison. `Test878` should not be used to tune the
new method. Do not infer causal benefit solely from a better training loss.

The existing Train J1/front saw these Train images. Its errors are descriptive
hard-case mining, not out-of-fold generalization estimates. Same-event and
same-image supports are excluded even in the mining diagnostic. All Train
region labels describe Q-covered pixel fractions, not building-instance
damage percentages.

The loader calls `torch.load` for metric checkpoints. Load only locally
generated, trusted checkpoint files; do not supply arbitrary downloaded
weights.
