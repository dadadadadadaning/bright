# Frozen-Reader Val450 evaluation of two Train-only Memory search metrics

This evaluates the original J3 projected-cosine Top-8, the **original bank +
source-diverse Top-8** control (`j3_old_diverse`), and two newly trained search
projections, `j1_hard` and `mislead`, on the **same original Train bank**, with
the **same frozen Q/front and original J3 Reader**. No accept/reject gate or
Reader is trained. The only changed inference input is which eight original
bank rows are retrieved. The native A1/Q
same-pass readout is preserved, and the five historical arms must pass their
unchanged absolute `1e-5` reproduction gate after all 450 Val images.

The evaluator requires a completed legacy/native E0 and a completed, hashed
Train-only `damageformer_srac_memory_metric_v2` training output. Debug prefixes
are never formal. Val labels enter only after all supports and predictions are
fixed for each image; Test is not read. New directories must be empty.
The evaluator includes its own native A1/Q replay and retrieval-diagnostic
helpers, so it does not depend on whether a separate bank-v2 replay-fix archive
has been installed on the server.

On the server, after the metric v2 Train stage has completed:

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export E0_PROTOCOL=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130/protocol
export METRIC_V2_OUTPUT=/path/to/completed/metric_v2_train
unset MATMUL_TF32 CUDNN_TF32 DETERMINISTIC METRIC_V2_VAL_LIMIT
bash bda_benchmark/experiments/damageformer_srac_memory_metric_eval_v2/run.sh preflight
export METRIC_V2_VAL_OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_METRIC_V2/val_$(date +%Y%m%d_%H%M%S)
bash bda_benchmark/experiments/damageformer_srac_memory_metric_eval_v2/run.sh evaluate
cat "$METRIC_V2_VAL_OUTPUT/summary.json"
bash bda_benchmark/experiments/damageformer_srac_memory_metric_eval_v2/run.sh report
```

Compare `j3_old_diverse` directly with `j3_j1_hard` and `j3_mislead` in the
same Val pass, as well as each new projection against `j3`. The `mislead`
Train query cohort was mined with the original global Top-8 vote, not the
source-diverse Top-8 or the frozen Reader's actual pixel changes. Read
`audit.json` for per-class IoU, paired transitions, per-event metrics,
support-source changes, and post-hoc query/neighbor soft-label compatibility.
The direct effect of changing the projection on a frozen Reader can be smaller
than its retrieval-quality improvement because the Reader was fitted with the
old neighbor distribution. Use Train/Val only for choosing a later design;
do not repeatedly tune on the previously inspected Test878.
