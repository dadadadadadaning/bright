# SRAC Train Memory: quality candidate pool + projected diversity

This is an **independent bank-selection experiment**. It does not train a
gate, front, Q, J3 metric, or Reader. It does not read Test, and Val labels
never select entries or neighbors. Its only changed variable relative to the
original frozen J3 run is which real Train Q-region rows enter a same-capacity
bank. It has its own manifest/evaluator/source hashes; a bank from this
package cannot be silently accepted as a v2 quality bank.

## Bank selection

For every original event × dominant-class bucket, preserve its exact quota.
Read all 2,918 SHA-verified Train shards associated with a passed legacy/native
E0. Keep a bounded candidate pool (default 4 × bucket quota), initially ranked
by the **original four-way soft-label purity**, then Q mass capped at 16, then
a seeded stable tie. These values only choose real Train rows; the stored
four-way soft label, token, geometry and source IDs are copied unchanged.

Project candidate tokens with the **frozen original J3 retrieval projection**.
Start with the highest-quality candidate. Repeatedly select the candidate
farthest from its closest already selected key in cosine space, with a fixed
small purity contribution (default 0.2). Prefer at most four entries per Train
image within one bucket; relax this cap only when every remaining candidate
exceeds it, and record each relaxation. This is a coverage heuristic, not a
claim that a Q region is a building or that each selected entry is reliable.
Its quality prefilter can still exclude useful small/mixed Damaged examples;
compare the recorded per-class purity and downstream retrieval/IoU before
adopting it.

The evaluation repeats the frozen same-pass Val450 factorial with five
historical arms and `grid_probq`. New-bank arms are `j3_divbank_top8` and
`j3_divbank_diverse`; the latter additionally uses the v2 Top-64 source
diversity rule. Original-bank arms are `j3` and `j3_old_diverse`. All share one
front/Q pass, the original J3 metric/Reader, and unchanged 1e-5 historical
metric reproduction gate. A debug prefix is never formal.

## Server commands

Run from `/root/shared-nvme/BRIGHT-master` after transferring this package:

```bash
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export E0_PROTOCOL=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130/protocol
unset MATMUL_TF32 CUDNN_TF32 DETERMINISTIC BANK_DIV_LIMIT
bash bda_benchmark/experiments/damageformer_srac_memory_bank_diverse_v1/run.sh preflight

export BANK_DIV_OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_BANK_DIVERSE_V1/bank_$(date +%Y%m%d_%H%M%S)
bash bda_benchmark/experiments/damageformer_srac_memory_bank_diverse_v1/run.sh build
cat "$BANK_DIV_OUTPUT/summary.json"

export BANK_DIV_VAL_OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_BANK_DIVERSE_V1/val_$(date +%Y%m%d_%H%M%S)
bash bda_benchmark/experiments/damageformer_srac_memory_bank_diverse_v1/run.sh evaluate
cat "$BANK_DIV_VAL_OUTPUT/summary.json"
bash bda_benchmark/experiments/damageformer_srac_memory_bank_diverse_v1/run.sh report
```

Keep `BANK_DIV_OUTPUT` set to the **completed** bank directory. Both output
directories must be fresh. Optional variables: `BANK_DIV_SEED`,
`BANK_DIV_PER_IMAGE_CAP`, `BANK_DIV_CANDIDATE_MULTIPLIER`,
`BANK_DIV_QUALITY_WEIGHT`; leave defaults unchanged for the first comparison.
`BANK_DIV_LIMIT=N` only runs a diagnostic Val prefix and cannot be a formal
result.

The build output stores `manifest.json`, `selected_rows.json`, and `bank.pt`.
Its per-bucket `old_key_redundancy` and `new_key_redundancy` compare nearest
same-bucket projected cosine; a lower value indicates less duplication but is
**not** itself evidence of improved segmentation.
The formal Val output stores `summary.json`, `audit.json`, `images.jsonl`, and
per-image SHA-checked predictions/support IDs. Report a new-bank score only
when `summary.json` says `complete=true`, `full_validation_evidence=true`,
and `reproduction_passed=true`. Analyze Damaged rescue **and** Intact→Damaged
harm, all four IoUs, source duplication, label compatibility, and per-event
results. The old/new bank input intervention is measured under a frozen Reader;
its result does not establish how a retrained Reader would behave.
