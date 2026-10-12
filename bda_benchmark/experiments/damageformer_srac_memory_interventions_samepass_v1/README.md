# Same-pass frozen-Memory support interventions

This read-only Val450 audit uses the same checked E0/E1 weights, Train bank,
split and legacy/native numerical settings. Each image has one front forward:
the resulting Q, J2/J3 Top-8 and all predictions belong to that same pass.
It **does not require exact Top-8 row-order equality to the earlier E0 run**.
The normal `a1_pixel`, `direct_region`, `j1`, `j2`, `j3` results must still
reproduce the original historical reference for all scalar/class IoU,
precision, recall and F1 values at the unchanged absolute `1e-5` threshold.
The prior E1 is required for source identity and completion, but its integer
confusion matrices are recorded as an informative comparison rather than a
second, stricter gate for an independent native-CUDA replay.

For each fixed, trained J2/J3 decoder, this program compares normal Top-8
support with eligible uniform-random Train rows, shuffled soft labels on the
same retrieved rows, and all support slots invalid. Label shuffling assigns
one permuted soft label to each *unique retrieved bank row* within an image;
the same historical row never has conflicting labels across query regions.
The random condition
preserves the number of valid slots and same-event/image exclusions, but it
does not match the event or class distribution. Label shuffling keeps the
keys and metadata in place and preserves the multiset of labels among unique
retrieved rows, but may make few effective changes if an image's neighbors
have similar labels. It does not necessarily preserve the occurrence-weighted
class histogram when a bank row appears in multiple Top-8 lists. The `off`
condition invokes that same Reader's
learned-slot fallback; it does **not** return to separately trained J1 or A1.
These are frozen-model dependence tests, not a causal estimate of gains from
retrieval versus a newly trained capacity-matched control.

Run on the Linux GPU server after the passed legacy/native E0/E1:

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export REFERENCE_ROOT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130
export OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_INTERVENTIONS_SAMEPASS_V1/run_$(date +%Y%m%d_%H%M%S)
unset MATMUL_TF32 CUDNN_TF32 DETERMINISTIC RANDOM_SEEDS SHUFFLE_SEED
bash bda_benchmark/experiments/damageformer_srac_memory_interventions_samepass_v1/run.sh preflight
bash bda_benchmark/experiments/damageformer_srac_memory_interventions_samepass_v1/run.sh diagnose
bash bda_benchmark/experiments/damageformer_srac_memory_interventions_samepass_v1/run.sh report
```

The default uses one random seed (`3407`) to keep the initial Val450 audit
tractable. This package samples eligible rows exactly uniformly without
replacement using only the required Top-8 draws per query. Repeating three
seeds still roughly triples the random-control decoder cost. If
random-vs-normal changes are important, repeat the diagnostic
with `export RANDOM_SEEDS='3407 3408 3409'` and a **new** `OUTPUT`, or compare
multiple independent one-seed runs. Each seed appears in the arm name, e.g.
`j3_random_3407`.

Inspect `summary.json` first. Only `full_validation_evidence=true` permits
formal interpretation of `audit.json` metrics, transitions and per-event data.
Per-image Top-8 IDs, confusions and transition counts are saved in hashed
`image_evidence/*.npz` files and checked against the ordered journal before
the final report is marked complete.
`diagnostic_unverified.json` is retained when the formal gate fails, but must
not be presented as a successful Memory result. This run uses Val only; no
Test images or training are involved. Keep the existing Test878 result for
reporting the previously frozen models, not choosing these interventions.
