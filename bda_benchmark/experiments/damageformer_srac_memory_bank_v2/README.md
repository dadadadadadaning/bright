# SRAC Memory bank × retrieval factorial, frozen J3

This package tests **how the historical Memory is built and searched**. It does
not retrain the front, Q, metric projection or Reader, alter a checkpoint, or
use Val/Test labels for the bank or retrieval. The only new support bank is
built from the 2,918 hashed Train region shards already associated with the
passed legacy/native E0. The original Train bank remains read-only.

## Interventions

| Reader arm | Bank | Top-8 selection |
| --- | --- | --- |
| `j3` | Original | Original projected cosine Top-8 |
| `j3_old_diverse` | Original | Top-64 shortlist, rank-ordered image/event diversity |
| `j3_quality_top8` | New quality-prioritized | Original projected cosine Top-8 |
| `j3_quality_diverse` | New quality-prioritized | Same Top-64 diversity rule |

The new bank preserves **exact total capacity and every event × dominant-class
count**. Each stored row retains its four-way soft label, feature, geometry,
event, image and region identity. Within a bucket, Train label purity ranks
eligible rows; Q mass capped at 16 and a stable seeded hash break ties. The
builder prefers a per-image limit and records any relaxation needed to fill the
original bucket quota. No rule asserts that Q is a building instance.

The diversity rule first admits at most one support per source image and two
per source event. If fewer than eight supports are available within the Top-64,
it relaxes the image cap, then the event cap, always using original similarity
rank order. Same-event/same-image query exclusions remain strict. No Val label,
baseline class prediction or query truth enters this selection.

All four variants share **one front/Q forward per Val image**. The same frozen
J3 Reader then runs once for each variant's supports. A1, direct region, J1,
J2, original J3 and `grid_probq` are included.
The five historical arms must pass the unchanged **absolute 1e-5 class/scalar
reproduction gate** for a formal Val450 result. Top-8 row ID equality with a
different E0 replay is not a gate because independent native CUDA forwards may
slightly change ranks. A debug prefix (`BANK_V2_LIMIT`) is never formal.

## Server commands

From `/root/shared-nvme/BRIGHT-master` after installing the code:

```bash
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export E0_PROTOCOL=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130/protocol
unset MATMUL_TF32 CUDNN_TF32 DETERMINISTIC BANK_V2_LIMIT
bash bda_benchmark/experiments/damageformer_srac_memory_bank_v2/run.sh preflight
export BANK_V2_OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_BANK_V2/bank_$(date +%Y%m%d_%H%M%S)
bash bda_benchmark/experiments/damageformer_srac_memory_bank_v2/run.sh build
export BANK_V2_VAL_OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_BANK_V2/val_$(date +%Y%m%d_%H%M%S)
bash bda_benchmark/experiments/damageformer_srac_memory_bank_v2/run.sh evaluate
cat $BANK_V2_VAL_OUTPUT/summary.json
bash bda_benchmark/experiments/damageformer_srac_memory_bank_v2/run.sh report
```

Keep `BANK_V2_OUTPUT` set to the completed bank directory across build and
evaluation. If a shell closes, restore it from the printed path. All output
directories must be fresh. The bank builder reads and hashes every Train shard;
the evaluator verifies the manifest, ledger, bank SHA, exact quotas, source code
and unchanged original E0 bundle, then locks the new bank SHA for Val450.

`audit.json` contains full pixel metrics, all class IoUs, paired rescue/harm and
false-positive changes, per-event metrics/transitions, support diversity,
rank/set changes based on Train (image, region) identities, diversity fallback
counts and post-hoc Q soft-label compatibility. `images.jsonl` binds every
Val image to a SHA256-checked NPZ containing its target, all prediction maps and
each Top-8 ID array. Val GT enters only after all four predictions are fixed.

`grid_probq` is a fixed-Q probability readout, not a Memory arm. The reported
Q-purity strata are observational; Q regions are not building instances. Soft
compatibility is the dot product of a query's four-way label fraction and its
neighbors' mean four-way fraction. Even perfectly matched mixed labels can
score lower than pure labels, so compare the four arms **within the same Q
stratum**, not the mixed and pure means against each other. A
larger score for a rebuilt bank is evidence about this frozen J3 Reader under
that input intervention, not a guarantee that retraining a Reader would have
the same effect. A lower score can reflect input distribution shift. Use Train
and Val to choose a later design; the previously viewed Test878 should not be
repeatedly used for this choice.
