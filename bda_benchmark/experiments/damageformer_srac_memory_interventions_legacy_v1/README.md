# Legacy/native Memory intervention audit

This is an **inference-only** matched-weight test of the already-trained J2/J3
Readers. It leaves the hashed hardquery package, model checkpoints, Memory bank,
and E0/E1 artifacts untouched. The entry point rejects a stale or failed E0,
checks the exact source hashes and numerical settings through
`hardquery_memory_v1.io_utils.load_verified_bundle`, and requires a passed formal
450-image E1. It then reruns all five historical arms. **Every original confusion
matrix must exactly match E1**, and the unchanged historical `1e-5` reproduction
gate must pass, before intervention outcomes are marked complete.

For each of J2 (cosine retrieval) and J3 (old metric retrieval), the same trained
decoder runs under four support conditions:

| Condition | Change to original support |
| --- | --- |
| normal | Original cross-event Top-8, labels and metadata |
| shuffle_labels | Same support IDs, keys, geometry/events and valid slots; shuffle the valid soft label vectors within the current image |
| random | Uniformly sampled eligible Train rows; same number of valid slots and same event/image exclusion |
| off | Mask all historical support slots; this Reader uses its own learned-slot fallback |

The `off` result is **not** J1: J1 has separately trained Reader and pixel
classifier weights. These interventions measure whether each *frozen* J2/J3
model depends on its selected supports and labels; they do not measure what a
model retrained for each intervention could achieve. A score drop after random
support or label shuffling also reflects input distribution shift; it is not an
isolated estimate of retrieval's potential gain. `shuffle_labels` may have
few effective changes in class-homogeneous images; the output records the count.

Run on the original Linux training server, with the same checked paths and GPU
environment as the passed legacy/native E0/E1:

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export REFERENCE_ROOT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130
export OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_INTERVENTIONS_LEGACY_V1/run_$(date +%Y%m%d_%H%M%S)
bash bda_benchmark/experiments/damageformer_srac_memory_interventions_legacy_v1/run.sh
```

The launcher reads every model path, split path, device and numeric override from
`REFERENCE_ROOT/protocol/protocol.json`; it does not guess or replace them. For
orientation, this run's known source paths were
`/root/shared-nvme/BRIGHT-checkpoints/SRAC_SNAPSHOT_JOINT123_V1/run_20261009_191815/joint`
and `/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth`; the E0 record
remains authoritative. A code-hash mismatch means the current source no longer matches
the passed E0; obtain a fresh matched E0/E1 under the unchanged reproduction
threshold rather than disabling the identity check.

The new output contains `summary.json` with four-class scores, event breakdowns,
and conditional rescue/harm/false-positive transitions relative to normal J2/J3
and J1, plus `images.jsonl`. Val targets
are used only for scoring *after* predictions; no Test data or training is used.

CPU contract check: `python -m unittest bda_benchmark.experiments.damageformer_srac_memory_interventions_legacy_v1.test_evaluate -v`.
