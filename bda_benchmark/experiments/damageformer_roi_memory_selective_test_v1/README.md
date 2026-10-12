# Fixed selective-Memory Test878 evaluation

Run only after Train/Val has fixed the four **last** posterior heads and all
retrieval settings. This new package does not change the trained model, bank,
or old experiment files. It loads the complete formal posterior run and
compares `source_hard`, `query_only`, `posterior_nogate`,
`posterior_query_gate`, and `posterior_quality_gate` on the same 878 Test images.
`source_hard` is the frozen region source model, not the original historical A1
checkpoint. The primary Memory comparison is against `query_only`.

The evaluator requires Test GT under `dataset/target`. Missing GT aborts before
Test inference; no mIoU is invented. It checks Train2918/Val450/Test878
disjointness, trained final step and each last checkpoint, source/front/bank
identities and file hashes, and the exact Train-only retrieval settings. It first
recomputes full Val450 with the four final heads. Every head and `source_hard`
must reproduce its trained mIoU within `0.005` percentage points and its
confusion matrix within L1 fraction `1e-5`; otherwise it writes
`val_reproduction.json`, stops, and never loads Test images or labels. The Test
ID list is checked for split overlap before this replay. The old soft-region branch
is retained only as a diagnostic. This Val check does not choose a checkpoint.
No Test images are used for checkpoint, threshold, or hyperparameter selection.

Copy this whole package to the same relative location on the server. Then:

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
bash bda_benchmark/experiments/damageformer_roi_memory_selective_test_v1/run.sh preflight

export ALLOW_TEST_ACCESS=YES
export POSTERIOR_RUN=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_SELECTIVE_V1/formal5000_seed3407_20261008_091529/posterior
export OUTPUT_ROOT=/root/shared-nvme/BRIGHT-checkpoints/ROI_MEMORY_SELECTIVE_TEST_V1/test878_seed3407
bash bda_benchmark/experiments/damageformer_roi_memory_selective_test_v1/run.sh test
```

Use a new `OUTPUT_ROOT` if that path already exists. `summary.json` gives
four-class metrics, per-event metrics, and pairwise rescues/harms relative to
`query_only`; `audit.json` also includes location and image-level diagnostics.
No pixel masks, predictions, model weights, or new Memory bank are saved.
If Test labels are missing, the check stops after creating an output directory
with the Val replay result; use another fresh `OUTPUT_ROOT` after fixing the data.

Test is a final, one-time evaluation. Do not select a variant based on Test
and rerun the comparison with adjusted settings.
