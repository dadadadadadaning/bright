# Existing SRAC region paths on labelled Test878

This is a read-only evaluation of the **existing** A1/front/J1/J2/J3 weights. It
does not rebuild the Memory bank or train the proposed probability baseline or
selective reader. Its prerequisite is the previously passed, full 450-image,
`legacy/native`, all-path E0 protocol. That gate remains at `1e-5`.

The same forward pass provides nine predictions per Test image:

| Name | Exact comparison represented |
| --- | --- |
| `a1_pixel` | Original A1 logits resized to image size, then softmax. |
| `pixel_grid` | Original A1 feature `F` classified on its native grid, softmax, then probability resized. |
| `feature_mean` | Mean-pool `F` by fixed soft Q, project features back, then apply original A1 head. |
| `region_mean` | Mean-pool `F` by Q, apply original A1 head to each region, softmax, then project probabilities back. |
| `grid_probq` | Classify `F`, softmax, average probabilities by Q, then project back. |
| `region` | Original trained SRAC region output, with its adapter, embedding and separate region classifier. |
| `j1` | Existing joint decoder with its control-slot path. |
| `j2` | Existing joint decoder with cosine-retrieved Train Memory. |
| `j3` | Existing joint decoder with metric-retrieved Train Memory. |

The four fixed-Q probes use the exact mean operator from
`damageformer_srac_stage_diagnostics_v1.feature_core`; they do **not** change Q,
F, or the A1 head. `region` and J1/J2/J3 have separately trained classifiers,
so a difference from a probe does not isolate superpixel quality. J2/J3 reuse
the original cross-event Test retrieval policy; ground truth is consulted only
after predictions are fixed, for scoring.

On the Linux GPU host, from the BRIGHT repository root:

```bash
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export EXP=/root/shared-nvme/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_region_test_v1
export JOINT_RUN=/root/shared-nvme/BRIGHT-checkpoints/SRAC_SNAPSHOT_JOINT123_V1/run_20261009_191815/joint
export A1_CKPT=/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth
export DATASET=/root/shared-nvme/BRIGHT-master/bda_benchmark/dataset
export PROTOCOL_RUN=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130/protocol
export OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_TEST_V1/test_$(date +%Y%m%d_%H%M%S)

bash "$EXP/run.sh" preflight
bash "$EXP/run.sh" full
"$PYTHON" -m bda_benchmark.experiments.damageformer_srac_region_test_v1.summarize "$OUTPUT/result.json"
```

If the passed E0 used relocated source artifacts, set `FRONT_CKPT`, `CACHE`, or
`OLD_METRIC_CKPT` to **the same files used for that E0** before running. E0
verification also requires the original GPU, runtime settings, source files,
Train/Val lists, and dataset root. The Test split defaults to
`$DATASET/splitname/standard_ML/test_set.txt`; you can set `TEST_LIST` if the
same standard list has another path. The script refuses an existing `OUTPUT`.

For an input/runtime check only, set a new `OUTPUT` and run
`bash "$EXP/run.sh" prefix` (`MAX_IMAGES` defaults to 2). That result is labelled
as a diagnostic prefix and never as full Test evidence.

`result.json` contains the nine paths' four-class IoU/precision/recall/F1,
foreground mIoU, per-event metrics, and paired GT × reference × candidate
transition counts. `images.jsonl` holds per-image confusion matrices. The
paired rows report Damaged rescues, harms, and false-positive change **in
pixels**, not buildings. The tool checks input filenames, split disjointness,
labelled image-file existence, and Test-file size/modification-time stability;
it does not claim full image-content hashes. Its numerical equivalence probes
are recorded. Native A1 replay failure stops evaluation; other algebraic
diagnostics are recorded even if they differ within the legacy numeric setup.

Use this Test run to report these already frozen comparisons. Select and tune
any later rebuilt Memory on Train/Val; repeatedly choosing designs by Test
scores would make that Test result unsuitable as a final unseen estimate.
