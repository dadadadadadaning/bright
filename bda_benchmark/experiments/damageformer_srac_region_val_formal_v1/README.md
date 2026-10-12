# Frozen Val450 region/Q and Memory paths

This package reads only the already-approved **Val450** list from a successful
legacy/native all-path E0. It does not train, change weights, build a bank, or
access Test. One frozen front forward per image yields A1, Q, the original
region output and J1/J2/J3. J2/J3 support is selected immediately after that
forward, before the four additional Q probes, to follow the E0 inference order.

The nine reported paths are `a1_pixel`, `pixel_grid`, `feature_mean`,
`region_mean`, `grid_probq`, `region`, `j1`, `j2`, `j3`. The four Q probes are
the unchanged formulas from the existing Test-region experiment. `region` is
the same native output called `direct_region` in E0. `a1_pixel` and
`pixel_grid` use different resize/probability orders, so their difference is
not a pure Q effect. The cleanest frozen-Q contrast is `pixel_grid` versus
`grid_probq`. The J arms are separately trained paths; their cross-path
differences do not isolate a causal effect of Memory.

## Run on the server

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
bash bda_benchmark/experiments/damageformer_srac_region_val_formal_v1/run.sh preflight

export E0_PROTOCOL=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130/protocol
export REGION_VAL_OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130/region_val_$(date +%Y%m%d_%H%M%S)
unset MATMUL_TF32 CUDNN_TF32 DETERMINISTIC REGION_VAL_LIMIT
bash bda_benchmark/experiments/damageformer_srac_region_val_formal_v1/run.sh validate
bash bda_benchmark/experiments/damageformer_srac_region_val_formal_v1/run.sh report
```

For a nonformal smoke prefix set `REGION_VAL_LIMIT=2` and use a **new** output
directory. A prefix cannot be labelled full Val evidence.

First read `summary.json`. `result.json` is formal only when
`full_validation_evidence=true`, the unchanged historical five-arm
`1e-5` classification reproduction passes, every immutability check passes,
the pool guard is clean, and the native A1 replay check passes. A full failed
run preserves its maps and report but is **not** promoted to valid evidence.
No exact E0 Top-8 row-ID equality is required: the legacy/native CUDA path can
change near-tied support ranks without failing the original classification
reproduction requirement.

`result.json` includes four-class and foreground mIoU for every path, paired
prediction transition counts, and per-event metrics. `images.jsonl` and
`predictions/*.npz` preserve each Val image's GT, nine predictions, fresh
J2/J3 support IDs, confusion matrices and SHA256. These are pixels, not
building-instance counts. The extra Q branches are frozen probes, not a new
trained `grid_probq + Memory` model.
The `report` command reads only the saved JSON and refuses to print a score
table unless the formal Val450, reproduction and immutability gates passed.
