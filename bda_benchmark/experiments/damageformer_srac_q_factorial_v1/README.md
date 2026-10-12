# Frozen-A1 Q verification and matched comparisons

This package answers where Q loses useful building information before any new Memory is trained. `source` replays the **original E0 front** on Val450 and saves exact same-pass Q, predicted building probability at Q resolution, predictions, image identity and SHA. `audit` reads those frozen artifacts and uses Val labels only offline. `train` then continues from the same SRAC front for 2000 updates per arm, updating only the RGB Q decoder and partition. A1, the feature adapter, region embedding and region classifier stay frozen. `evaluate` reports each completed arm with the historical five-arm E0 reproduction gate unchanged at `1e-5`.

`q_pixel_grid`, `q_region_mean`, `q_grid_probq`, and `q_region` are new-Q readouts. Original `pixel_grid`, `region_mean`, `grid_probq`, A1, region, J1/J2/J3 remain in every evaluation as same-pass references. `q_a1_pixel` separately checks the new model's frozen A1. The original Q source run is `arm=original_q`; a trained `baseline` is a **2000-step continuation**, not the untouched original Q. New Q results do **not** inherit the old Memory's bank, metric projection or Reader validity.

## Server workflow

Run from the BRIGHT repository root in the same legacy/native environment as the passed E0. Each output path must be fresh. The commands below read Train2918 and Val450 only; they do not read Test.

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export Q_EXP=bda_benchmark/experiments/damageformer_srac_q_factorial_v1
export E0_PROTOCOL=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130/protocol
export Q_RUN_ROOT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_Q_FACTORIAL_V1/run_$(date +%Y%m%d_%H%M%S)
unset MATMUL_TF32 CUDNN_TF32 DETERMINISTIC
bash "$Q_EXP/run.sh" preflight
```

Preflight only checks synthetic CPU contracts and module startup. It does **not** establish performance.

First freeze the untouched source Q/B and diagnose the two different kinds of mixing:

```bash
export Q_SOURCE_OUTPUT="$Q_RUN_ROOT/original_q_val"
bash "$Q_EXP/run.sh" source
export Q_AUDIT_SAMEPASS="$Q_SOURCE_OUTPUT"
export Q_AUDIT_OUTPUT="$Q_RUN_ROOT/original_q_diagnostic"
bash "$Q_EXP/run.sh" audit
bash "$Q_EXP/run.sh" summarize
```

The source run is formal only if `summary.json` shows `complete=true`, `full_validation_evidence=true`, `image_count=450`, `reproduction.passed=true`, `q_a1_reproduction.passed=true`, `native_a1_replay_ok=true`, and every `immutable` field true. `audit` is offline interpretation of this frozen same-pass evidence, not a new predictor. Inspect building recall and background dilution separately from Intact/Damaged mixing. If predicted B misses true buildings, do not hard-delete low-B seeds.

After a completed source run, `spatial` reads its saved per-image artifacts to count paired GT-Damaged oracle losses and gains. It stratifies them by four-neighbor GT class-boundary distance, frozen predicted building probability, and soft Q exposure to mixed GT-labelled regions. It also checks the overall change in oracle Damaged false positives. This step is read-only and does not rerun Val or create model predictions:

```bash
export Q_SPATIAL_SAMEPASS="$Q_SOURCE_OUTPUT"
export Q_SPATIAL_OUTPUT="$Q_RUN_ROOT/paired_spatial_original_q"
bash "$Q_EXP/run.sh" spatial
```

The offline report also includes `region_mixing` and `prediction_confusion_by_building_probability`. The first separates labelled Q mass mixing background with buildings from Q mass mixing Intact/Damaged/Destroyed. The second stratifies *already-frozen* A1/Q predictions by five fixed predicted-building-probability bins; every bin retains a 4x4 confusion matrix and the bins sum to the same full-resolution valid-GT denominator. Check whether Damaged misses cluster at low B before proposing an A1 localization retrain. Four-connected raster components in the report are size proxies, not building IDs. `area_grid_oracle` and `q_roundtrip_oracle` deliberately use Val GT and nearest expansion; they are diagnostics, not model performance or a bound on the real probability decoder.

Then train the **matched continuation baseline**, so each subsequent Q arm has the same 2000-update budget and fixed downstream region head. Do not use `best.pth` for formal comparisons.

```bash
export Q_TRAIN_ARM=baseline
export Q_TRAIN_OUTPUT="$Q_RUN_ROOT/train_baseline"
bash "$Q_EXP/run.sh" train
export Q_EVAL_CHECKPOINT="$Q_TRAIN_OUTPUT/last.pth"
export Q_EVAL_OUTPUT="$Q_RUN_ROOT/val_baseline"
bash "$Q_EXP/run.sh" evaluate
```

Only run the experiment matched to the observed failure. For foreground/background mixing, try `building_reconstruction` before any hard foreground-only partition. For genuine Intact/Damaged crossing, try `semantic_reconstruction`. These two losses use Train labels only; Val labels never affect Q or model selection within a run. For a possible SAR boundary cue, **three arms are required together**: `sar_cue`, equal-parameter `rgb_capacity`, and deliberately shifted `shifted_sar`. One isolated `sar_cue` score cannot establish that aligned SAR helped.

For each selected arm, give it its own fresh Train and Val directories:

```bash
export Q_TRAIN_ARM=semantic_reconstruction
export Q_TRAIN_OUTPUT="$Q_RUN_ROOT/train_semantic_reconstruction"
bash "$Q_EXP/run.sh" train
export Q_EVAL_CHECKPOINT="$Q_TRAIN_OUTPUT/last.pth"
export Q_EVAL_OUTPUT="$Q_RUN_ROOT/val_semantic_reconstruction"
bash "$Q_EXP/run.sh" evaluate
export Q_AUDIT_SAMEPASS="$Q_EVAL_OUTPUT"
export Q_AUDIT_OUTPUT="$Q_RUN_ROOT/diagnostic_semantic_reconstruction"
bash "$Q_EXP/run.sh" audit
```

Compare each formal candidate with the matched trained baseline, then inspect class IoUs, Intact-to-Damaged errors and per-event behavior alongside Q/B diagnostics:

```bash
export Q_COMPARE_BASELINE="$Q_RUN_ROOT/val_baseline"
export Q_COMPARE_CANDIDATE="$Q_RUN_ROOT/val_semantic_reconstruction"
export Q_COMPARE_OUTPUT="$Q_RUN_ROOT/compare_semantic_reconstruction"
bash "$Q_EXP/run.sh" compare
```

For actual-model paired rescue, harm, and Intact-to-Damaged false positives, use the saved matched Val predictions. Both sides must pass the full reproduction and integrity gates:

```bash
export Q_PAIRED_OUTPUT="$Q_RUN_ROOT/paired_semantic_reconstruction"
bash "$Q_EXP/run.sh" paired_prediction
```

This writes `report.json` with overall and per-event transition cubes for `q_grid_probq` and `q_region`, verified against recorded confusion matrices. The original saved source run remains valid for read-only diagnosis. Because this package hashes all executable files, evaluate the matched trained baseline and candidate with the **same updated package version** before comparing them.

`--limit` and smoke modes are debugging prefixes; they always retain `full_validation_evidence=false`. The 450-image progress line alone does not mean the final gates passed. Keep failed `summary.json` and per-image artifacts rather than relabelling them as formal. Final structure selection belongs to Train/Val; the previously inspected Test878 must not be used repeatedly for tuning.

This is a staged experiment, not automatic architecture selection. If low-B real buildings dominate errors, train a separate improved localization/A1 branch and then compare old/new Q under **both** A1 versions at the same budget. This package does not silently substitute a new A1 for the archived E0, because the strict reproduction gate and its reference metrics would no longer apply. If an improved Q survives Train/Val, rebuild its Train-only Memory bank and Val query cache; the original bank rows, metric projection and Reader remain controls, not directly interchangeable inputs. The package does not yet train a new A1 or a matching new-Q Memory Reader.

## What this deliberately does not claim

- The original Q has 3 refinement rounds and each pixel still has only its original local 3×3 candidate seeds. More rounds cannot recover a building already absent from the feature grid. This package records Q/B quality but does not silently change rounds, stride, resolution or seed count. Those require separately matched retraining after the frozen diagnostic indicates which bottleneck is real.
- Raising predicted building probability or dropping low-B regions may suppress genuinely missed buildings. The `building_reconstruction` arm tests a Train-only building label objective while keeping the full-image 4-class pathway. It does not turn Q into building instances.
- A new Q changes the Train region tokens and labels used by Memory. Before claiming any Memory gain, rebuild Train references with that Q and compare original/new banks at the same capacity, event/class quotas, and retrieval/Reader budget. Do not use the old J3 bank as if its rows described the new Q.
