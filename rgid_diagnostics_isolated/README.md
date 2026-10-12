# RGID GT/Center staged experiments (isolated)

This directory is independent of the existing model and experiment code. It
only imports an RGID experiment selected with `--experiment-root`; it never
edits that experiment.

## What is implemented

- Stage 1: lightweight damage-class-CC GT sidecars, V2 training/calibration,
  G0-G3 evaluation, and the complete 2x2 V1/V2 × binary/damage GT comparison.
- Stage 2: spatial Center precision/recall at 1/2/3/5 head pixels and sequential
  loss attribution across local maxima, threshold, distance NMS, marker cap,
  and parent loss at head resolution.
- Stage 4 evaluation contract: A pixel damage, B parent aggregation, and C RGID
  instance aggregation, including pixel metrics, GT-instance-conditional
  building damage metrics, and per-event metrics.

Stage 3 structural changes are intentionally not applied before Stage 2 says
which failure dominates. Threshold calibration is appropriate only when the
audit shows threshold filtering; high resolution is appropriate only for
localization/small-parent losses; hard negatives are appropriate for duplicate
peaks.

## Storage design

The sidecar stores only replacement `instance_ids`. It does not copy
`fused_roi` or `parent_mask`. Expected size is roughly 256 MiB for 64 training
samples and 128 MiB for 32 validation samples, rather than another 6.9 GiB.

## Server variables

```bash
ROOT=/root/shared-nvme/BRIGHT-master
FIXED="$ROOT/bda_benchmark/experiments/rgid_s9ic_rebuilt_v2"
TOOLS="$ROOT/rgid_gt_center_experiments_isolated"
DATA="$ROOT/bda_benchmark/dataset"

RUN=/root/shared-nvme/BRIGHT-checkpoints/S9Ic_RGID/rebuilt_v2_20260831
TRAIN_CACHE="$RUN/cache/train64_balanced_p3_retry1"
VAL_CACHE="$RUN/cache/val32_balanced_p3_rebuilt1"

V1_HEAD="$RUN/head_runs/logit_v1_seed3407"
V1_CKPT="$V1_HEAD/rgid_step_500_calibrated.pth"

V2="$RUN/damagecc_v2_seed3407"
TRAIN_GT="$V2/gt_sidecars/train64"
VAL_GT="$V2/gt_sidecars/val32"
V2_HEAD="$V2/head"
EVAL="$V2/evaluations"
mkdir -p "$V2/logs" "$EVAL"
```

The training cache must still contain 64 `.pt` files. If it was deleted for
space, regenerate it from the unchanged train64 list and P3 checkpoint before
continuing. Do not point `TRAIN_CACHE` at an empty placeholder directory.

## Stage 1A: build V2 GT sidecars

```bash
python -u "$TOOLS/build_gt_sidecars.py" \
  --cache-dir "$TRAIN_CACHE" \
  --dataset-root "$DATA" \
  --output-dir "$TRAIN_GT" \
  2>&1 | tee "$V2/logs/build_train_gt.log"

python -u "$TOOLS/build_gt_sidecars.py" \
  --cache-dir "$VAL_CACHE" \
  --dataset-root "$DATA" \
  --output-dir "$VAL_GT" \
  2>&1 | tee "$V2/logs/build_val_gt.log"
```

Verify counts and manifests:

```bash
test -f "$TRAIN_GT/manifest.json" && echo "train GT OK"
test -f "$VAL_GT/manifest.json" && echo "val GT OK"
echo "train sidecars: $(find "$TRAIN_GT" -maxdepth 1 -name '*.pt' | wc -l)"
echo "val sidecars:   $(find "$VAL_GT" -maxdepth 1 -name '*.pt' | wc -l)"
du -sh "$TRAIN_GT" "$VAL_GT"
```

## Stage 1B: train and calibrate V2

This uses the repaired logit-space focal loss from `rgid_s9ic_rebuilt_v2`.
Target creation remains CPU-heavy, so the run may be slow; do not infer GPU
failure from low instantaneous utilization.

```bash
python -u "$TOOLS/train_gt_variant.py" \
  --experiment-root "$FIXED" \
  --train-cache "$TRAIN_CACHE" \
  --gt-sidecar "$TRAIN_GT" \
  --output-dir "$V2_HEAD" \
  --steps 500 \
  --batch-size 4 \
  --accumulation 1 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --checkpoint-steps 100,300,500 \
  --num-workers 0 \
  --device cuda \
  --seed 3407 \
  2>&1 | tee "$V2/logs/train_v2.log"

python -u "$TOOLS/calibrate_gt_variant.py" \
  --experiment-root "$FIXED" \
  --checkpoint "$V2_HEAD/rgid_step_500.pth" \
  --train-cache "$TRAIN_CACHE" \
  --gt-sidecar "$TRAIN_GT" \
  --output "$V2_HEAD/rgid_step_500_calibrated.pth" \
  --batch-size 4 \
  --device cuda \
  2>&1 | tee "$V2/logs/calibrate_v2_step500.log"
```

## Stage 1C: complete 2x2 evaluation

The same validation features and parents are used in all four cells.

```bash
# V1 model evaluated on binary GT
python -u "$TOOLS/evaluate_gt_variant.py" \
  --experiment-root "$FIXED" --checkpoint "$V1_CKPT" \
  --val-cache "$VAL_CACHE" \
  --output "$EVAL/v1_model_binary_gt.json" --device cuda

# V1 model evaluated on damage-class GT
python -u "$TOOLS/evaluate_gt_variant.py" \
  --experiment-root "$FIXED" --checkpoint "$V1_CKPT" \
  --val-cache "$VAL_CACHE" --gt-sidecar "$VAL_GT" \
  --output "$EVAL/v1_model_damage_gt.json" --device cuda

# V2 model evaluated on binary GT
python -u "$TOOLS/evaluate_gt_variant.py" \
  --experiment-root "$FIXED" \
  --checkpoint "$V2_HEAD/rgid_step_500_calibrated.pth" \
  --val-cache "$VAL_CACHE" \
  --output "$EVAL/v2_model_binary_gt.json" --device cuda

# V2 model evaluated on damage-class GT (primary comparison)
python -u "$TOOLS/evaluate_gt_variant.py" \
  --experiment-root "$FIXED" \
  --checkpoint "$V2_HEAD/rgid_step_500_calibrated.pth" \
  --val-cache "$VAL_CACHE" --gt-sidecar "$VAL_GT" \
  --output "$EVAL/v2_model_damage_gt.json" --device cuda
```

Summarize:

```bash
python "$TOOLS/summarize_stage1.py" \
  --v1-binary "$EVAL/v1_model_binary_gt.json" \
  --v1-damage "$EVAL/v1_model_damage_gt.json" \
  --v2-binary "$EVAL/v2_model_binary_gt.json" \
  --v2-damage "$EVAL/v2_model_damage_gt.json"
```

The causal comparison is V1 model versus V2 model under the same damage GT.
The binary-GT columns are compatibility checks, not the primary V2 claim.

## Stage 1D: Oracle gap under damage GT

Use the previously installed isolated diagnostic runner. `gt-source damage`
constructs the same damage-class CC labels from the dataset.

```bash
DIAG_RUNNER="$ROOT/rgid_diagnostics_isolated/run_diagnostics.py"

python -u "$DIAG_RUNNER" oracle-center \
  --project-root "$ROOT" \
  --cache-dir "$VAL_CACHE" \
  --checkpoint "$V2_HEAD/rgid_step_500_calibrated.pth" \
  --dataset-root "$DATA" \
  --output-dir "$V2/diagnostics/oracle_damage_gt" \
  --gt-source damage \
  --device cuda \
  2>&1 | tee "$V2/logs/oracle_damage_gt.log"
```

## Stage 2: Center loss-stage audit

Run V1/binary first as a reproducibility reference, then V1/damage and
V2/damage. The last comparison shows whether improved GT changed actual Center
localization rather than only changing the evaluation labels.

```bash
python -u "$TOOLS/center_stage_audit.py" \
  --experiment-root "$FIXED" --checkpoint "$V1_CKPT" \
  --cache-dir "$VAL_CACHE" \
  --output-dir "$V2/center_audit/v1_binary" \
  --radii 1,2,3,5 --taxonomy-radius 3 --device cuda

python -u "$TOOLS/center_stage_audit.py" \
  --experiment-root "$FIXED" --checkpoint "$V1_CKPT" \
  --cache-dir "$VAL_CACHE" --gt-sidecar "$VAL_GT" \
  --output-dir "$V2/center_audit/v1_damage" \
  --radii 1,2,3,5 --taxonomy-radius 3 --device cuda

python -u "$TOOLS/center_stage_audit.py" \
  --experiment-root "$FIXED" \
  --checkpoint "$V2_HEAD/rgid_step_500_calibrated.pth" \
  --cache-dir "$VAL_CACHE" --gt-sidecar "$VAL_GT" \
  --output-dir "$V2/center_audit/v2_damage" \
  --radii 1,2,3,5 --taxonomy-radius 3 --device cuda
```

Read `center_stage_audit.json`:

- high `filtered_by_threshold`: calibrate `peak_threshold` on training data;
- high `no_local_max_within_radius`: improve Center target/head/localization;
- high `suppressed_by_distance_nms`: only then change NMS distance;
- high `truncated_by_marker_cap`: only then raise the marker cap;
- high `parent_lost_at_head_resolution`: evaluate a higher-resolution head;
- high `duplicate_or_unassigned_near_gt`: add duplicate-peak hard negatives.

## Stage 3 gate

Do not combine all Center changes. Select exactly one next experiment from the
dominant Stage-2 category, retain the V2 GT, seed and decoder, and rerun the same
2x2/Oracle protocol. Rollback is recalibrated only after Center improves.

## Stage 4 A/B/C damage aggregation

The current RGID cache does not contain damage logits. Before this stage, export
one `.pt` per sample with:

```text
damage_logits      FloatTensor[C,H,W]
damage_gt          LongTensor[H,W]
parent_mask        BoolTensor[H,W]
rgid_instance_ids  LongTensor[H,W] from selected G3
gt_instance_ids    LongTensor[H,W] (optional but needed for building-level F1)
event              string (recommended for per-event metrics)
```

Then run:

```bash
python -u "$TOOLS/evaluate_damage_aggregation.py" \
  --input-dir /path/to/damage_aggregation_inputs \
  --output "$V2/evaluations/damage_aggregation_abc.json" \
  --classes 1,2,3
```

Method A leaves pixel argmax unchanged. Method B assigns one damage class per
binary parent. Method C assigns one class per RGID instance. Stage 4 succeeds
only if C improves both damage metrics and instance metrics; instance F1 alone
does not establish a better building-damage model.

