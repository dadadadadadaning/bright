# Predicted building/background split: controlled training

This isolated extension imports the existing `damageformer_srac_approx_v1`
trainer and modules. Upload this directory alongside the original directory.
It does not edit old files or overwrite existing output directories.

## Experiments

* `s3_original`: optional exact original S3 implementation control.
* `s3_predsplit`: S3 with each seed split into predicted background/building.
* `grid_predsplit`: same modulation, weighted pooling, statistics and classifier,
  but fixed grid assignments, with the same predicted split.

Threshold is fixed at 0.5 on detached A1 `p_building` at feature resolution.
Main features AND mean/std are pooled using child ID `2*seed+group`.
Each pixel reads only its own group during projection. All four output classes
remain allowed. This is not instance separation and does not guarantee connectivity.
Modulation edges remain the original unsplit edges. The original unsplit RGB,
coordinate and building reconstruction auxiliary loss is preserved for learned
assignments; the fixed grid has no partition auxiliary loss, as before.
Thus the grid comparison includes the documented difference in partition learning.

No GT is used to construct groups; GT remains ordinary training supervision.
No Memory, no new gating, no backbone training, no Test input. The dataset's
`type='test'` is the original deterministic loader mode for the VAL list, not the
Test split. Do not substitute a renamed Test list for the validation list.

All heads start fresh with identical parameter initialization at a given seed.
Do not warm-start just one variant from old S3. Protocol stays batch 1, crop512,
stride8, AdamW 1e-4 / weight decay5e-3, cosine decay, clip1, CE plus 0.1 partition
loss for learned assignments, 5000 steps / validation every500 by default.
Only best.pth and last.pth are saved, not per-step weights. The inherited manifest
records arguments and parent provenance; read this file for the split definition.

## Run on server

Set A1_CKPT to the verified A1 file and OUTPUT_ROOT to a new data-disk directory.
`bash bda_benchmark/experiments/damageformer_srac_predsplit_v1/run.sh preflight`
then `.../run.sh sequence` runs the two NEW variants serially. For a 20-step
smoke run use STEPS=20 EVAL_EVERY=20 and a separate OUTPUT_ROOT. Evaluation still
uses the full Val list; 20 steps do not establish effectiveness.

Compare with A1 foreground mIoU 0.6168659209491562 and original S3 0.575377,
not probability diagnostic 0.614587. Each saved checkpoint contains the full
4-class confusion and reproduced A1 score. Inspect A1 reproduction before
interpreting changes. Require consistent Val protocol and per-class outcomes;
repeat seeds if promising. This experiment does not yet export event or boundary
breakdowns; those require a matched evaluation extension, not region-only metrics.
Previous spatial audit Exit 1 must be diagnosed from its log; an existing JSON
alone does not prove the latest invocation completed successfully.
