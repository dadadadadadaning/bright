# Full region readout: content-source control

This isolated diagnostic does not change old experiment files. It requires the
existing fourstage V2 implementation and a matching fixed FRONT snapshot.
Do not use A1 weights, Memory weights, or a six-head checkpoint as FRONT_CKPT.

## Question

Can the **original full region readout architecture** perform better using
original fused/Soft-ROI features than using image-modulated features?

One frozen eval-mode front pass captures raw (first fusion), normal (second
fusion), roi (exact original pixel-classifier input), and one shared Q.
Three identically initialized, independent fresh readouts use the original
RegionEmbedding: attention pooling + mean/std bypass + normalization + four-class
classifier + Q probability reprojection. No new input LayerNorm is added.
Internal original LayerNorms remain: this does NOT guarantee all ROI amplitude
information survives. This is not an isolated test of Soft ROI training benefit.

Only each fresh embedding/classifier learns. Swin, fusion, localization, original
pixel/region heads, Q decoder and BN statistics stay frozen. Front is still run
twice internally to obtain the normal control; roi/raw readouts do not consume
the modulated feature. No memory, hard building split, GT partition, new loss,
pixel-residual fusion or joint front fine-tuning is introduced.

## Run on server after uploading this whole directory

Use the bright4090 environment and repository root.

```bash
bash bda_benchmark/experiments/damageformer_region_content_v1/run.sh preflight
export FRONT_CKPT=/path/to/existing/front_snapshot.pth
export OUTPUT_ROOT=/path/to/new/smoke_directory
bash bda_benchmark/experiments/damageformer_region_content_v1/run.sh smoke
export OUTPUT_ROOT=/path/to/new/formal_directory
bash bda_benchmark/experiments/damageformer_region_content_v1/run.sh train
```

The paths above are placeholders. Default: BATCH=2 ACCUM=8 EBS16, 5000 optimizer
updates (80000 crops **per head**, same batches), AdamW lr1e-4 wd.005,
100-update warmup + cosine decay, region NLL+Lovasz, full Val450 every500.
Smoke:20 updates, first2 validation images; not a performance estimate.
Front source hash/checkpoint checks are retained; do not bypass a mismatch.
Initial full Val must reproduce original region checkpoint score within1e-5.
Original pixel-input replay and full-region replay are checked on validation.

Outputs: manifest, initial/validation metrics including class confusion,
training losses, summary, and 3 best + 3 last **readout-only** checkpoints.
No full front copies, optimizer checkpoints, feature caches, or Test use.
No resume support. New output directories must not exist.
Best/last heads need the exact front snapshot and the source name to evaluate.
manifest records front/hash/dim/source. Store the snapshot safely, do not delete it.

## Interpret

- roi/raw > normal: original feature source helps this full readout under a
  frozen checkpoint; not proof about end-to-end retraining or all modulation.
- roi vs raw: ROI content-use comparison conditional on already trained features.
- old_pixel/old_region: fixed references, not matched newly trained classifiers.
- Compare at the same step, then per-head Val-best; examine Damaged IoU/F1 and
  Damaged->Intact before declaring recovery. Per-pixel harm/rescue requires a
  further paired-prediction evaluation, not just aggregate confusion matrices.
- Remaining failures may involve optimization, region embedding, mixed regions,
  or feature learning; do not attribute all to Q or insufficient steps.

CPU contracts are provided; actual server GPU/data smoke is still required.
