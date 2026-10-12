# Joint Region V1 — NEW training protocol, not a strict S3 continuation

Train T0-T4 from the SAME torchvision Swin-T encoder pretraining file. No A1,
S3 or B2 task checkpoint is loaded. Both RGB and SAR encoders load identical
feature weights; localization, fusion, damage and region heads initialize fresh.
Only feature keys are loaded strictly; ImageNet norm/head keys are discarded.
The initializer checks format, not dataset provenance: user must supply genuine
general-purpose pretraining, not renamed task weights. SHA256 is recorded.

T0: Soft ROI A1 architecture, all active layers trainable.
T1: pixel adapter [z,z,z].
T2: [z,building-split grid pooled context,z].
T3: [z,building-split learned pooled context,z].
T4: [z,building-split learned pooled context,z-context].
T1-T4 use equal-width readouts. T3/T4 have extra active assignment parameters;
T1 is a matched READOUT control, not an exact total-active-parameter control.
Report manifest trainable parameter counts. T4 supplies a linear reparameterization
of existing inputs, NOT new information; any benefit is an optimization hypothesis.

Important simplification versus historical S3: projected mean pooling, no weighted
attention/statistics skip/modulation and no partition auxiliary loss. All variants
use identical localization CE+Lovasz and final damage CE+Lovasz. Soft assignments
are learned from RGB features + coordinates through final damage supervision;
no connectivity guarantee or explicit cross-modal assignment input is claimed.
Do not compare 5000-step S3 frozen-head results as matched-training ablations.

Original-resolution pixel logits are never downsampled; only residual upsampled.
Zero output layer preserves pixel logits initially, but does not guarantee safety
after training. First backward gives no region gradient through the zero layer;
this is expected. Building threshold .5 is detached; localization loss trains its
branch. No GT partition or routing, no Memory, no Test. type=test is deterministic
loader behavior applied to val_set.txt only. Never rename Test as Val.

Defaults: crop512, microbatch1, accumulation16 (EBS16), 40000 actual optimizer
steps, AdamW1e-4 WD.005, warmup min(1000,steps//10), cosine decay, clip1,
BF16 forward / FP32 losses and validation. Existing BatchNorm uses microbatch
statistics: EBS16 accumulation is NOT BatchNorm batch16. Keep microbatch fixed
across comparisons. Strict image counts 2918/450, no overlap/duplicates.
Only best.pth and last.pth weights (atomic replacement); manifest and JSONL metrics
also saved. No resume CLI yet; do not claim resumability. New output required.

Run preflight first, then STEPS=20 EVAL_EVERY=20 with a new smoke output root.
Smoke still validates all450 images; CUDA memory and throughput must be tested
on server. No full backbone/GPU training validated by local lightweight tests.
Sequence runs T0,T1,T3 serially. Train T2/T4 separately if warranted.
All full runs must use identical settings and encoder weight SHA. Compare fresh
T0, not historical61.6866%. Do not infer success or novelty until measured.

Implementation optimizations: batched indexed scatter/gather rather than dense
pixel-by-region matrix, FP32 aggregation, dim64 context, no unused reconstruction,
BF16 and accumulation, frozen inherited unused memory/bag modules. No per-step
checkpoints, no GPU launch performed by code generation.
