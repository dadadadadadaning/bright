# Stage A: differentiable local partition, isolated V1

This implements the FIRST stage of the proposed plan, NOT full SRAC/SPFormer/SSN,
not a new backbone, and not the later Memory/contrastive/attention readout stage.
No existing experiment files or weights are edited. It imports the existing base
model, dataset, losses and evaluation; upload this whole folder into the existing
BRIGHT repository, not as a standalone project. It does not launch remote jobs.

## Variants

| Name | Partition | Interpretation |
|---|---|---|
| pixel | none | Same architecture as T0; optional matched rerun |
| single | original convolution -> 9-way logits | Exact old T3 head control |
| iter1 | RGB+XY embedding -> local center similarity once | Separates similarity rule from refinement |
| iter3 | identical embedding -> 3 assignment/update rounds | Primary new candidate |
| grid | fixed home cell | Same residual readout, no learned partition |

single and iter1/iter3 use equally sized association networks (9 output channels).
For iter1/iter3 those channels are an embedding, NOT nine logits. Their complete
trainable parameter counts match. Grid has fewer active parameters; do not claim
it is total-parameter matched. Seed stride and maximum split-group budget match,
but actual occupied groups need not match; diagnostics report both.

RGB fused multi-scale features + coordinates determine assignment; fused_roi
provides pooled damage features. The existing base ALREADY fuses four Swin stages.
Assignment uses cosine similarity minus spatial squared distance in seed-stride
units, divided by temperature. Initial centers are means within home seed and
predicted building/background group. Iterations have full gradients. Empty
centers are masked; FP32 sums/normalization are used under BF16 forward.
Hard building grouping is detached and uses prediction >=0.5, NOT GT. It separates
foreground/background, NOT individual buildings. Connectivity is not guaranteed.
Local nine seed candidates are unrelated to RGB-SAR alignment search windows.

The residual readout stays [pixel,context,pixel] with the old dimensions. Its last
layer is zero-initialized, so initial output is the base pixel prediction; this
does not guarantee later safety. Association gradients can be zero on the first
step until this last layer moves. No new partition/boundary/contrastive losses,
no Memory, no confidence thresholds for selecting evaluation pixels.

## Locked protocol

- Same torchvision Swin-T features weights for every variant; fresh task heads.
- Formal: 50,000 optimizer steps, batch4 x accum4 = EBS16, 800,000 crops512.
- AdamW 1e-4, weight decay .005, warmup1000 + cosine, grad clip1.
- Existing localization CE+Lovasz and final damage CE+Lovasz, same as prior V2.
- BF16 forward, FP32 losses and Val; BatchNorm still sees MICRO-batches.
- Train2918 / Val450, duplicates and overlap rejected; no Test.
- Val every1000; only best.pth and final last.pth. No resume, no overwrites.
- Smoke: 20 optimizer steps and first 2 Val images, not a metric comparison.
- Model/data/pretrain hashes recorded. Logs include entropy, hard/soft occupied
  groups, building fraction, association gradient norm (after clipping, last batch).
- Actual training/validation results require the user's Linux GPU server.

Defaults temperature=.2, spatial_weight=.1, stride=8 are engineering starting
points, not literature-optimal values. Changing them creates a new experiment.

## Server commands (from /root/data/BRIGHT-master)

Upload this entire folder first. Dependencies must be present in the same repo.

```bash
conda activate bright4090
bash bda_benchmark/experiments/damageformer_iterative_partition_v1/run.sh preflight
```

Set SWIN_PRETRAIN to the ACTUAL original torchvision weight file; do not use
A1/B2/best.pth task checkpoints. Recover the recorded old path if still valid:

```bash
export SWIN_PRETRAIN="$(python - <<'PY'
import json
from pathlib import Path
p=Path('/root/data/BRIGHT-checkpoints/DAMAGEFORMER_JOINT_REGION_IMAGE_V2/formal50000_seed3407/t0/manifest.json')
data=json.loads(p.read_text())
w=Path(data['arguments']['pretrain'])
if not w.is_file():
    raise SystemExit(f'Original pretrained file missing: {w}; locate the same file before training.')
import hashlib
actual=hashlib.sha256(w.read_bytes()).hexdigest()
if actual != data['pretrain_sha256']:
    raise SystemExit('Pretrained file hash differs from T0; do not mix initialization.')
print(w)
PY
)"
ls -lh "$SWIN_PRETRAIN"
```

If this fails, stop and locate the real file. Setting PYTHONPATH never creates
missing files. Keep the existing environment; no new CUDA extension is required.

```bash
export OUTPUT_ROOT=/root/data/BRIGHT-checkpoints/DAMAGEFORMER_ITERATIVE_PARTITION_V1/smoke_seed3407
bash bda_benchmark/experiments/damageformer_iterative_partition_v1/run.sh smoke_sequence
```

Only after ALL smoke variants pass:

```bash
mkdir -p /root/data/BRIGHT-train-logs/DAMAGEFORMER_ITERATIVE_PARTITION_V1
export OUTPUT_ROOT=/root/data/BRIGHT-checkpoints/DAMAGEFORMER_ITERATIVE_PARTITION_V1/formal50000_seed3407
nohup bash bda_benchmark/experiments/damageformer_iterative_partition_v1/run.sh sequence \
  > /root/data/BRIGHT-train-logs/DAMAGEFORMER_ITERATIVE_PARTITION_V1/sequence.log 2>&1 &
echo "PID=$!"
```

Default sequence: single -> iter1 -> iter3 -> grid, fail-fast, never concurrent.
To include a fresh T0: export VARIANTS="pixel single iter1 iter3 grid" BEFORE start.
Do not interrupt older jobs or delete previous outputs automatically. If output
exists select a fresh root. 50k training is a user decision, not run by preflight.

```bash
python -m bda_benchmark.experiments.damageformer_iterative_partition_v1.status --root "$OUTPUT_ROOT"
```

## Interpretation

iter3 vs iter1 isolates refinement rounds. iter3 vs single compares partition
mechanisms. iter3 vs grid tests adaptive partition at the same MAX seed budget,
not necessarily the same effective region count. Compare all to same-protocol
pixel baseline, not old differently trained A1. Repeat seeds before strong claims.
Later cross-attention readout, multi-prototype Train-only Memory and contrastive
learning must be isolated additional stages, not implied to be implemented here.

Next audits: full-image foreground mIoU/Damaged F1, class x boundary/interior
rescue/harm, event differences and group-size denominators. Current diagnostics
are NOT connectivity, boundary recall or proof of local damage boundaries.
Do not reuse the OLD branch-audit constructor on iter3 checkpoints without adapting
it; its model architecture/assignment mechanism is different.
