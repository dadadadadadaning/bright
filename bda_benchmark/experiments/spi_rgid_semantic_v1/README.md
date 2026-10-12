# SPI-RGID Semantic V1 (isolated orchestration)

This directory is the single entry point for the current paper main line:

> Can pseudo building geometry, learned without official instance annotations,
> improve the locked P3 semantic prediction beyond the LocCC+A6 control?

It does not define another damage classifier and does not modify P3, MICB,
CMR, RGID S9-Ic, SPI T1, or their historical outputs. It only validates the
experiment contract and dispatches the already isolated C0--C3 causal audit.

## Frozen causal chain

| Variant | Geometry | Semantic decision | Purpose |
|---|---|---|---|
| C0 | none | raw P3 | 60.5020% reference |
| C1 | matched 5x5 windows | locked A6 | spatial smoothing control |
| C2 | P3 localization CC | locked A6 | building-structure control |
| C3 | pseudo-only learned RGID G3 | locked A6 | learned instance-geometry test |

The only admissible C2--C3 change is the partition. Coverage, P3 probabilities,
majority rule, purity threshold, margin threshold, and evaluation split remain
identical.

## Usage on the Linux server

```bash
cd /root/data/BRIGHT-master/bda_benchmark

bash experiments/spi_rgid_semantic_v1/run.sh preflight
bash experiments/spi_rgid_semantic_v1/run.sh controls

# Cache-free: frozen P3 is recomputed online; only the small RGID head is trained.
RGID_STEPS=500 NUM_WORKERS=4 \
bash experiments/spi_rgid_semantic_v1/run.sh train-online

RGID_CKPT=/root/data/BRIGHT-checkpoints/RGID_PSEUDO_ONLY/seed3407/rgid_step_500_calibrated.pth \
RGID_PROVENANCE_JSON=/root/data/BRIGHT-checkpoints/RGID_PSEUDO_ONLY/seed3407/provenance.json \
bash experiments/spi_rgid_semantic_v1/run.sh rgid

python -m bda_benchmark.experiments.spi_rgid_semantic_v1.scripts.summarize_c3 \
  /root/data/BRIGHT-eval-results/SPI_GEOMETRY_CAUSAL_VAL450/SPI_GEOMETRY_CAUSAL_validation_full_with_rgid_g3_v1_seed3407.json
```

The online trainer writes `rgid_step_500_calibrated.pth` and
`provenance.json` under `RGID_PSEUDO_ONLY/seed3407`. Use those exact two files
for `rgid` mode.

The optimized online path preserves the legacy Center definition but computes
each distance transform inside the instance bounding box. Binary connected
components are produced in DataLoader workers, and every optimizer log row is
the aggregate of all gradient-accumulation micro-batches. `target_seconds` and
`step_seconds` expose the remaining CPU share directly.

`controls` re-runs C0--C2 only. `rgid` refuses to run unless the checkpoint
provenance states that official instances, Standard-ML Val450, and the original
test split were never used for training or calibration.

## What is deliberately excluded

- MICB or Memory voting;
- official-instance training;
- test-time pseudo-label retraining;
- event-specific thresholds selected on Val450;
- tuning A6 or the 5x5 control;
- a new damage head before C3 proves a positive geometry effect.

This boundary prevents a positive result from being attributed to several
simultaneously changed mechanisms.

## Staged post-C3 modules

The following modules are implemented and tested, but they are deliberately not
enabled by `rgid` mode. They become admissible only after C3 passes the locked
geometry gate:

- `method.py`: prediction-only split-risk scoring and split-on-demand. A LocCC
  parent is replaced by RGID children only when it has at least two stable
  centers, at least two decoded children, and a train-calibrated risk score.
- `heads.py`: the R0 low-resolution, R1 high-resolution and R2 dual-scale Center
  ablations. All modes decode on the same high-resolution grid.
- `objectives.py`: quality-weighted soft region supervision, geometric
  equivariance and frozen-P3 KL anchoring. It never hard-broadcasts a pseudo
  label to every pixel during training.
- `event_protocol.py`: event-disjoint folds and a cross-event acceptance gate.

Create event-disjoint folds from the old training list with:

```bash
python -m bda_benchmark.experiments.spi_rgid_semantic_v1.scripts.build_event_folds \
  --input-list dataset/splitname/standard_ML/train_set.txt \
  --output-dir /root/data/BRIGHT-splits/SPI_RGID_EVENT_FOLDS \
  --fold-count 4
```

The generated manifest records the source hash and exact event membership. It
does not read images, semantic targets or official instances.
