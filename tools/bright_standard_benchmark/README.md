# BRIGHT six-model unified benchmark

This directory evaluates the requested checkpoints under one protocol:

- U-Net
- SiamCRNN
- SiamAttnUNet
- DeepLabV3+
- DamageFormer
- ChangeMamba

## Run on the Linux server that stores the checkpoints

```bash
cd /root/shared-nvme/BRIGHT-master
bash tools/bright_standard_benchmark/run_all.sh
```

The defaults match the paths supplied for this task. Paths can be overridden:

```bash
DATA_ROOT=/path/to/dataset \
LIST_FILE=/path/to/test_set.txt \
CHECKPOINT_ROOT=/path/to/checkpoints \
OUTPUT_ROOT=/path/to/results \
PYTHON_BIN=/path/to/python \
bash tools/bright_standard_benchmark/run_all.sh
```

Use `TTA=1` only when all six models should be evaluated with the same
identity/horizontal-flip/vertical-flip ensemble. The default is no TTA. Use
`AMP=1` only when mixed-precision inference is intentionally part of the fixed
protocol. Completed model directories are skipped so that an interrupted run
can be resumed; set `FORCE=1` to recompute them.

## Output

Each model directory contains:

- `metrics.json`: checkpoint hash, protocol, global metrics and timing;
- `per_class_metrics.csv`;
- `per_event_metrics.csv`;
- `per_disaster_type_metrics.csv`;
- `per_image_metrics.csv`;
- `confusion_matrix.csv`;
- `predictions/raw/*.png` and `predictions/colored/*.png`.

`report/` contains:

- `model_comparison.csv`;
- overall and per-class metric figures in PNG/PDF;
- an event-level heatmap;
- normalized confusion matrices;
- `qualitative_comparison.png/pdf`, using exactly the same cases for every model;
- individual qualitative case panels;
- `report.md` and `selected_samples.json`.

The qualitative panel uses real pre-event RGB, post-event SAR, ground truth and
saved predictions. SAR contrast stretching is display-only and does not affect
evaluation.

## Protocol notes

- Four labels are fixed as Background, Intact, Damaged and Destroyed.
- Damage predictions are evaluated directly and are not multiplied by the
  localization head; this matches the original standard inference scripts.
- Both 4-class mIoU and foreground 3-class mIoU are reported.
- `Damaged -> Intact` is reported explicitly because it is the main observed
  error mode.
- Checkpoint loading is strict by default. A mismatched checkpoint stops the
  run instead of silently producing invalid scores.
- The ChangeMamba checkpoint is assumed to be MambaBDA-Tiny and uses
  `vssm_tiny_224_0229flex.yaml`, consistent with the repository's released
  BRIGHT checkpoint description.
