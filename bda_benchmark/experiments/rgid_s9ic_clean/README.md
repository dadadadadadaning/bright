# RGID S9-Ic Clean Experiment

This directory is an isolated implementation of Reliable Geometry-guided
Instance Decomposition for the frozen S8 `fused_roi` feature. It does not
modify or import the legacy S9 evaluation helpers.

Core protocol:

- one deterministic Gaussian center per non-border pseudo-instance;
- CenterNet focal loss with FP32 internal computation;
- FP32 boundary BCE + Dice;
- G1: native center NMS + watershed;
- G2: G1 + boundary gate;
- G3: G2 + geometry rollback.

Install and test:

```bash
cd experiments/rgid_s9ic_clean
bash run_all_tests.sh
```

## Cached multi-batch protocol

Export train/validation features independently:

```bash
python scripts/export_bright_rgid_cache.py \
  --checkpoint /path/to/best_val_miou.pth \
  --data-list /path/to/train_set.txt \
  --dataset-path /path/to/dataset \
  --output-dir /path/to/rgid_train_cache \
  --device cuda:0
```

Train, calibrate only on the recorded train cache, then evaluate:

```bash
python scripts/train_cached_rgid.py \
  --train-cache /path/to/rgid_train_cache \
  --output-dir /path/to/rgid_run \
  --device cuda:0

python scripts/calibrate_cached_rgid.py \
  --checkpoint /path/to/rgid_run/rgid_step_500.pth \
  --train-cache /path/to/rgid_train_cache \
  --output /path/to/rgid_run/rgid_step_500_calibrated.pth \
  --device cuda:0

python scripts/eval_cached_rgid.py \
  --checkpoint /path/to/rgid_run/rgid_step_500_calibrated.pth \
  --val-cache /path/to/rgid_val_cache \
  --output /path/to/rgid_run/val_g0123.json \
  --device cuda:0

python scripts/compare_rgid_results.py \
  --results /path/to/rgid_run/val_g0123.json \
  --output-csv /path/to/rgid_run/val_g0123.csv
```

The original BRIGHT/S9 evaluation files are not modified.
