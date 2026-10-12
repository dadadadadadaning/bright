# P3-SRAC Controlled Experiments

## 1. Scope

This is a controlled transfer inspired by the public SRAC overview. It is not
an official SRAC reproduction. The unavailable VMambaSSN, MMA-P and metadata
injection details are deliberately excluded.

The experiment changes the prediction unit, not the P3 backbone:

```text
frozen P3 RGB/SAR encoders + frozen RGB localisation
                         |
                 multiscale P3 features
                         |
          learned RGB-guided local superpixels
                         |
        region relation mean/std/max + raw SAR stats
                         |
              primary three-class region head
                         |
        frozen P3 building probability factorisation
                         |
       Background / Intact / Damaged / Destroyed
```

SR1 does not add a residual to P3. It replaces P3's final pixel damage head.
SR2 adds a class-balanced training-only region memory. Validation and test
samples can query the frozen memory but can never update it.

## 2. Controlled chain

| Variant | Only change | Question |
|---|---|---|
| SR0 | no change | Does the wrapper reproduce P3 exactly? |
| SR1 | learned superpixels become the primary classifier | Does region-level decision outperform the pixel head? |
| SR1F | preserve P3 Background/Building probability; region head predicts only conditional I/D/X | Was SR1 harmed by changing occurrence and severity at once? |
| SR2 | class-balanced train memory and retrieval | Does instance retrieval help rare damage classes? |

The original SR1 result showed that its RGB-localisation factorisation was a
confound. Run SR1F as the single correction. Do not run SR2 unless SR1F passes
its validation gate. Do not tune either
variant against the test split.

## 3. Server commands

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

bash run/run_P3_SRAC_controlled_5090.sh preflight
bash run/run_P3_SRAC_controlled_5090.sh guard

VARIANT=sr1 bash run/run_P3_SRAC_controlled_5090.sh smoke

LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/P3_SRAC_CONTROLLED
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/P3_SR1_screen3000_seed3407.log"

nohup env \
  VARIANT=sr1 \
  OPTIMIZER_STEPS=3000 \
  EVAL_EVERY=500 \
  bash run/run_P3_SRAC_controlled_5090.sh screen \
  > "$LOG" 2>&1 &

tail -F "$LOG"
```

After training, compare the locked P3 and selected SR1 weights:

```bash
BASE=/root/shared-nvme/BRIGHT-checkpoints/P3_SRAC_CONTROLLED/P3_SR0_guard_seed3407/validation_metrics.json
CAND=/root/shared-nvme/BRIGHT-checkpoints/P3_SRAC_CONTROLLED/P3_SR1_screen3000_seed3407/best_event_worst.pth
OUT=/root/shared-nvme/BRIGHT-checkpoints/P3_SRAC_CONTROLLED/P3_SR1_advancement.json

python script/standard_ML/summarize_P3_SRAC.py \
  --baseline "$BASE" \
  --candidate "$CAND" \
  --output "$OUT"
```

SR1 advances only if all conditions pass:

- Valid mIoU improves by at least 0.30 percentage points.
- Damaged IoU improves by at least 0.50 percentage points.
- Event mean and event worst do not decline.
- At least 10 of all 14 events are non-decreasing.

Only then run SR2 with the same training protocol:

```bash
VARIANT=sr2 bash run/run_P3_SRAC_controlled_5090.sh smoke
VARIANT=sr2 OPTIMIZER_STEPS=3000 \
  bash run/run_P3_SRAC_controlled_5090.sh screen
```

## 4. Interpretation

- SR1 passes: P3's remaining bottleneck is its pixel decision unit; proceed to
  retrieval.
- SR1 fails and region purity is low: the learned partition is not tracking
  damage structures; inspect region size, purity and boundary visualisations.
- SR1 fails despite high purity: region classification removes useful pixel
  detail or the frozen P3 embedding is not retrieval-ready. Stop SR2.
- SR2 improves only rare Destroyed but hurts events: retrieval is domain
  conditioned; metadata/event calibration becomes a separate SR3 experiment.
- SR2 fails: memory lookup is not the missing mechanism; do not add more
  attention or larger memory blindly.
