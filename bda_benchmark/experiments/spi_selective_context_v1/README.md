# SPI Selective Context V1

This isolated experiment tests an innovation **other than building splitting**:
soft, uncertainty-selective object context for the locked P3 semantic output.
It does not modify P3, RGID, MICB, T1, or historical outputs.

## Causal variants

| Variant | Context | Update rule | Question |
|---|---|---|---|
| S0 | none | raw P3 | locked 60.5020% baseline |
| S1 | LocCC | hard A6 majority | reproduce the strongest existing control |
| S2 | LocCC | soft distribution blend | is soft context safer than hard broadcast? |
| S3 | LocCC | uncertain interior only | do uncertainty and boundary protection improve S1? |
| S4 | matched 5x5 window | same selective blend | is any S3 gain genuinely structural? |

The official instance annotations and original test split are never opened.

## Locked Test878 evaluation

After P3 and fixed S2 have been frozen, the one-shot Test evaluator compares only
those two systems.  It does not evaluate S1, S3, or S4 and refuses to overwrite an
existing result:

```bash
ALLOW_TEST_ACCESS=YES \
bash experiments/spi_selective_context_v1/run.sh test_locked
```

The default result is written to:

```text
/root/data/BRIGHT-eval-results/SPI_SELECTIVE_CONTEXT_TEST878_LOCKED_V1/
P3_FIXED_S2_TEST878_LOCKED_seed3407.json
```

Do not tune fixed-S2 thresholds, minimum area, or the selected checkpoint after
reading this Test result.
Ground-truth semantic TIFFs are loaded only after all predictions for an image
have been fixed.

## Why this follows the existing evidence

- T1 equal-component weighting hurt raw mIoU, so no new training is started.
- MICB hard broadcast hurt mIoU, so a region label is not copied blindly.
- LocCC+A6 improved P3, so its region geometry and gate are retained.
- Boundary hotspots were unreliable, so the boundary receives a lower update.
- RGID C3 remains a separate geometric experiment and is not mixed into this run.

## Server commands (RTX 4090)

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash experiments/spi_selective_context_v1/run.sh preflight
SMOKE_IMAGES=2 NUM_WORKERS=4 \
  bash experiments/spi_selective_context_v1/run.sh smoke
NUM_WORKERS=4 \
  bash experiments/spi_selective_context_v1/run.sh audit 2>&1 | \
  tee /root/data/BRIGHT-train-logs/SPI_SELECTIVE_CONTEXT_seed3407.log
```

This is one frozen-model pass over Val450, uses no feature cache, and should be
run only after the smoke result reproduces P3 on its subset without exceptions.

## Decision rule

S3 is allowed to motivate a trainable module only if all checks pass:

1. mIoU gain over S0 is at least 0.5 pp;
2. Damaged F1 gain is at least 1.0 pp;
3. no foreground-class IoU falls by more than 0.5 pp;
4. at least the scaled equivalent of 7/11 events are nonnegative;
5. S3-S4 event-stratified bootstrap lower bound is positive.

If S3 does not pass, keep S1 as the deployable region-feedback baseline and do
not train a PointRend/OCR-style head. Finish the independent C3 RGID comparison.
