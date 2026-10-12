# OA-HSDNet SAR Event-Style controlled experiments

## Contract

All E1-E3 candidates start from the same `N1-shared Heads500` checkpoint.
They use the same loss, learning rates, Deep phase, seed and validation-only
selector.  SAR Stage2 backbone weights stay frozen; Joint and Test are locked.

| Run | Sampling | SAR Stage1/2 transform | Trainable addition |
|---|---|---|---|
| E1 | event-balanced | none | none |
| E2 | event-balanced | cross-event MixStyle | none |
| E3 | event-balanced | event style/content adapter | zero-gated adapter |

E2 is disabled automatically during validation.  E3 is inserted after SAR
Stage1/2 and before downsampling, so the transformed representation is consumed
by deeper SAR stages.  Its transition is initialized to zero, making the first
forward exactly equal to the source N1-shared model while retaining a non-zero
first-step transition gradient.

## Server commands

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

bash run/run_OABH_sar_event_style_sequence.sh preflight
```

Verify or override the shared starting checkpoint:

```bash
export BASE_HEADS=/root/shared-nvme/BRIGHT-checkpoints/OA_HSDNET/OA_HSDNET_N1_SHARED_heads500_seed3407/best_event_worst.pth
test -s "$BASE_HEADS" && echo READY
```

Run candidates separately (recommended):

```bash
bash run/run_OABH_sar_event_style_sequence.sh e1
bash run/run_OABH_sar_event_style_sequence.sh e2
bash run/run_OABH_sar_event_style_sequence.sh e3
```

Or run the complete validation-only chain:

```bash
LOG=/root/shared-nvme/BRIGHT-train-logs/OA_HSDNET/OABH_E1_E3_controlled.log
nohup bash run/run_OABH_sar_event_style_sequence.sh all > "$LOG" 2>&1 &
tail -F "$LOG"
```

Do not run Test unless a candidate passes the locked validation gate.

