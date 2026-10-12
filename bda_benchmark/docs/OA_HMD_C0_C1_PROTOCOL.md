# OA-HMD C0/C1 controlled protocol

## Research question

Can a modality-preserving Stage3 RGB/SAR relation improve P3's
Intact-vs-Affected decision without changing P3 localisation or the
Damaged-vs-Destroyed boundary?

This protocol does **not** claim that the full OA-HMD pyramid is already
validated. It implements only the first causal test needed before Stage2,
Stage4 severity, ordinal supervision, or intact-only FCA are considered.

## Active paths

```text
pre RGB + historical P3 SAR view
        -> locked P3
        -> localisation logits + four-class base logits

pre RGB -> locked H6-A2 RGB encoder -> raw R3
post native 1-channel SAR -> locked H6-A2 SAR encoder -> raw S3

R3 -> RGB-only 1x1 projection ----+
                                      -> [R,S,|R-S|,R*S]
S3 -> SAR-only 1x1 projection ----+          -> 2-D mixer
                                               -> affected candidate

final logits = P3 logits + tanh(alpha) * affected candidate
```

The residual is added equally to Damaged and Destroyed logits. Therefore it
changes only Intact-vs-Affected and exactly preserves
`Destroyed logit - Damaged logit`.

## Input contract

The H4 heterogeneous loader returns two SAR views:

1. train-statistics-normalised native one-channel SAR for H6-A2;
2. un-clipped display-domain SAR for exact reconstruction of P3's historical
   repeated-three-channel input.

The two views are never mixed before their declared encoders.

## Variants

- `c0`: exact frozen P3 guard; no trainable relation branch.
- `c1`: frozen P3 and H6-A2 encoders plus a trainable Stage3 explicit
  difference head and scalar transition.

## Explicit exclusions

C1 has no FCA, optical flow, Conv3D, pseudo-temporal stacking, pixel router,
prototype memory update, superpixel smoothing, region graph, or ordinal loss.
Inherited H6 modules not on the raw encoder Stage3 path are not executed.

## Validation gate

Relative to the independently evaluated C0 baseline, C1 must satisfy all:

- Valid mIoU: at least +0.30 percentage points;
- Damaged IoU: at least +0.50 points;
- Affected Recall: at least +0.50 points;
- Damaged gate-miss: at least -0.50 points;
- Event mean and event worst: non-decreasing;
- non-negative events: at least 10/14.

Failure means stopping C1. Stage2 must not be added merely because C1 trained
successfully.

## Server commands

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
source /base/mambaforge/etc/profile.d/conda.sh
conda activate bright5090

bash run/run_OA_HMD_5090.sh preflight

LOG_ROOT=/root/shared-nvme/BRIGHT-train-logs/OA_HMD
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/OA_HMD_C0_C1_$(date +%Y%m%d_%H%M%S).log"

nohup env \
  OPTIMIZER_STEPS=3000 \
  EVAL_EVERY=500 \
  SAVE_EVERY=1000 \
  bash run/run_OA_HMD_sequence.sh screen \
  > "$LOG" 2>&1 &

echo $!
tail -F "$LOG"
```

The sequence performs preflight, the exact C0 guard, C1 Smoke20, C1 3000-step
training, and one strict independent validation. Test is not accessed by
default. Set `EVAL_TEST_ON_PASS=1` only if one diagnostic test is explicitly
desired after every validation gate passes.
