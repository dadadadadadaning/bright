# OA-HMD M0: P3 Intact/Affected Boundary Audit

## Question

Can P3's missed `Damaged` pixels be recovered by one global shift of the
`Intact` versus `Affected = Damaged + Destroyed` boundary, without learning a
new representation?

## Locked contract

- frozen P3 checkpoint and historical P3 RGB/SAR input;
- validation split only;
- one P3 forward pass per image;
- offsets evaluated only inside the frozen hard P3 building prediction;
- the same beta is added to Damaged and Destroyed logits, preserving their
  conditional odds exactly;
- ground truth is used only to compute metrics;
- no training and no test access.

The declared sweep is `[-0.50, -0.40, ..., 0.40, 0.50]`, including an exact
`beta=0` P3 baseline.  The full validation run checks that this baseline
reproduces the existing OA-HMD C0 independent validation metrics.

## Advancement gate

A non-zero global beta must simultaneously achieve, relative to beta zero:

- Damaged IoU >= +0.30 pp;
- Affected Recall >= +0.50 pp;
- Damaged gate-miss <= -0.50 pp;
- Valid mIoU, Event mean, and Event worst do not decrease;
- at least 10 of 14 events are non-negative in mIoU4.

Decisions:

- `PROCEED_M1_GLOBAL_ORDINAL_CALIBRATION`: a global boundary is viable;
- `PROCEED_EVENT_CONDITIONED_CALIBRATION_AUDIT`: aggregate calibration fails,
  while multiple events benefit and their preferred beta directions conflict;
- `STOP_GLOBAL_CALIBRATION_REPRESENTATION_REQUIRED`: a boundary shift cannot
  explain the missing Damaged performance.

## Outputs

- JSON: complete protocol, metrics, gates, per-event optima, and decision;
- `.betas.tsv`: one row per global beta;
- `.events.tsv`: best beta and gains for each event.

M0 never produces a deployable checkpoint and must not be reported as a test
method.  It only selects the next validation-only research branch.
