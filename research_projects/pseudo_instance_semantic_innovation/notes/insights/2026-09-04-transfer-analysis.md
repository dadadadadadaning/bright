# Transfer analysis and experiment ordering

## Core transferable mechanism

The evidence does not support another hard instance classifier. It supports a
selective object-context operator:

1. LocCC supplies predicted building support.
2. A6 purity and probability margin reject ambiguous regions.
3. A foreground-only soft region distribution supplies context.
4. normalized pixel entropy selects points that need correction.
5. erosion depth reduces changes at unreliable boundaries.
6. a matched local-window variant tests whether gains require building context.

This combines only the transferable principles of PointRend, SegFix and OCR;
it does not copy their architectures or claim their original contributions.

## Experiment ladder

1. Finish RGID C3 independently: RGID+A6 versus LocCC+A6.
2. Run selective-context preflight and two-image smoke.
3. Run one full frozen-P3 Val450 audit.
4. If S3 fails, stop this branch with one GPU pass spent.
5. If S3 passes, implement sparse region-prototype contrast (ReCo-inspired)
   with a frozen-P3 KL anchor and compare against CE and Lovasz baselines.
6. Only after single-seed success, run three seeds and event-disjoint folds.
7. Access the held-out test once after method and thresholds are frozen.

## Reviewer-style idea verdict

- Higher: medium-high; directly targets the observed hard-broadcast failure.
- Faster: high; first gate is inference-only and cache-free.
- Stronger: medium; causal local control and event bootstrap are built in.
- Cheaper: high; one P3 pass, no 208 GiB cache, no new backbone.
- Broader: medium-high; applies to semantic segmentation with noisy object
  regions beyond disaster imagery.
- Fatal risk: the gain may still come from generic smoothing. S4 is therefore
  mandatory; without a positive S3-S4 confidence interval, reject the claim.

