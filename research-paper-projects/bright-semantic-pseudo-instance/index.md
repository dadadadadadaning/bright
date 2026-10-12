# BRIGHT Semantic-to-Pseudo-Instance Research Project

## Research question

Without using official instance masks or instance IDs to train the main method, can pseudo building structure derived from semantic damage labels provide useful instance-aware supervision that improves the **raw semantic damage prediction**, while also producing usable building instances?

## Scope boundary

- Main branch training uses semantic damage labels only.
- Official true instances are used as an evaluation ruler, a fully supervised upper bound, and annotation-budget baselines.
- The primary outcome is the raw semantic head before instance reconstruction or post-processing.
- RGID is the geometric decoder and diagnostic line; Memory is paused and is not part of the main hypothesis.
- Event/disaster metadata may define evaluation strata, but is not a damage label.

## Current verdict

**Accept with revisions.** Region-level semantic refinement is now demonstrated: locked LocCC+A6 reaches 61.5399% versus the reproduced P3 60.5020%, and it significantly exceeds a matched 5x5 local control. Joint component-balanced training failed. The remaining decisive experiment is C3: pseudo-only learned RGID versus LocCC under the identical locked A6 rule.

## Project files

- Search list: [20260901-bright-semantic-pseudo-instance.md](searches/20260901-bright-semantic-pseudo-instance.md)
- Evidence records: [20260901-bright-semantic-pseudo-instance.jsonl](searches/20260901-bright-semantic-pseudo-instance.jsonl)
- Main insight and experiment plan: [20260901-bright-semantic-pseudo-instance.md](notes/insights/20260901-bright-semantic-pseudo-instance.md)
- 2026-09-03 literature/code update: [20260903-bright-instance-innovation-update.md](searches/20260903-bright-instance-innovation-update.md)
- Current status and ranked innovations: [20260903-current-status-and-next-innovations.md](notes/insights/20260903-current-status-and-next-innovations.md)
- Reading log: [reading-log.md](reading-log.md)

## Existing local evidence used

- RGID: learned G3 F1 0.599920, oracle G3 F1 0.656250, oracle G2 F1 0.663134.
- H7: P3 validation mIoU 0.605019, building-oracle mIoU 0.663547.
- CMR-V1/V2/C0: candidate recall exists, but no safe locked operating point; local D-I hotspots are not reliable damage localization.
- Standard-ML Val450: LocCC+A6 61.5399%, P3 60.5020%, matched local-window control about 60.4987%.
- T1 component-balanced joint training and MICB region broadcast did not improve the semantic main metric.
