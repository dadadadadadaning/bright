# Reading log

## 2026-09-01

- Verified the BRIGHT dataset paper, official CVPRW 2026 challenge page, and challenge-outcome preprint.
- Verified official publication pages and code repositories for BESTIE, IRNet, Panoptic-DeepLab, TernausNetV2, BoxInst, PointSup/PointWSSIS, and PointSAM.
- Compared prior mechanisms with the local pseudo-region, center/boundary/offset, watershed, gate, rollback, H7 oracle, RGID oracle, and CMR audit evidence.
- Main transfer conclusion: borrow confidence-aware pseudo-label refinement and instance-balanced supervision; do not import Memory, event-centering, or hard local-token selection into the semantic-improvement experiment.

## 2026-09-03

- Read the BRIGHT Challenge 2026 solution and discussion sections, including first- and second-place architecture details, cross-event analysis, metric recommendations, and limitations.
- Verified the current BRIGHT repository and the DFC 2025 Building-Guided official implementation.
- Reconciled the literature with new local evidence: LocCC+A6 improves Val450 and beats matched local smoothing, while T1 component balancing, MICB broadcast, and CMR do not support the main semantic claim.
- Ranked the next innovations: reliability-gated split-on-demand first, transformation-equivariant center/boundary learning second, optical-anchored building-query severity reasoning only after C3 succeeds.
- Added an isolated SPI-RGID semantic orchestration contract so C0--C3 cannot silently mix official-instance training, Memory, new classifiers, or test-time adaptation.
