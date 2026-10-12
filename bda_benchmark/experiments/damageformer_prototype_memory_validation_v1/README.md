# Prototype Memory Validation V1

This protocol freezes the reproduced Stage-1/A1 Base (`valid_mIoU=61.6866%`) and
the observable `base_pred=Intact` candidate pool. It locks correction coverage at
2% before any Test access. Only `Intact -> Damaged` is permitted.

Primary and challengers:

- P2: original sample-weighted, leave-one-event-out single class prototype (frozen primary).
- P2b: single prototype obtained by averaging per-event class centroids.
- P7: K event-balanced prototypes obtained by clustering per-event class centroids.

Causal controls:

- P3: class-count-preserving shuffled train labels, 10 repeats.
- P4: swapped Intact/Damaged prototype semantics.
- P5: random unit prototypes, 10 repeats.

The report contains exact pixel metrics, per-event deltas, event-stratified image
bootstrap intervals, rescue/harm pixels, and size-stratified building impacts.
The historical fixed `z(Specialist)+z(Prototype), lambda=1` is stopped and retained
only as a negative/non-complementary result.

This is validation evidence only. Test access remains forbidden until the method,
coverage, checkpoints and all thresholds are frozen.

