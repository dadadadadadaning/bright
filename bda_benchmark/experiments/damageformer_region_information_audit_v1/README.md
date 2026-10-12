# DamageFormer Region Information Audit V1

This isolated experiment answers two questions without touching Test:

1. **Partition gate**: do frozen building-internal regions respect labelled geometry?
2. **Information gate**: do correct region features add information beyond frozen A1?

The partition report contains boundary recall, pixel-weighted region purity,
under-segmentation error, a majority-label oracle upper bound, small affected-component
preservation, and event/building-size/damage-ratio breakdowns.

The information gate trains parameter-matched `Linear(146,3)` probes:

- `R0`: A1 state + zero region slots.
- `R1`: A1 state + real region descriptor.
- `R2`: A1 state + globally shuffled region descriptor.
- `R3`: zero A1 slots + real region descriptor.
- `R4`: A1 state + matched regular-grid descriptor.

Every region/grid descriptor uses the same frozen pooling fields and the same frozen V2
token encoder. No GT is used to construct regions or input features. GT is used only as
the audit/probe target. A conservative admission requires R1 to exceed R0, R2 and R4 by
at least 0.20 percentage points Macro-F1.

Important: BRIGHT rasterizes building-level damage labels. These audits must not be
described as proof of physical sub-building crack localization.

