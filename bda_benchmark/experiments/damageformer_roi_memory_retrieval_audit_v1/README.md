# DamageFormer ROI Memory Retrieval Audit V1

This is an isolated, read-only diagnostic. It does not train a model, modify a
checkpoint or bank, or access Test.

It adds the evidence missing from the existing selective-Memory audit:

- GT-query-conditioned same-class Hit@K;
- region-macro and region-mass-weighted variants;
- Damaged posterior-to-Intact and Destroyed posterior-to-Damaged/Intact;
- support-class composition for every GT query class;
- kNN evidence restricted to predicted building/background query groups;
- predicted query/support group compatibility;
- per-event results;
- normal, random, shuffled-label, same-event, and same-predicted-group controls.

Same-class Hit@K and support-class composition use the stored support identity
before any label intervention. Posterior metrics use the labels actually
injected into the posterior; both original and injected class matrices are
saved. Posterior transitions are conditional on retrieving at least one valid
support, while Hit@K counts unsupported queries as misses and is accompanied by
support coverage.

Val GT is used only after retrieval indices are fixed. Same-group retrieval uses
the parity of the existing predicted `B>=threshold` hard route; it is not a GT
building mask and not a true building-instance constraint.

The summary separately reports all hard-labelled query regions and the
GT-purity >= 0.95 diagnostic subset. Missing event/class cells are `null`, not
a fabricated zero.

Run `run.sh preflight`, then a real-data `smoke`, then `audit`. Every output
directory must be new. `smoke` deliberately uses the formal source/bank but
limits the check to two Val images; it does not mix formal and smoke bank
identities.
