# BRIGHT soft + Memory four-arm smoke fix

This revision fixes the `An unavailable parent was selected` failure in the
four-image smoke check. The original V3 ranker ignores the added child-value
eligibility flag; the bridge now masks unavailable parent keys in a temporary
retrieval view before ranking. The verified V3 bank, row IDs, same-event
exclusion, and strict post-selection validation remain unchanged.

Download `bright_soft_memory_joint_v1_fix1.zip` and follow
`soft_memory_joint_v1/README.md`. Use a new tool directory and a new `check`
run. The failed run's child values are bound to the previous code hash and
must not be reused.

The package passed 118 CPU tests, including a regression against the actual
V3 `select_neighbors` implementation, and archive extraction verification.
Real BRIGHT/CUDA smoke and formal Val450 training have not been run here.

ZIP SHA256: `85dfd4151e6f485ce8b0a7c4423b3e17c2985ad285c61dd42617ddd46030293e`
