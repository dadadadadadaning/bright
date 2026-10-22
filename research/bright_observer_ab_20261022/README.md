# One-image route observer comparison under Compute Sanitizer

This continues fault diagnosis; it does **not** continue optimizer training.
The six-arm training endpoint is already2500. This diagnostic restores the
same1500 checkpoint used in the earlier memory comparison, uses the first
canonical Test image only, and runs two fresh processes sequentially:

- `light`: original model/runtime/feature-capture hooks and input validators,
  plus scalar memory telemetry and passive failure recording. No added CPU
  route snapshots, parent-index mirror, or per-operator route wrappers.
- `watched`: same computation, checkpoint, image, original checks and scalar
  telemetry, plus the existing added route observer from memory1500_bfzr43fv.

Both cases run under Compute Sanitizer memcheck with CUDA_LAUNCH_BLOCKING=1.
They keep the original E/F/B/D/R/L order, precision, loss-independent inference,
support construction, and probability/route validation. No tensor correction,
route renormalization, `empty_cache`, extra GC or mixed precision is added.
The scalar monitor wraps the same front modules in both cases. `light` leaves
the original feature-capture hooks intact; it is not a wholly hook-free model.
The output explicitly says route-snapshot checks were not performed in `light`.

The launcher verifies original source/checkpoint/input provenance and the
completed earlier memory pair. It reuses shared advisory experiment locks,
requires the GPU idle, creates a new output directory and never overwrites an
old run. No training updates and no automatic full Test or training follow-up.
One unsuccessful case stops the pair, with no retry. Default600sec per child,
60..1800sec accepted; setup/hash and shutdown time is additional. Launch is
detached and survives SSH disconnect. Ctrl+C in logs stops tail only.

```
python -B bright_observer_ab_v1.pyz start
python -B bright_observer_ab_v1.pyz logs
python -B bright_observer_ab_v1.pyz status
python -B bright_observer_ab_v1.pyz stop
```

Use the printed per-job runner.pyz path to select an exact job. Results live in
SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS/observer_ab1500_* with `light/` and
`watched/` memory.jsonl, reports, sanitizer logs and integrity evidence. Normal
output is small text; if the watched route observer catches a change, it retains
its existing CPU tensor snapshots, which may use more disk. No checkpoint or
dataset copies are produced.

Interpretation: lower memory in light implicates the added observation path or
its interaction with the tool, not a specific callback by itself. Growth in
both cases shifts focus to the original runtime/capture hooks/model operators.
One-image success cannot prove sustained multi-image memory stability or resolve
the intermittent route bit changes. Numeric differences between the two runs
must be inspected; neither is a new performance benchmark.

This preserves all inherited archive entries byte-for-byte and adds separately
named adapters. Missing/full-test evidence is never promoted. End-state checks
cover heads, optimizer state, RNG, input hashes, original runtime/guard, source
and checkpoint. Original validators remain active in the light case.
