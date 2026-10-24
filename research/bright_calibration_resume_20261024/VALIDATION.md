# Validation scope

Validated 2026-10-24 with Python3.10 / CPU Torch2.1.2: **66 tests passed**
(4 model-equivalence, 21 launcher, 29 preflight, 12 continuation tests).

The server checkpoint, dataset and RTX4090 are not available in the delivery
workspace. No formal training, GPU stability or BRIGHT metric result was
measured here. The remote start command remains for the user to execute.

CPU equivalence uses the published JointRegionMemoryClassifier and slots,
the unchanged old readout adapter, finite fixture tensors, pixel-table NLL,
and the actual Lovasz functions extracted from the original training source.
It checks:

- All forward outputs, readout diagnostics, parameter schema, inputs and RNG.
- All parameter gradients, including a non-None zero inactive gain gradient.
- Four AdamW updates and full optimizer state, including weight decay on a
 nonzero inactive gain parameter.
- Live gradients through the original Reader after pruning the unused path.
- 63 GatherBackward nodes versus45, with the18 removed nodes receiving zero
 gradients in the published control.
- Original, severity_memory and shuffled_memory retain their behavior.

Floating comparisons use `atol=1e-7, rtol=1e-6`. CUDA reduction ordering and
historical GPU execution are not covered by these tests.

The recovery tests also use a real CPU model, Dropout, AdamW and the published
checkpoint save function. After800 steps, model/optimizer restoration plus
raw-loader position reconstruction and RNG restoration produce exactly the
same801–804 losses and final parameter/optimizer/RNG state as uninterrupted
execution. This is a continuation contract test, not a BRIGHT replay.

Additional tests cover parent provenance and corruption rejection, inherited
evaluation journal identity, pending-state preservation, exact1700 update
budget, sample order, checkpoint-before-evaluation, failure accounting,
source snapshot/config binding, occupied GPU/lock handling, supervisor
timeouts, incomplete completion rejection and append-only shared logging.

Run from this directory with CPU Torch installed:

```bash
python -B -m unittest discover -p 'test_*.py' -v
```

`test_equivalence.py` uses environment variables for external source locations:
`BRIGHT_SEVERITY_PACKAGE`, `BRIGHT_REFERENCE_MODEL`, `BRIGHT_LOVASZ_SOURCE`.
The default paths refer to the delivery workspace. The runtime zipapp does
not bundle these reference models/tests; it uses the installed, pinned server
study sources. `python -B build.py` creates the deterministic runtime archive
and its SHA256 sidecar.
