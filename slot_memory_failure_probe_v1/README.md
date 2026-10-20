# Single-image CUDA diagnostic

This is a diagnostic for a failed slot-memory ablation, not a training repair or
resume driver. It leaves the original addon, run, checkpoints and result flags
unchanged. It creates a fresh diagnostic directory under
`SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS`.

Run the published zip application with the training Python:

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /tmp/bright_failure_probe.pyz
```

The default input is ablation run `run_20261020_025710_3840142`, seed 3407.
`--locate-only` stops after CPU sample-order verification. `--run-root`,
`--seed`, `--addon-dir`, and `--output-base` are explicit overrides.

The probe:

1. Verifies the original addon release and source ledgers.
2. Reconstructs the original first-epoch DataLoader order, including its base-seed
   consumption, and checks **every** completed journal image/step/epoch/seed
   and every arm's update count before choosing the next candidate image.
   A plain `randperm(seed)` is not equivalent and is rejected by the tests.
3. Selects the latest completed checkpoint whose bytes and sidecar agree.
   It verifies checkpoint plan, runtime configuration, protocol and all head
   fingerprints against the training manifest and sidecar.
4. Loads the original frozen runtime once, with `CUDA_LAUNCH_BLOCKING=1`.
   The explicit blocking-flag change is recorded; other execution/protocol
   settings must still agree. It loads only the candidate image, using the
   original Train loader's inference transforms. All actually used file hashes
   must match the checkpoint's input ledger.
5. Checks the original six arms sequentially in a fresh process. Each arm loads
   its checkpoint head and recorded checkpoint RNG state, then runs the original
   forward, original NLL+Lovász loss, original extra logging NLL, and backward.
   There is no optimizer construction, clipping or optimizer update.
6. Records the actual NLL autograd saved targets immediately after NLL, after
   the combined loss, after logging NLL, and before backward. The scoped wrapper
   calls the original function unchanged. Target range/hash/version changes
   stop the diagnostic; labels are never corrected, filtered differently, or
   clamped by the probe.
7. Stops on the first error. A CUDA assertion's poisoned process is not reused
   for another arm. If NLL inputs were captured, failure saving uses already
   retained **CPU** tensors without trying to read the failing GPU again.

The diagnostic reports the exact arm/stage of **its own** execution. Checkpoint
1000 weights do not reconstruct the state after update1396. Extra synchronization,
tensor references and snapshots also change timing/allocation. A successful probe
therefore does not prove that the original failure is fixed. It does not establish
the failing historical arm.

Artifacts:

- `probe_summary.json`: candidate, checkpoint, arm outcomes and failure stage.
- `trace.jsonl`: detailed label fingerprints, ranges, storage metadata and actual
  imported source paths/hashes.
- `traceback.txt`: diagnostic failure traceback, if any.
- `nll_failure_inputs.pt`: retained CPU NLL inputs on failure, when available.
- `runtime/`: original frozen-runtime guard artifacts in the new directory.

The original pending validation reports remain pending. No Test split is used.
The script neither upgrades Torch nor changes the original loss implementation.

## Local verification

CPU tests cover sample-order reconstruction, provenance rejection, exact original
NLL output/gradient preservation under observation, private filtered labels,
mutation detection, exception cleanup, and CPU-only failure saving. They do not
claim CUDA failure reproduction or validate the unavailable real dataset.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s slot_memory_failure_probe_v1 -p 'test_*.py' -v
```

`inspect_failure.py` is vendored unchanged from the preceding read-only collector
so the downloadable application is self-contained.
