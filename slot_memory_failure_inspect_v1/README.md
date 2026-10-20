# Read-only ablation failure inspection

`inspect_failure.py` is a standalone Python 3.9+ stdlib utility. It defaults to
the failed six-arm run `run_20261020_025710_3840142`, seed 3407. An explicit
`--run-root` may select another ablation run; the root plan and the seed's
training manifest must agree. Extended/source experiments are rejected.

Run with the same Python executable used for training:

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /tmp/bright_inspect_failure.py
```

It prints a short report containing:

- The run identity and workflow/training failure status.
- The last complete all-arm journal record and contiguous update prefix.
- Checkpoint SHA256 comparisons against their existing sidecars.
- The worker traceback and bounded error lines from its log.
- Hashes of recorded source paths and installed Torch package/build metadata.

It does not import Torch or repository modules, load weights, read images,
access CUDA, change files, launch or terminate processes, resume training,
or mark intermediate results complete. Keep it outside the hashed training
addon directory.

A newline is required for a completed journal record. Gaps, malformed records,
arm mismatches and inconsistent optimizer counts suppress the next-index
inference. An incomplete final line is reported separately. A file changing
during inspection cannot establish a stable final step.

The next unjournaled index is **not proof of the actual failure step or arm**.
A failed update may have changed only some arms before a record was written.
A verified checkpoint means its bytes and existing sidecar agree; this utility
does not independently validate tensors or implement recovery.

Current source hashes do not prove which modules the failed process imported.
Torch metadata describes the Python installation used for inspection, not an
independent observation of the failed process or the GPU driver.

Local checks use synthetic temporary files:

```bash
python -m unittest discover -s slot_memory_failure_inspect_v1 -p 'test_*.py' -v
```

These checks do not reproduce or fix the CUDA assertion.
