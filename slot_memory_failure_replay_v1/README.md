# Bounded slot/Memory failure replay

This standalone diagnostic uses the installed, hash-verified addon and source.
It does not modify their files, the failed run, its checkpoints, or its pending
evaluation flags. All new files go to a unique directory under
`SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS`.

For the reported run it restores checkpoint 1000, verifies the complete recorded
sample order, and performs the original updates 1001–1397. These are **real AdamW
updates**, with the original loss, six-arm order, learning-rate schedule, clipping
and per-arm RNG handling. It stops at 1397; it does not continue to 2500.
The generic bound is at most 500 new updates per arm.

The primary worker inherits the failed run's recorded `CUDA_LAUNCH_BLOCKING`
setting. It adds flushed CPU phase markers around the unchanged `update_arms`.
It does not add dense NLL snapshots to this replay. After all arms finish 1396,
it saves a separate diagnostic checkpoint with complete heads, optimizer state,
RNG, parent/journal identity and inherited input ledger. The current worker's
verified input subset is recorded separately; unvisited inherited files are not
claimed to have been reverified.

If the primary attempt fails after the complete 1396 checkpoint and before a
complete 1397 record, the supervisor waits for that process to exit and launches
a **new process**. It restores the diagnostic 1396 state and tries only 1397 with
blocking enabled and saved-NLL-target snapshots. A partial-arm failed update is
never published as a complete checkpoint. Cached CPU NLL inputs are saved when
available, without reading CUDA again in the failure handler.

Successful prefix replay is compared with the historical log. Image, step,
optimizer counters and LR must match. Loss, gradients and norms report exact
matches, first differences and worst errors. The fixed comparison tolerance is
`atol=1e-5, rtol=1e-4`; ordinary finite drift is reported rather than aborted.
This is diagnostic evidence, not proof of identical historical internal state.

Checkpoint saving, phase markers and fresh-process allocation history change
timing. The original step-1000 evaluator restores RNG, but its allocator/cache
history is not reconstructed. Passing this diagnostic does not establish the
cause of the original CUDA assertion or a fix.

## Run

Use the same server Python environment that ran the failed experiment:

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -u bright_slot_memory_failure_replay_v1.pyz
```

The defaults target the reported failed run and seed 3407. `--run-root`,
`--seed`, `--addon-dir` and `--output-base` are available for explicit paths.
Internal worker options are used by the supervisor only.

The final `BEGIN ABLATION REPLAY SUMMARY` / `END ABLATION REPLAY SUMMARY` block
contains the compact outcome. The full `summary.json`, per-worker summaries,
`markers.jsonl`, `replay.jsonl`, and any traceback/checkpoint are retained in the
printed diagnostic directory. The exit code remains nonzero if the primary
worker failed, even when the fresh blocking worker passes.

## Local validation

Run from the delivery repository with CPU PyTorch installed:
Set `BRIGHT_ABLATION_REFERENCE` to the verified source package containing
`deps/helpers/continuation.py`; the integration tests use its actual fingerprint
and learning-rate helpers.

```bash
export BRIGHT_ABLATION_REFERENCE=/path/to/verified/source-package
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python -m unittest discover -s slot_memory_failure_replay_v1 -p 'test_*.py' -v
python slot_memory_failure_replay_v1/build_zipapp.py
python bright_slot_memory_failure_replay_v1.pyz --help
```

Tests exercise original addon updates under observation, exact AdamW/RNG
restoration, bounded replay and fresh-worker handoff with CPU fixtures, complete
checkpoint publication, lineage protection and historical-order comparison.
They do not reproduce a CUDA device assertion or substitute for the server run.
The vendored `probe.py`, `inspect_failure.py` and `nll_trace.py` are unchanged
from the preceding diagnostic release.
