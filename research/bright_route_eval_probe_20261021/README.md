# Val1500 route-failure diagnostic

This artifact investigates the boundary evaluation failure from
`resume_20261021_090401_436171`, image index 167 (the 168th image),
`la_palma-volcano_00000822`. It performs **zero optimizer updates**.

The old observer recorded `path=d, stage=probability_validated`. That identifies
the last successfully validated probability, not the failing forward. The
original evaluator's order is Grid, E, F, B, D, R, L, so R is the inferred next
forward. The traceback reaches the child `route_q` validator in memory unpool.
The new observer records the active arm before entering each forward.

The original check rejects negative weights or a per-pixel route sum outside
`torch.allclose(..., atol=2e-5, rtol=2e-5)`. The exception alone does not establish
which predicate failed. Neither a tolerance change nor route normalization is
implemented here. The previously passed Val450 and all source runs stay intact.

## Scope

- Verify the original checkpoint1500, original source manifests, completed
  standalone Val450 evidence, and the new failed run's zero-update status.
- Restore all six saved heads, AdamW states and RNG. Initialize the remaining
  Train iterator in the same place as continuation, but never consume a training
  batch, call backward, or execute an optimizer step.
- Evaluate the first168 Val images in their declared order, through the original
  evaluator and support factory. The prefix report is a smoke/diagnostic result,
  never a completed Val450 result or a completed experiment.
- On image168, record the original route-validator calls at classifier entry
  and unpool, including caller locations, active arm, weights/IDs hashes,
  storage/version metadata, CPU float32/float64 sums, negative counts and first
  bad pixels. CUDA predicates are separately recomputed; they are not the
  original validator's intermediate tensors.
- For earlier images, avoid extra successful-route tensor copies/reductions.
  If a ValueError occurs earlier, capture that failure instead of skipping it.
- On a route ValueError, compare pre/post snapshots and note matching bytes
  accepted by an earlier original check. On RuntimeError the observer makes no
  further tensor reads; original evaluator/runtime cleanup retains its behavior.
- Retain CPU snapshots, full markers and traceback in a new diagnostic directory.
  Never repair inputs, skip images/arms, promote old reports or retry automatically.

Observations synchronize CUDA and change timing. This is a prefix reproduction
attempt, not an exact replay of the old CUDA execution. Passing it does not
resolve the prior failure, certify GPU health, or authorize a scientific claim.
There is no continuous tensor watch; changes between captures can be missed.

## Run

Use `/root/shared-nvme/conda-envs/bright4090/bin/python -B` with
`bright_route_eval_probe_v1.pyz start`, `logs`, or `status`.
`start` checks evidence, pins its own artifact, and launches a background worker.
`logs` and `status` select this diagnostic's latest directory, never the old
training log. Ctrl+C exits the log viewer; it does not stop the background worker.

The launcher shares the existing continuation and standalone-evaluation locks.
A live run holding either lock prevents a second launch. It also checks existing
source worker locks before GPU work. Outputs are under
`SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS/route_eval1500_*`.

After completion, inspect `summary.json`, `route_summary.json`, `route_trace.jsonl`
and `route_cpu_case.pt`. A failure's `active_arm`, callsites, CPU mass statistics,
negative counts, separately recomputed CUDA predicates, and hashes guide the next
fix. `diagnostic_complete=true` means only this prefix diagnostic completed;
`complete` and `full_validation_evidence` remain false.

## Build and validation

`python build.py /path/to/bright_slot_memory_resume_v1.pyz`

The base archive must have SHA256
`0f7fd0a85ca2db5727509875c84c1beba1011d2bd4300a8b6b5ea86b59e75f07`.
Its helper modules are bundled byte-for-byte with a verification ledger.
The observation support files in sibling research directories are bundled too.
`route_trace.py` is copied unchanged from the earlier route probe; its CLI is not
used. The artifact requires the user's existing verified server repository/data.

CPU tests cover real original classifier forwards and evaluator ordering,
restoration of real six-head AdamW state, zero updates, partial journal accounting,
hook restoration, original exceptions, bounded owned CPU snapshots, source
preflight, and preservation of nonformal result status. They do not reproduce the
server CUDA fault. See `VALIDATION.md` for the exact command.
