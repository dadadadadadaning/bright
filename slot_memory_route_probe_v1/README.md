# Shared-route single-image diagnostic

This standalone archive investigates a parent-route validation failure from the
bounded six-arm replay. It operates in the original server Python environment and
writes only a fresh `route_probe_*` directory under the diagnostic output base.
The original experiment, checkpoints, addon and source packages are inputs.

The input replay failed on arm B after E and F returned from their optimizer
steps. Its traceback identifies `parent_ids = validate_routes(q, ids, ...)`.
This identifies the shared parent route, before the child-route checks. It does
not establish whether route values changed or why validation failed.

## What is executed

1. Verify the failed replay, original journal/manifest, addon/source ledgers and
   exact parent checkpoint SHA. Verify the image against the historical journal
   and reconstructed original DataLoader order.
2. Restore all six heads, full independent AdamW states and RNG from that
   checkpoint. Preserve the original numeric protocol and recorded
   `CUDA_LAUNCH_BLOCKING` value.
3. Prepare only the historical failure image, then call the original
   `ablation_runner.update_arms` once with its original losses, per-arm RNG,
   clipping, optimizer steps and arm order.
4. Observe both actual validator bindings and parent/soft/single route snapshots
   across preparation, forward, loss, backward entry and optimizer boundaries.
   The optimizer pre-hook occurs after backward and gradient clipping.
5. Restore all temporary observation hooks. Save metadata, and bounded retained
   CPU tensor evidence if a failure, watch change or observation error occurs.
   Never save or promote trained model checkpoints.

With the default evidence, the image is from historical step **1303**, but the
restored checkpoint is **1000** and the one diagnostic optimizer update is
**1001**. These fields are separate in every summary. This is an isolated probe,
not a historical-state replay or a resumption of the failed experiment.

## Observation semantics

Each original validator is called exactly once with its original arguments.
Its return object or original exception is preserved. Diagnostic errors are
recorded separately. Validators are located through the live classifier's
function globals and `sparse_unpool` globals, rather than by importing a second
copy of the source.

CPU snapshots own their storage; their byte hashes detect even changes made via
`.data` that do not increment the original tensor's version counter. CPU float32
and float64 reductions are diagnostic recomputations. GPU reductions are also
labeled as recomputations, distinct from the original validator's outcome.
Snapshots synchronize CUDA and change allocation/timing. A passing probe does
not resolve the earlier route failure or original NLL device assertion.

Storage audits read tensor metadata only. They report shared storage between
watched inputs and trainable parameters, present gradients or AdamW state.
Aliases between frozen inputs are reported separately. Disjoint views of the
same allocation are distinguished from proven byte overlap and conservative
strided bounding intervals. An observed alias or changed watch is recorded
without replacing the original update's acceptance checks, so the next original
validator can still supply its own outcome. Such observations produce a nonzero
exit status even if the isolated update otherwise finishes.

Tensor retention is bounded to initial/latest/first-change watched cases and
latest/first-failure cases per validator binding. No per-call tensor history or
autograd graphs are retained. Failure serialization uses those CPU copies only.

## Run

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -u bright_slot_memory_route_probe_v1.pyz
```

Optional arguments: `--replay-output`, `--output-base`, `--addon-dir`.
The default replay directory is the evidence supplied for this investigation.
A mismatching or incomplete lineage stops before GPU work.

The final terminal block is delimited by `BEGIN ABLATION ROUTE PROBE SUMMARY`
and `END ABLATION ROUTE PROBE SUMMARY`. Full metadata is in `summary.json` and
`observations.jsonl`; original exceptions are in `traceback.txt`. A retained
`route_cpu_case.pt`, when present, contains diagnostic tensors, not trained
model weights.

Local tests use CPU synthetic inputs and the original updater. They cannot
validate CUDA hardware, the server dataset, or reproduce historical GPU timing.
