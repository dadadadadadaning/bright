# Validation scope

Executed in the cloud workspace on 2026-10-13 using Python 3.12 and PyTorch 2.2.1+cpu. Server uses Python 3.10 / CUDA; real server execution is still required.

## CPU contracts

`/workspace/bright-env/bin/python -m unittest discover -s /workspace/bright-followup/joint-live-recovery-v2 -p 'test_*.py' -v`

101 tests passed, including:

- Literal FP32 pooling, independent grouped/scatter FP64 controls, duplicated routes, tiny/zero mass and dtype/layout validation.
- Exact same-call capture before the unchanged comparison; disagreement is not overwritten by a success flag.
- Explicit CPU backend dispatch, no automatic CUDA-fault fallback, snapshot reuse, and scoped restoration after failures.
- Single-image CPU probe runs all three heads with zero optimizer updates; a stats-only early exit or nonfinite Reader loss cannot pass.
- Original checkpoint/head/AdamW/RNG integrity and all three equal update counts.
- Synthetic split/resume produces the same weights, AdamW states, RNG and data order as uninterrupted CPU training; this tests the resume engine, not real CUDA equivalence.
- Failed checkpoint revalidation and changed input/data order stop before the next update.
- Required live/CPU evidence, independent FP64 aggregate, packet identities/hashes and persisted stage bindings cannot be bypassed by invoking the train worker directly.
- Legacy and new recovery lock exclusion, child cleanup on interruption, process ordering, heartbeat and report refresh.

Tests use the unchanged original training helper and BRIGHT pool source as local dependencies. On the server they are already present at the bound original paths. Unit tests do not establish that E0, Val450 or GPU execution pass.

Eight additional offline-launcher tests passed: exact embedded manifest extraction, payload corruption, invalid archive paths, undeclared members, file checksum mismatch, extract-only with no process, one detached background process with argument-list execution, and foreground report forwarding. The generated offline entry's `--help` was also executed successfully.

## Production-shaped synthetic CPU exercise

The actual original BRIGHT CPU function and scoped CPU runtime were exercised with `[1,512,256,256]` content, 1024 regions, 9 routes and FP32 weights. Original/raw-squared statistics were concatenated before the call.

All three outputs were bitwise equal to the original CPU FP32 function. Independent grouped FP64 and scatter FP64 agreed; comparison to CPU FP32 passed the unchanged `atol=rtol=2e-5`. Scoped hooks restored. Total synthetic capture and analysis time was about 17 seconds on the cloud CPU with 4 threads; this is not a server performance estimate.

Detailed observations are in `validation_production_cpu.json`. This was synthetic input with no model weights or BRIGHT image data.

## Unverified

Real GPU full-model live capture, real-source CPU probe, CPU transfer overhead, checkpoint1500 Val450, 1501–5000 updates and final Val450 have **not** run here. No Test split was accessed. The index assertion's original root cause is not claimed to be established. The explicit CPU backend is a declared workaround requiring its own gates, not proof that the old GPU backend is repaired.
