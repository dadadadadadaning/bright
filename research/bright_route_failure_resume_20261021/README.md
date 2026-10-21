# Continue the six-arm experiment with original route-check evidence

The captured image168 probe completed all six arms. Its 40 detailed route records
had zero negative weights. Maximum CPU float32, CPU float64 and CUDA sum errors
were 3.5762786865234375e-7, 2.1679443307220936e-7 and 2.384185791015625e-7.
The effective sum tolerance at reference1 is approximately4e-5. These successful
observations do not support increasing that tolerance, and do not explain the
earlier failure. Dense observation changes execution timing.

This release progresses the existing experiment while retaining the original
rejection decisions. It is an observation extension, not a claimed CUDA fix.

## Workflow

1. Verify the exact original seed3407 checkpoint1500 and its existing standalone
   Val450 certificate using the previous launcher's preflight.
2. Create a new continuation directory and restore the original six heads,
   optimizers and random state. Source runs stay unchanged.
3. Repeat the complete boundary Val450 and fixed-Train evaluation. An incomplete
   or failed boundary evaluation stops the run before new optimizer updates.
4. Only after boundary checks succeed, execute updates1501–2500 with the original
   arm definitions, NLL/Lovasz loss, schedule, sample order and checkpoint cadence.
5. Keep the original endpoint evaluations, integrity checks and completion rules.
   No Test evaluation, input repair, skipped arm/image, or automatic retry.

Run with the original server Python:

```text
bright_route_failure_resume_v1.pyz resume
bright_route_failure_resume_v1.pyz logs
bright_route_failure_resume_v1.pyz status
```

The original background launcher and lock are retained. `logs` follows the latest
continuation's `train.log`, not the completed prefix diagnostic log. A PID confirms
startup only. Completion requires the final run summary and validation evidence.

## What the observer changes

It installs scoped wrappers around the two `validate_routes` bindings used at
classifier entry and by slot operations. Each instrumented validator uses the
**same Python code object** as the original, with a private globals dictionary.
Only the global `bool` and `torch.allclose` lookups are observed:

- Original bool conversions and allclose are each called once, in original order.
- Original short-circuit behavior is preserved. A negative weight still prevents
  the sum check from running, just as in the original source.
- Successful checks perform no extra CUDA operations, CPU tensor copies or
  numeric recomputations. Predicate operands are retained only until return.
- On rejection, keep the original exception. For ValueError, copy the actual
  reduction operands used by that allclose call before separately examining Q.
- For RuntimeError, make no new tensor reads in this observer. Original runtime
  and evaluator cleanup behavior remains unchanged.
- A CPU check after failure is evidence only; it never overrides a rejected
  original CUDA check. No clamp, normalization, retry or tolerance increase.
- Active forward arm and image are recorded before model execution. A failure
  after the D probability check can therefore identify R directly.

Python dispatch and temporary operand lifetimes still change timing. Successful
continuation would establish completion for this recorded run; it would not
establish the cause or permanent repair of previous intermittent failures.

## Evidence in the new run directory

- `route_observer_identity.json`: original source/code hashes and observation scope.
- `route_observer_summary.json`: number of observed calls/failures and restoration.
- On route rejection: `route_failure.json` and `route_failure_cpu.pt`, including
  original boolean decisions, original allclose decisions/operands, callsites,
  active image/arm, and CPU checks performed after rejection.
- Existing `summary.json`, `traceback.txt`, probability observations and training
  journals remain available through the unchanged continuation workflow.

Original scalar decisions are observed directly. Operand tensor contents are
copied after failure, so this does not freeze their bytes at the exact comparison
instant or eliminate all possible races/corruption. Evidence-copy failure is
reported while retaining the original model exception.

## Build

```text
python build.py ../bright_training_next_20261021/bright_training_next_v1.pyz
```

The inherited archive must have SHA256
`994ee877c6edf332e4ebae70c48c81442877dcb6ab053e8cab4323ebb0075e3d`.
Every inherited helper is included byte-for-byte and listed in
`INHERITED_TRAINER.json`. The source validator and dataset files on the remote
server are not edited. See `VALIDATION.md` for local test scope and limitations.
