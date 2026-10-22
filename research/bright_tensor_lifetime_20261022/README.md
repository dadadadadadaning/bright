# Small synthetic tensor-lifetime comparison

Motivation: memory1500_bfzr43fv completes two images normally (peak1.986GiB),
but under Compute Sanitizer allocated memory rises from0.339 to7.963GiB in the
first front.base and by about0.5GiB per head, reaching22.769GiB during image2.
This suggests retained allocations; it does not identify ownership or cause.

This standalone script does not import BRIGHT, access Train/Val/Test, read
checkpoints, construct a model, or perform optimizer updates. It runs two fresh
processes sequentially: plain Python, then the same synthetic arithmetic under
Compute Sanitizer memcheck. Each performs eight rounds of three8MiB float32
tensors. Values0.25 ->0.75 ->0.5625 are exactly representable. After the local
function returns, only weak references are retained. Record allocator allocated,
freed and active counters, live Python weakrefs, and Python trace/profile flags.
Explicit gc.collect at rounds4 and8 is a declared intervention in BOTH cases.
No empty_cache, allocator changes, mixed precision, or silent retry.

The pair uses physical GPU0 (CUDA_VISIBLE_DEVICES=0 in both children), requires
GPU0 idle, takes the same shared advisory locks as prior BRIGHT diagnostics,
and has a180-second budget per child. A1GiB PyTorch allocated-byte guard is
checked at successful observation boundaries. This is not a hard total-GPU
memory cap: sanitizer/driver overhead and temporary allocations can exceed it.
Each independent child exits on error; no CUDA queries follow an exception.
Logs only, no tensor snapshots. Original files are not modified. New evidence
is written under SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS/tensor_lifetime_*.

Run with the original server Python:

```
python -B bright_tensor_lifetime_v1.py start
python -B bright_tensor_lifetime_v1.py status
python -B bright_tensor_lifetime_v1.py logs
```

The start command prints the per-job runner.py paths. Prefer these for a fixed
job. It detaches from SSH. Ctrl+C during logs stops only the tail. The job ends
itself within the bounded case budgets (plus launch/termination overhead).

Interpretation:
- Python weakrefs remain live and allocations grow: retained Python objects or
  frames are a useful lead. Inspect trace/profile flags and response to GC.
- Weakrefs die but allocated bytes grow: investigate C++ ownership/allocator/
  tool interaction; weakrefs cannot enumerate all C++ references.
- This synthetic pair stays small: move back to model-specific operations or
  observer lifetime, with a small isolated workload. Do not declare the model
  or hardware cleared.
- Allocator counters may be uninformative with a non-native allocator or
  PYTORCH_NO_CUDA_MEMORY_CACHING. Runtime/backend/environment are recorded;
  compare like-for-like before drawing conclusions.

Completing this probe never completes Test878, proves hardware health, or
resolves the historical route corruption. A missing sanitizer error summary
cannot be accepted as a zero-error run.

Validation: eight CPU/standard-library tests cover weakref lifetime including
a deliberately retained frame, absence of Torch imports at launcher startup,
same child computation, real timeout termination, missing sanitizer evidence,
stopping after plain failure, and restricted success claims. No local CUDA or
Compute Sanitizer is available, so real server memory behavior is unverified.
