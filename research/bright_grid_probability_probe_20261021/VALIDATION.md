# Validation scope

- Nine CPU tests pass in both available Torch test environments. Tests cover an
  unchanged valid input and RNG, one-ULP upper-bound violation, separate NaN/Inf
  counts and JSON serialization, type/shape/device/normalization errors,
  stopping at image 98 and restoring module mode, loader order rejection,
  non-promotion of a passing probe, and candidate/checkpoint evidence checks.
- Eight synthetic cases were also checked against the actual original
  `slot_memory_ablation_v1/evaluation.py::_validate_probability` using Torch
  2.1.2+cpu. Diagnostic validity classification agreed in all cases.
- The built zipapp's `--help` runs without Torch. Static call inspection found
  no calls to backward, optimizer step, training, worker, or launch entry points.
- The real BRIGHT dataset, frozen model and CUDA server are not present in this
  workspace. End-to-end GPU behavior and reproduction of the historical failure
  are not tested here. A successful frozen-prefix probe is not a completed
  six-arm evaluation and is not evidence that the previous failure is fixed.
