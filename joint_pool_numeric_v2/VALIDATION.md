# Validation performed in the cloud workspace

Environment: Python 3.12, PyTorch 2.2.1+cpu. No CUDA device, server checkpoint, or real failing snapshot was available.

- 54 CPU unit/integration contracts passed with no skips at initial integration: 20 numerical core, 26 coordinator, 8 worker contracts.
- FP32 literal arithmetic matched the repository's original `association.soft_pool_unpool` exactly on a CPU fixture.
- Independent grouped FP64 reference was tested without calling scatter/index_add or the literal implementation, and compared with FP64 scatter at an auxiliary 1e-12 tolerance on small fixtures. The reported diagnostic bound remains 2e-5 absolute plus 2e-5 relative.
- Duplicate routes, unnormalized weights, empty/tiny-mass regions, cancellation, saved squared features, asymmetric relative tolerance, noncontiguous comparisons, source/hash binding, failed-child isolation and partial completion were covered.
- Actual worker CLI completed a synthetic production-shape case: content [1,512,256,256], weights [1,9,65536], IDs [9,65536], 1024 regions. No BRIGHT images were used. The two FP64 references differed by at most 3.56e-15 in tokens. This is performance/implementation evidence for the CPU diagnostic, not evidence about the user's numerical failure.
- The original BRIGHT tracked files and the V1 recovery package were not edited by this work.

Real GPU native/contiguous results, the new `input_000002.pth`, and full Val450 remain unverified. A passing single-input diagnostic does not establish readiness for training recovery or justify a tolerance change.
