# Validation — 2026-10-22

Artifact: `bright_memcheck_memory_v1.pyz`, 209386 bytes (CUDA initialization-order fix).

SHA256: `9c46b2f2137184a73391dd9213d224de5966606b28a149adb3a995eb093f8554`.

Final local run: **14 tests passed in 1.267s**, Python3.10.22 / PyTorch2.1.2+cpu.

The first release's CPU tests did not cover CUDA lazy initialization. Server run `memory1500_jlccsmkv` exposed a diagnostic startup bug: early `mem_get_info()` initialized CUDA before the original deterministic policy could be applied. This revision removes the early sample and guards every sample with the non-initializing `is_initialized()` query. It does not weaken or remove the original deterministic policy guard.

- A regression test executes the actual project's `configure_runtime` function with emulated CUDA lazy initialization, verifying that an early observation leaves the original policy configuration possible, followed by sampling after runtime initialization.
- A separate test passes the real uninitialized CPU-build PyTorch module to the sampler and verifies that it does not attempt CUDA initialization.

- Original historical six-arm CPU model outputs, parameters and RNG are unchanged with telemetry installed; module hooks are removed on exit, and before-forward records identify the correct arm.
- Injected frontend OOM preserves the original exception and performs no further allocator/CUDA queries.
- Memory summaries use successful samples and historical allocator peaks, not synthetic post-fault values.
- Complete two-image journals remain diagnostic; truncated journals are rejected. AdamW.step is forbidden by the fixture.
- Plain and instrumented commands differ only in the explicitly recorded case and Sanitizer wrapper. Each gets a fresh process.
- A real child timeout terminates the child process group and preserves logs.
- Plain failure skips instrumented execution, without retry. Even pair completion never marks full Test, historical recovery, or hardware clearance.
- Missing explicit zero-error Sanitizer evidence cannot be accepted as a clean instrumented case. Error-header extraction includes allocation and invalid-access records.

ZIP CRC, compilation of all archive Python entries, byte hashes of **56 inherited entries**, source/archive equality for six new modules, packaged `--help`, and whitespace checks passed. The original published memcheck modules remain byte-for-byte present; the two-image adapters are separately named.

The frontend CUDA memory path and real Compute Sanitizer were not run locally. The measurements use scalar CUDA allocator APIs and hooks and therefore change timing. This release is a resource diagnostic, not an asserted OOM fix or an experiment performance result.

Reproduction in the development workspace:

```bash
BRIGHT_RESUME_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_resume_v1 BRIGHT_ABLATION_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_ablation_v1 BRIGHT_ABLATION_REFERENCE=/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2 /workspace/bright-audit-legacy-env/bin/python -B test_memory.py -v
```
