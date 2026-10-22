# Validation — 2026-10-22

Artifact: `bright_memcheck_memory_v1.pyz`, 209227 bytes.

SHA256: `9521778fab6d64d73b72a35eef3820a7199100a27e090643c367f61677147a93`.

Final local run: **12 tests passed in 1.305s**, Python3.10.22 / PyTorch2.1.2+cpu.

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
