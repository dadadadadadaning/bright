# Release validation — 2026-10-22

Artifact: `bright_route_memcheck_v1.pyz`, **190325 bytes**.

SHA256: `93d2a756138c496a8a3602c821c5de03c970a44137a0c18225c947c2c625a488`.

The final local run passed **13 tests in 6.663 seconds** with Python3.10.22 / PyTorch2.1.2+cpu. It exercised:

- Historical six-arm model/support outputs remain identical and parameters/bindings are restored.
- An injected bit51 parent-index change in R's classification head is localized to its before/after operation window, and independent mirror/repeated-copy observations are preserved.
- A simulated first-CPU-copy-only change retains evidence that the device comparison and subsequent copies agreed with the baseline, without assigning a hardware cause.
- A changed reference mirror is distinguished from an unchanged original index in the saved observations.
- A simulated CUDA exception does not trigger additional corroboration reads.
- A complete356 synthetic-image evaluation remains diagnostic, with `complete=false` and `full_test_evidence=false`; wrong targets and truncated journals are rejected.
- The expected prior incident identity is validated.
- Sanitizer success requires an explicit zero-error summary; a missing summary or an earlier nonzero summary is not accepted.
- A real child-process timeout terminates the child and retains the log. A timeout cannot accept a stale successful result, and no retry occurs.

ZIP CRC, compilation of all packaged Python modules, the byte hashes of all49 inherited entries, source/archive equality for the six added modules, and packaged `--help` execution passed.

The released five-step sweep is inherited byte-for-byte. `build.py` creates separate diagnostic assessment/scoring adapters with checked single-occurrence substitutions; it does not alter the model, slot construction, checkpoint, original sources or published sweep.

Local tests did **not** execute real CUDA, Compute Sanitizer or the user's checkpoint/data. The user server must perform that part. Instrumentation alters execution timing and allocation layout; neither a clean diagnostic nor an observed bit difference alone proves or clears hardware failure.

Reproduction command in the development workspace (not a server command):

```bash
BRIGHT_RESUME_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_resume_v1 BRIGHT_ABLATION_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_ablation_v1 BRIGHT_ABLATION_REFERENCE=/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2 /workspace/bright-audit-legacy-env/bin/python -B test_memcheck.py -v
```
