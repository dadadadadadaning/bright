# Validation

Executed locally with Python 3.10.22 and PyTorch 2.1.2+cpu. No remote RTX4090 execution was available. Synthetic data tests exercise bookkeeping and original CPU model operations; they do not establish the cause of the historical CUDA failure or predict server throughput.

## Results

- `test_night.py -v`: **19 tests passed**, 29.849 seconds.
- After extending the existing full-inference test to finalize its full878 report, reran `InferenceTests.test_all878_metrics_transitions_and_rng_equal_without_updates`: **passed**, 24.284 seconds.
- Packaged `--help`: passed without Torch/GPU initialization.
- Deterministic rebuild: identical SHA256.
- ZIP CRC, compilation of every packaged Python member, AST parsing of source files: passed.
- Original inherited members are byte-identical to the pinned recovery archive; new packaged modules match local source bytes.

Archive SHA256:

```
4610a882f4f3b0936ff105d5a8a8a0375e169ef00ce88d1eb03781d707bd3d2f
```

## Meaningful coverage

1. Actual original `JointRegionMemoryClassifier` and real `SupportFactory` on small CPU fixtures: all six arm predictions and head states match with/without route snapshots. B/D/R shared route objects are identified.
2. Injected shared child-route mutation after B: watched mode catches the change before D executes, preserving before/after CPU tensors.
3. The same injected failure in passive mode: original D route validation fails, original failed `allclose` operands/results and child-route dimensions are captured.
4. Injected CUDA-like exception: primary exception retained, no new snapshots after the failed call, wrappers restored. This is an exception-handling test, not an actual device-side assert.
5. NaN/Inf CPU diagnostic statistics remain JSON serializable.
6. Original fixed-Test inference loop with 878 synthetic records: confusion/transition aggregates, RNG and head hashes match with/without observation. Optimizer calls are prohibited. Full report promotion requires all878 and integrity checks; 622 records remain diagnostic-only.
7. Night coordinator preserves passive failures while allowing independent predeclared comparisons; watched-prefix failure or timeout blocks full Test. Setup/integrity faults stop later stages.
8. A real spawned sleeping child is terminated by the budget and its partial log preserved.
9. Case acceptance requires model, optimizer, RNG, input/source and route evidence. Stale success metadata after timeout is rejected.
10. Read-only preflight fixtures reject wrong checkpoint, wrong failure/order and incomplete journal. Legacy certificate gates are mocked only in those fixture tests; production calls the original gates.

## Reproduction

From the repository root, with the historical CPU fixtures present:

```bash
BRIGHT_RESUME_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_resume_v1 BRIGHT_ABLATION_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_ablation_v1 BRIGHT_ABLATION_REFERENCE=/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2 /workspace/bright-audit-legacy-env/bin/python -B research/bright_test_route_night_20261022/test_night.py -v
```

These tests do not run the complete remote worker against its checkpoint/data/ledger. The server preflight and post-inference gates remain required. Observation synchronizes CUDA and changes timing; `historical_failure_resolved` remains false even when every stage passes.
