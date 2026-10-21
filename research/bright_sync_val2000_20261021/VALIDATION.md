# Validation — 2026-10-21

Local environment: Python 3.10.22, PyTorch 2.1.2+cpu. No remote SSH, BRIGHT tensors, or CUDA reproduction was available here.

## Build

```bash
python research/bright_sync_val2000_20261021/build.py
```

Output SHA256: `14a3cb440ce49899da7d7b81061b9b2df66ab88526203919f28823bf8b3617ab`.

All inherited modules are byte-identical to the pinned index diagnostic archive (`cc639347f9c07409c10721fa8b12728219ec5d6893d98b97b238bbd6dfb3a164`); `SYNC_VAL_BASE.json` lists their hashes. Only the entrypoint and two new modules supply this standalone evaluation. The inherited index observer is not imported or activated.

## Tests

```bash
BRIGHT_RESUME_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_resume_v1 BRIGHT_ABLATION_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_ablation_v1 BRIGHT_ABLATION_REFERENCE=/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2 /workspace/bright-audit-legacy-env/bin/python -B -m unittest discover -s research/bright_sync_val2000_20261021 -p test_sync_val.py -v
```

15 CPU/synthetic tests cover:

- Run the actual six-arm evaluator on450 tiny synthetic images both with and without the new observer; compare metrics, per-event results, transitions, head hashes, and complete image-journal hash exactly. Check unchanged optimizer/head/RNG state and prohibit optimizer.step. CPU results retain smoke status.
- Inject forward, loss, and loader errors. Preserve the exact exception object and original traceback; skip CUDA RNG cleanup on failure, restore scoped Python bindings. Evidence-write failure does not replace the primary error.
- Finalize only a fresh2000 report after all450-image gates. Preserve the pending report and failed source run. Declare the synchronous protocol explicitly using the saved runtime manifest, since the original result format omits protocol_info; reject manifest/report protocol mismatches.
- Reject incomplete integrity checks, partial reports, wrong step, Test/smoke flags, protocol tampering, wrong image order, changed NLL objective despite rehashing, nested source outputs, and missing checksums.
- Verify the inherited archive ledger and the launcher's pinned worker copy and CUDA_LAUNCH_BLOCKING=1 environment before subprocess startup.

These checks verify implementation contracts. They do not prove the historical CUDA failure is fixed or provide real validation metrics. Successful remote evaluation must report450 images and all integrity checks; it cannot complete the original training run or establish equivalence to asynchronous execution.
