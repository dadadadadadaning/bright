# Release validation — 2026-10-22

Artifact: `bright_test_sweep_v1.pyz`, 170913 bytes.

SHA256: `cb4dc39c75ab29971a9f1fe5d313ec7c1fd4b11dae3aaed5f74b561e9f7a9b70`.

Local environment: Python 3.10.22, PyTorch 2.1.2+cpu. There was no access to the user's GPU, real checkpoint files or Test878 images during release validation.

## CPU checks

The final `test_sweep.py -v` run passed all **13 tests**, in **26.504 seconds**. Tests use the historical model/support implementation for small tensor cases and synthetic image records for evaluation contracts.

- Six-arm model output is unchanged by the observers, and patched bindings are restored.
- An injected shared-route mutation inside D's Reader is localized between `_read_before` and `_read_after`, with CPU evidence retained.
- A simulated CUDA exception preserves the primary failure and restores bindings without attempting another GPU snapshot.
- Original step500/1000 checkpoint loading verifies optimizer counters and rejects changed hashes; resumed checkpoints with inconsistent training-journal prefixes are rejected.
- All five step labels are exercised with 878 synthetic image records each. Accepted reports export 35 rows; omitting step1500 exports 28 rows and explicitly marks that node missing.
- Partial, wrong-step, tampered-journal and forged-metric reports are rejected. Four-class and foreground means are checked separately. An event with zero valid pixels exports null metrics.
- Evaluation failure continues to the next distinct fixed checkpoint; setup/integrity failure aborts the remaining cases. No automatic retry occurs.
- A real child-process timeout terminates the child and preserves its log; a stale success report cannot override the timeout.

The archive also passed ZIP CRC validation, compilation of every packaged Python module, inherited-module hash-ledger validation, new-module/source byte comparison, and packaged `--help` execution.

## Scope

These checks do not establish CUDA stability or successful evaluation on the user's data. Real checkpoint lineage, source/input integrity and full-image scoring are checked again on the server. Observations synchronize CUDA and alter execution timing; passing an observed run does not resolve the historical route-mutation incident.

## Reproduction in the development workspace

```bash
BRIGHT_RESUME_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_resume_v1 BRIGHT_ABLATION_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_ablation_v1 BRIGHT_ABLATION_REFERENCE=/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2 /workspace/bright-audit-legacy-env/bin/python -B test_sweep.py -v
```

The paths above describe the development workspace, not commands to run on the user's training server.
