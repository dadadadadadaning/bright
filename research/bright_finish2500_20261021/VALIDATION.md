# Validation — 2026-10-21

Local runtime: Python3.10.22, PyTorch2.1.2+cpu. No remote SSH access, BRIGHT full-image tensors or RTX4090 execution was available. Tests verify implementation contracts, not model accuracy or resolution of the historical asynchronous CUDA failure.

Build SHA256:

`cb8a94d4235d2f0ce70f7c70c9d8e1c32285ba7a7a8faf58eceaf472b9e624e2`

The archive preserves every inherited helper byte from synchronous-Val archive `14a3cb440ce49899da7d7b81061b9b2df66ab88526203919f28823bf8b3617ab`. It adds four workflow modules and replaces only the archive entry point. Server source packages and existing runs are not patched.

16 tests pass:

```bash
BRIGHT_RESUME_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_resume_v1 BRIGHT_ABLATION_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_ablation_v1 BRIGHT_ABLATION_REFERENCE=/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2 /workspace/bright-audit-legacy-env/bin/python -B -m unittest discover -s research/bright_finish2500_20261021 -p test_finish.py -v
```

Coverage:

- Actual original six-arm updates with AdamW and nonzero optimizer moments on tiny synthetic images. Restore an artificial1000-step parent, run1001–1004, compare every recorded loss/sample and complete checkpoint head/optimizer/RNG fingerprints against uninterrupted expected updates. No evaluator call occurs during training. Original files remain byte- and timestamp-identical.
- Inject a B-arm backward failure. Do not record a partial paired step; retain the preceding complete checkpoint and original files.
- Score878 tiny synthetic Test images using real Torch heads and the original probability/confusion/transition helpers. Check shared frozen extraction, no optimizer calls, preserved RNG/module modes/head states, exact journal order, per-event and paired counts, and no Test loss fitting.
- Explicit Test878 admission with synthetic canonical split and input files. Val450 identifiers are not relabelled. Reject Train/Test overlap, missing input files and undeclared ledger entries.
- Reject wrong loader order, query-GT leakage, corrupted paired transitions and cross-split input.
- Preserve an injected primary device-assert-like exception without attempting poisoned RNG cleanup; leave no complete result.
- Finalize only fresh pending Val/fixed-Train reports after all integrity flags; maintain distinct full-validation and Test flags and keep pending bytes.
- Refuse Test before completed matching Val/fixed-Train evidence. A simulated evaluation failure after training retains training-complete status and prevents subsequent Test execution or automatic retraining.
- Verify fixed2500 selection and byte-identical inherited archive modules.

The distributed `--help` entry point was executed successfully. Remote GPU training, actual canonical Test input availability, and all scientific metrics remain to be verified by the user's run.
