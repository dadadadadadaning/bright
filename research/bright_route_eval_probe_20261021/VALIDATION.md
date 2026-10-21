# Validation

Environment: Python3.10.22, PyTorch2.1.2+cpu. No server CUDA execution was performed.

Run from the repository root:

```bash
BRIGHT_ROUTE_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_route_probe_v1 \
BRIGHT_RESUME_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_resume_v1 \
BRIGHT_ABLATION_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_ablation_v1 \
BRIGHT_ABLATION_REFERENCE=/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2 \
/workspace/bright-audit-legacy-env/bin/python -B -m unittest discover \
  -s research/bright_route_eval_probe_20261021 -p test_route_eval_probe.py -v
```

12 CPU/synthetic tests pass:

- Last successful D probability is not mislabeled as failing R forward.
- Real original classifier probabilities, parameters and optimizer remain unchanged.
- Successful prefix validators do not perform extra tensor snapshots.
- Dense observations preserve input bytes, RNG, original returns and bindings.
- Negative-weight and wrong-mass evidence retains the exact original exception.
- Equal bytes accepted earlier are explicitly linked to their callsite/arm.
- Runtime exceptions trigger no new tensor reads by the observer.
- Failure CPU cases can be saved without reading original tensors again.
- Every bundled original resume helper matches its hash ledger.
- Preflight rejects a failed run that already made new updates.
- Worker restores real six-head model/AdamW states and completes a synthetic prefix
  with no optimizer steps; it never promotes the prefix to a formal experiment.
- Worker failure preserves the active arm and completed-image count.

`--help` is also checked without importing Torch or accessing server files.
The real server checkpoint, split, code hashes and execution protocol are checked
by launcher/worker preflight; a successful local test cannot certify those inputs.
