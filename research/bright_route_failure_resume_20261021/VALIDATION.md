# Local validation

Python3.10.22 / PyTorch2.1.2+cpu. No remote GPU experiment was executed here.

15 CPU/synthetic cases passed, including the original continuation suite and
the new wrapped-continuation suite. The final three wrapped integration cases
were run separately after adapting the test launch identity to the zipapp.

```bash
BRIGHT_RESUME_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_resume_v1 \
BRIGHT_ABLATION_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_ablation_v1 \
BRIGHT_ABLATION_REFERENCE=/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2 \
/workspace/bright-audit-legacy-env/bin/python -B -m unittest discover \
  -s research/bright_route_failure_resume_20261021 -p test_failure_resume.py -v
```

Validated properties:

- An accepted real validator calls original allclose once, returns the original
  ID view and leaves input bytes unchanged, with no failure-time copying.
- Negative weights preserve short-circuit behavior and skip allclose.
- Bad sums save the exact original allclose operand tensors.
- A simulated false original allclose remains a failure even if the subsequent
  CPU check of its operands passes.
- RuntimeError is re-raised without subsequent observer tensor reads.
- Evidence I/O failure does not replace the original model exception.
- Real original classifier logits, gradients and AdamW parameter updates match
  exactly with and without scoped predicate observation on the CPU fixture.
- Scoped runtime/factory/evaluator wiring captures active R, image and step, and
  restores bindings after a real validator rejects a synthetic negative route.
- All inherited archive helpers match the pinned byte hashes.
- Original and wrapped continuation suites each verify successful completion,
  partial paired-update failure/recovery, and endpoint validation recovery without
  extra optimizer updates. Model/RNG checkpoints and journal ordering match.

The workflow fixture uses a tiny model and synthetic evaluation reports. Route
semantics are tested separately with the real original classifier. These tests
do not prove a fix for the server's intermittent CUDA error or certify accuracy.
