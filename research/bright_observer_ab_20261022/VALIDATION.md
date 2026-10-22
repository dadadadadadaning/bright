# Validation

Nine CPU tests passed using the existing six-arm fixtures (Torch2.1.2+cpu):

- Both observer modes preserve all six prediction tensors, parameters, RNG and
  original method/hook bindings.
- The light case still rejects deliberately invalid parent route weights.
- A synthetic OOM propagates without subsequent CUDA telemetry calls.
- Single-image diagnostic finalization checks the journal and integrity flags;
  requesting promotion to full Test evidence is rejected.
- Both commands use Compute Sanitizer and the same worker with explicit modes.
- Inherited package bytes match their ledger; generated source matches ZIP.
- Completing both modes remains zero training updates and no historical fix.

The inherited timeout/stop/provenance implementation is unchanged from the
previous tested memory package. ZIP CRC, compilation, CLI help and whitespace
checks were also performed. No local GPU, remote dataset or original checkpoint
is available; real CUDA execution and OOM behavior have not been validated here.

Reproduction:

```
BRIGHT_RESUME_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_resume_v1 BRIGHT_ABLATION_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_ablation_v1 BRIGHT_ABLATION_REFERENCE=/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2 /workspace/bright-audit-legacy-env/bin/python -B test_ab.py -v
```
