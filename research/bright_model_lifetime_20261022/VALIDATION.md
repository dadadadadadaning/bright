# Validation

Local CPU environment: Python / PyTorch 2.1.2+cpu, inherited repository fixtures.
Fourteen focused tests passed:

- Original six-arm CPU predictions, weights and RNG preserved in both modes.
- Minimal mode leaves the runtime and forward method bindings unchanged and
  does not add forward hooks; instrumented mode restores its methods/hooks.
- Original invalid-route input rejection remains active.
- A primary simulated CUDA exception skips CUDA RNG restoration and GC.
- Tensor inventory returns weakrefs/metadata only; no strong-reference retention.
- A known tensor reference cycle can be collected after inventory returns.
- Explicit GC does not modify live tensor values or RNG and calls no empty_cache.
- Valid one-image diagnostic closure, refusal to promote it to full Test, both
  memcheck command constructions, archive/source lineage and pair state flags.

Command:

```bash
BRIGHT_RESUME_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_resume_v1 BRIGHT_ABLATION_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_ablation_v1 BRIGHT_ABLATION_REFERENCE=/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2 /workspace/bright-audit-legacy-env/bin/python -B test_life.py -v
```

The server GPU/model and real Compute Sanitizer are unavailable locally. No
claim is made that these CPU tests fix the OOM or earlier route corruption.
