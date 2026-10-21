# Validation scope

Result: all14 tests passed; packaged CLI help also passed.

Python3.10.22, Torch2.1.2+cpu. Tests run against the packaged executable and the
original local classifier/evaluator sources. They do not reproduce server CUDA.

```bash
BRIGHT_ABLATION_TEST_SOURCE=/workspace/bright_ablation_publish_20261020/slot_memory_ablation_v1 BRIGHT_ABLATION_REFERENCE=/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2 /workspace/bright-audit-legacy-env/bin/python -B -m unittest discover -s research/bright_index_eval_probe_20261021 -p test_index_probe.py -v
```

Contracts covered:

- Legal expanded gathers preserve original output, source/index bytes and RNG.
- Invalid upper/lower gather bounds stop before original execution; retained CPU
  cases own their storage and remain savable without new tensor observation.
- Legal negative advanced indices, int32 index_select, scatter_add and prefix
  observation without extra CPU index copies.
- Original operator exceptions are retained.
- Actual reference classifier probabilities match exactly; model/optimizer
  states are unchanged.
- Actual evaluator success restores RNG and scoped hooks. On failure, the first
  B-forward error is preserved and RNG cleanup makes no subsequent state call.
- Only the declared launch-blocking protocol difference is allowed.
- Checkpoint2000 failure/prefix preflight rejects step1500 and missing journals.
- Launcher copies the artifact, starts a pinned worker with launch-blocking1,
  and records zero optimizer updates.
- Every inherited archive helper matches its source hash.

The initial evaluator failure test exposed that returning from a skipped RNG
restore would still allow later GPU integrity hashing. The observer now re-raises
the original exception at that boundary, and the test verifies exception identity.
No original source files or formal evaluator implementations were edited.
