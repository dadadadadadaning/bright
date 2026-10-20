# Validation

Validated on 2026-10-20 using Python 3.10 and PyTorch 2.1.2+cpu.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 BRIGHT_ABLATION_REFERENCE=/workspace/new_submission_review_20261019/New/joint_anchor_fresh_chain_v2 /workspace/bright-audit-legacy-env/bin/python -m unittest discover -s slot_memory_route_probe_v1 -p 'test_*.py' -v
```

Result: **34 tests passed**.

- 13 preflight/output tests use the real manifest/journal/checkpoint SHA
  validators, with only the source bridge mocked. They verify exact replay-bound
  parent selection, changed inputs, failure context, path/symlink protection,
  collision refusal and unchanged input bytes/mtimes.
- 8 route-observer tests verify both actual function-global binding locations,
  one original call per observation, return/exception identity, scoped cleanup,
  negative/sum/nonfinite diagnostics, `.data` mutation detection, input/RNG
  preservation and saving retained CPU evidence after observation ends.
- 11 storage tests cover independent tensors, offset views, shared allocations
  with disjoint byte spans, strided holes, external shared memory, device
  separation, parameter/gradient/optimizer ownership, and metadata-only access.
- 2 integration tests use the original six-arm updater, actual reference
  `validate_routes`/`sparse_unpool`, and full AdamW/RNG restoration. The success
  path matches an independent unobserved update exactly in head parameters,
  optimizer state, metrics and RNG. A synthetic modification after F causes the
  actual B validator to raise; E/F updates, unchanged remaining states, CPU
  evidence, hook cleanup and absence of saved training checkpoints are checked.

The integration fixture performs three small real warm-up updates, then assigns
synthetic optimizer counters of 1000 to test the 1000/1001/1303 distinction. It
does not claim a 1000-step training reproduction. Its dataset, CUDA availability
boundary and runtime construction are synthetic.

The six reused modules (`probe`, `inspect_failure`, `restore_state`, `trajectory`,
`observer`, `nll_trace`) were checked byte-for-byte against the previously
published replay tool. All new Python sources compile. The deterministic
standalone archive is checked with `--help` and a missing-evidence invocation
that must stop before creating output.

No GPU or BRIGHT dataset is available locally. These checks validate diagnostic
behavior and isolation; they do not reproduce or fix the server CUDA failure.
