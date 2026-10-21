# Checkpoint2000 synchronous index diagnostic

The recorded six-arm continuation reached checkpoint2000, then failed at Val
index30 (`beirut-explosion_00000016`) while validating B probabilities. The CUDA
log reports ScatterGatherKernel index bounds assertions. All53421 observed route
validators returned; no route failure artifact was generated. The probability
finite check is a synchronization/reporting site, not proof that it caused the
index error. Subsequent CUDA RNG restoration failed during unwinding.

This executable diagnoses that failure; it does not resume training or repair
probabilities, routes, indices, model code, labels, or numerical tolerances.

## Execution

```text
bright_index_eval2000_v1.pyz start
bright_index_eval2000_v1.pyz logs
bright_index_eval2000_v1.pyz status
```

- Preflight pins the recorded parent trainer, failure image/arm, complete30-image
  journal, six-arm lineage, and checkpoint2000 sidecar/hash.
- Worker verifies the checkpoint payload, all six optimizer counters/moments,
  journal prefix, and inherited input hashes. It restores the saved state in a
  fresh process and evaluates the first31 Val images through all six arms.
- CUDA_LAUNCH_BLOCKING=1 is set in the child environment before Torch/CUDA use.
  Runtime provenance records original and diagnostic settings separately. Every
  other protocol setting must match. This execution is diagnostic only.
- TorchDispatchMode records index operations. On the target image, integer
  gather/scatter/index_select and advanced-index inputs are copied to CPU before
  execution. Zero-stride expanded axes are compacted without changing values.
  Advanced indexing allows its normal negative indices; gather/scatter do not.
  Boolean indexing is left to PyTorch's original checks.
- Metadata includes source/index shape, strides, allowed bounds, extrema, active
  image/arm, and Python callsites. Small B-source tensors are also copied for a
  potential failing CPU case. Copies are observations, never operator inputs.
- If a CPU bounds check finds an illegal index, execution stops before that
  original index operation. Otherwise the original operation runs unchanged.
- The first exception is recorded immediately. On failure the evaluator's RNG
  cleanup re-raises that exception before additional CUDA RNG/hash operations.
  Failed diagnostics do not claim state-restoration or successful integrity.
- Outputs use a new directory. Existing parent/checkpoint/data/source files stay
  unchanged. Existing training/evaluation locks prevent concurrent formal work.
  No optimizer update, Test access, metric promotion, skipped failing image,
  automatic fallback or retry is performed.

## Outputs

`summary.json`, `manifest.json`, `diagnostic.log`, `index_summary.json`,
`target_indices.jsonl`; on failure `first_failure.json`, `traceback.txt`, and
`index_failure_cpu.pt` when a pre-operation CPU case is available. `logs/status`
from the pinned output probe refer to its own directory; `/tmp` copies use the
latest index diagnostic pointer.

`diagnostic_complete=true` means the31-image diagnostic and its unchanged-state
checks passed. `complete` and `full_validation_evidence` remain false. A pass is
not proof of a root-cause fix: synchronous execution and CPU observation change
timing, and a fresh worker does not reproduce the old live-process history.

## Local validation

See VALIDATION.md. The published archive inherits the recorded continuation's
helpers byte-for-byte, with the ledger in INDEX_PROBE_BASE.json. There is no GPU
or BRIGHT dataset in the local test environment.
