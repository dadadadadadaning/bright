# Validation scope

Tests run locally on Python 3.10 with PyTorch 2.1.2+cpu and NumPy. No server CUDA,
BRIGHT image data, production weights or training-quality validation is available
in this development environment.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
BRIGHT_ABLATION_REFERENCE=/path/to/joint_anchor_fresh_chain_v2 \
python -m unittest discover -s slot_memory_resume_v1 -p 'test_*.py' -v
```

The suite contains 40 tests:

| Area | Tests | Evidence |
| --- | ---: | --- |
| Full checkpoints | 10 | Model/AdamW/RNG roundtrip and uninterrupted continuation equivalence; partial counters, nonfinite states, journal tampering, diagnostic formats and overwrites rejected |
| Evaluation evidence | 14 | Exact Val450 identities, complete expected schedule, persisted checksums, loss/metric/transition consistency, fresh paths, endpoint-only validation recovery |
| Background launcher | 2 | Actual subprocess executes its copied archive, keeps the inherited lock after launcher return, blocks duplicate launch and releases the lock at exit; original files remain unchanged |
| Resume preflight | 11 | Original checkpoint1000 selection, exact splits/fixed sample, stable failed journal, lineage, parent locks and diagnostic-checkpoint rejection |
| Worker integration | 3 | Original six-arm updater with synthetic CPU heads/data, real restoration/checkpoint serialization, success, partial B failure and subsequent restart, and validation-only restart from an endpoint checkpoint |

Integration fixtures retain real six-arm updates and optimizer handling but replace
the production dataset/runtime, CUDA availability and evaluation computation.
Synthetic checkpoints use counters at 1000 after a short warmup; these tests do
not claim to train 1000 real BRIGHT updates.

The failure test injects a B-backward exception after E/F update at step1003.
Only step1002 becomes a recoverable checkpoint; fresh continuation from it matches
the uninterrupted synthetic reference. The endpoint test fails validation after
saving step1004, then verifies a new worker performs zero further optimizer updates,
restores the same model/AdamW/RNG state and completes both endpoint evaluations.
Reports are promoted only after the mocked runtime has closed. Prior source and
failed-output contents and modification times are checked for preservation.

Separately checked the validator against the actual original evaluator's output
schema on 450 synthetic CPU images. Only a copy of the smoke report had its
CUDA/formal-status guards adjusted for schema validation. The original CPU report
stayed smoke and was not promoted to formal evidence.

Packaging checks verify all 12 archive entries, Python source compilation,
deterministic rebuilt bytes, standalone `--help`, and missing-input rejection
before creating an output directory. The archive contains no tests, datasets,
weights or experiment logs.

The successful isolated historical-image route probe and these CPU tests do not
resolve the historical long-run route/NLL CUDA failure. Production continuation
must be assessed from its own server log, complete checkpoints and fresh Val450
reports; any failure stops the worker without skipping or retrying that update.
