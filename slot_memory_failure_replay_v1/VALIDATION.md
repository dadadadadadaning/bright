# Validation

Validated locally on 2026-10-20 using Python 3.10 and PyTorch 2.1.2+cpu.
The server's PyTorch 2.2.1+cu121 environment was not available locally.

All 44 tests passed:

- 15 diagnostic checkpoint tests: complete paired state and RNG roundtrip,
  exact subsequent stochastic update, inherited input ledger preservation,
  incomplete/malformed update rejection, tamper checks and overwrite protection.
- 5 observation tests using the original addon's `update_arms`: exact losses,
  gradient tensors, updated parameters, optimizer states and RNG; hook cleanup
  and retained CPU evidence after simulated backward failures.
- 10 restoration tests: independent storage, exact optimizer parameter groups
  and moments, complete counters, unchanged checkpoint payload and exact
  uninterrupted-versus-restored stochastic training.
- 12 historical trajectory tests: the original DataLoader shuffle convention,
  complete-prefix verification, file mutation detection, structural/LR failures
  and nonfatal finite metric drift.
- 2 driver integration tests: actual supervisor/worker functions, original
  updater and diagnostic helpers, with CPU runtime/data/process fixtures.
  A simulated second-arm candidate failure leaves only the complete prefix
  checkpoint; the subsequent dense worker reaches the same head, AdamW and RNG
  fingerprints as uninterrupted fixture training. Successful primary replay
  stops at its declared endpoint. Original fixture files retain content and mtime.

The integration tests use the original source `continuation.py` fingerprint and
learning-rate helpers through `BRIGHT_ABLATION_REFERENCE`. CUDA/process boundaries
are mocked in these two tests. CUDA assertions, real dataset throughput and the
server's historical allocator state are not validated by the CPU suite.

The standalone archive is built with a fixed explicit file list and deterministic
ZIP metadata. Its `--help` command works without importing Torch. The three
vendored prior diagnostic helpers match their preceding release byte for byte.
