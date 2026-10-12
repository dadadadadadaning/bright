# BRIGHT Memory key comparison package

Standalone delivery of the new `damageformer_srac_memory_key_compare_v1` experiment.

- Download `bright_memory_key_compare_v1.zip` and its `.sha256` file.
- Only the new experiment directory is installed; original models, source and results remain unchanged.
- Stage 1: same-member T128/F256 retrieval and soft-label diagnostics.
- Stage 2: matched query-only / T128 / F256 bounded pixel corrections of frozen grid_probq.
- Validation: 56 CPU unit/integration tests and 9 installer tests passed. Real CUDA / BRIGHT Val450 not run in the delivery environment.

Upload both files to `/root/shared-nvme/`. Verify and install:

```bash
cd /root/shared-nvme
if sha256sum -c bright_memory_key_compare_v1.zip.sha256; then
  /root/shared-nvme/conda-envs/bright4090/bin/python -c 'import zipfile; exec(compile(zipfile.ZipFile("/root/shared-nvme/bright_memory_key_compare_v1.zip").read("INSTALL_SERVER.py"), "INSTALL_SERVER.py", "exec"))'
fi
```

Read the installed experiment's README for the exact staged commands. Start with CPU preflight; wait for existing GPU jobs to finish before build/diagnose. The passed actual fresh E0 is required. No stage runs merely by installing.

Package SHA256: `fee3cad5e764dbd4cd21867709f78c4cdc69550424da08066d13f1d0feb67b33`

## Training source

- [Train matched correction heads](bda_benchmark/experiments/damageformer_srac_memory_key_compare_v1/train_correction.py)
- [Bounded pixel correction model](bda_benchmark/experiments/damageformer_srac_memory_key_compare_v1/correction.py)
- [Independent Val evaluator](bda_benchmark/experiments/damageformer_srac_memory_key_compare_v1/evaluate_correction.py)
- [Full server instructions](bda_benchmark/experiments/damageformer_srac_memory_key_compare_v1/README.md)

These source files exactly match the verified ZIP. Keep the same KEY_OUTPUT_ROOT used by the completed first-stage full diagnostic; do not create a new root for training. Formal fitting requires that diagnostic and the passed fresh E0. The 20-step smoke and formal5000 fit start from separate fresh head initializations.

## Shared-filesystem installer compatibility

The installer now handles filesystems that reject renameat2(RENAME_NOREPLACE): it exclusively creates a new directory and files, verifies their hashes, and preserves partial installs on failure. Existing files are never overwritten. Experiment payload bytes are unchanged. 18 installer tests passed, including a simulated EINVAL installation of the real ZIP.

Users with the original ZIP (SHA256 39d1cd66ff604d022d0384479834bcbf1d4aec2a586ede08be5a9f1042ae8b1d) may use INSTALL_COMPAT_OLD_ZIP.py without downloading another package. It accepts only that hash-pinned original ZIP. Installation does not run Train, Val or Test.
