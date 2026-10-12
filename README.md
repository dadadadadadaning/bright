## Follow-up: support token replay failure

The V2 build stopped on bata-explosion_00000000: 21 of 256 selected coordinates were outside the archived FP16 rounding interval plus the fixed tolerance. This is not explained by FP16 rounding alone; it does not evaluate Memory usefulness. The V2 ZIP and experiment remain unchanged.

Use the [standalone replay audit](diagnostics/memory_replay_v2/README.md) to check bank/shard/evidence identity and compare repeated fixed-input native forwards across inference_mode, no_grad and guarded no_grad. No training, bank replacement, runtime flag changes, Val or Test. 28 synthetic CPU tests and the actual standalone CPU CLI passed; real CUDA and server data have not been run by the delivery environment.

# BRIGHT Memory key comparison V2

Download [bright_memory_key_compare_v2.zip](bright_memory_key_compare_v2.zip) and [its SHA256 file](bright_memory_key_compare_v2.zip.sha256).

V2 is a separate experiment; V1 code and failed outputs remain unchanged. It retains actual FP32 replay tokens and checks compatibility with archived FP16 rounding intervals at the fixed 2e-5 absolute/relative tolerance. FP32 archives keep direct comparison. Reports distinguish definitely within tolerance, storage uncertainty and definite drift. Storage compatibility does not establish exact historical FP32 reproduction. Original bank rows/T128/labels and the independent Val450 absolute 1e-5 gate remain unchanged.

- 77 CPU unit/integration tests and five CLI entry checks passed.
- 18 installer tests passed; the actual ZIP passed a simulated EINVAL shared-filesystem install and read-only repeat.
- Real CUDA / BRIGHT images and Val450 were not run in the delivery environment.
- [Server installation and staged instructions](bda_benchmark/experiments/damageformer_srac_memory_key_compare_v2/README.md)
- [Saved replay evidence checker](bda_benchmark/experiments/damageformer_srac_memory_key_compare_v2/recheck_replay.py)
- [Matched correction training source](bda_benchmark/experiments/damageformer_srac_memory_key_compare_v2/train_correction.py)

V2 ZIP SHA256: `4557ff9e3a533eb26ad030cd9843c33ee48448191f39c114b10609be2edc077e`.

Upload the V2 archive to `/root/shared-nvme/`. Start with preflight and build_smoke using a fresh V2 output root and the actual passed E0. Do not begin formal training before the full retrieval diagnostic passes.

---

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
