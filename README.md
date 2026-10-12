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

Package SHA256: `39d1cd66ff604d022d0384479834bcbf1d4aec2a586ede08be5a9f1042ae8b1d`
