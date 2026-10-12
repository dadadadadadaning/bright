# Frozen A1 state probe for Q factorial training

This **diagnostic-only** wrapper runs the existing Q trainer without changing
its hashed source files. It records the first A1 `state_dict` key that differs
from the strict A1 checkpoint after the front adapter is loaded, before and
after initial Val, before and after each optimizer update, and before and after
final Val. Tensor bytes are compared exactly. A cheap tensor version check runs
on every optimizer update; a full byte comparison runs at step 1 and every
`A1_PROBE_FULL_EVERY` updates. A `.data` mutation that does not increment a
version counter can be localized only to that full-scan interval.

The probe intentionally interrupts at the final A1 guard, before the original
trainer can certify a completed checkpoint. It may write an **incomplete**
`wrapped_train/last.pth`; never pass this to formal evaluation. A probe result is **never**
a formal model score. The original failed report and its `1e-5` reproduction
threshold remain unchanged. The wrapper reuses the original E0 input,
checkpoint, runtime and split gates. Only Train and Val are read; Test is not.

Run from the BRIGHT repository root in a fresh shell using the same GPU-visible
device and legacy/native numeric settings as the passed E0:

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export E0_PROTOCOL=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130/protocol
unset MATMUL_TF32 CUDNN_TF32 DETERMINISTIC CUDA_VISIBLE_DEVICES

bash bda_benchmark/experiments/damageformer_srac_a1_state_probe_v1/run.sh preflight

export A1_PROBE_OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_A1_STATE_PROBE_V1/smoke_$(date +%Y%m%d_%H%M%S)
bash bda_benchmark/experiments/damageformer_srac_a1_state_probe_v1/run.sh smoke
```

The two-update smoke test checks whether the mutation appears immediately.
If it reports `finished_without_detected_A1_change`, perform the 2000-update
replay (about as costly as the original run, including full Val450):

```bash
export A1_PROBE_OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_A1_STATE_PROBE_V1/formal_replay_$(date +%Y%m%d_%H%M%S)
bash bda_benchmark/experiments/damageformer_srac_a1_state_probe_v1/run.sh formal_replay
```

Read `$A1_PROBE_OUTPUT/summary.json`. `first_change.phase` tells when a mutation
was first observed; `first_change.changed_state` lists exact keys, whether each
is a parameter or buffer, and its byte hashes. `state_trace.jsonl` records
phase checks. `wrapped_train/summary.json` is an intentionally incomplete
inner diagnostic run and must not be passed to formal evaluation. If status is
`trainer_error`, inspect `exception.txt`; it may indicate an environment or
input guard failed before the state check. A clean probe is evidence only that
this instrumented replay did not reproduce the mutation. A probe `complete=true`
means the diagnostic reached its final guard, not that training formally passed.
If a tensor changes through `.data` without a version bump, the first observed
step is only localized to the preceding full-scan interval. Rerun with
`A1_PROBE_FULL_EVERY=1` to identify the exact optimizer step if that distinction
matters; this incurs more CPU transfer and runtime.
