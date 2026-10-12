# SALT-CDE validation v1

This experiment is isolated from P3/P5 and never reads the official test list.

## Controlled variants

| Variant | Change |
|---|---|
| D0 | public DamageFormer-style pseudo-RGB control |
| D0N | native one-channel SAR with colocated fusion |
| D1 | D0N + explicit local correlation/transport |
| D2 | D1 + offsets propagated from the predicted outer building ring |
| D3 | D2 + counterfactual reliability gate |
| D2R | D1 + confidence-weighted bounded ring residual |
| D3V2 | D2R + global pair evidence, explicit correct/wrong gate labels, and RGB-only rejection fallback |

D0N is necessary: D1-D0 would otherwise mix the effect of native-SAR encoding
with the effect of transport.

D1/D2 use no extra base-head loss. D3 adds a `0.25` colocated-base auxiliary
loss because a trained fallback is part of the declared counterfactual method.

## Training budget

All stages use AdamW, LR `1e-4`, weight decay `5e-3`, CE + Lovasz, and EBS 16.

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash experiments/salt_cde_validation_v1/run.sh preflight

for variant in d0 d0n d1 d2 d3; do
  bash experiments/salt_cde_validation_v1/run.sh smoke "$variant"
done

# Optional cheap collapse screen: 2k optimizer steps / 32k crops.
for variant in d0 d0n d1 d2 d3; do
  bash experiments/salt_cde_validation_v1/run.sh probe "$variant"
done

# Main screen: continues from 2k, reaches 8k / 128k crops.
for variant in d0 d0n d1 d2 d3; do
  bash experiments/salt_cde_validation_v1/run.sh pilot "$variant"
done
```

Do not reject a close method at 2k.  Use 8k to rank models, take the best two
to `short` (20k / 320k crops), and run only the selected method plus its exact
control with `final` (50k / 800k crops).

The safer sequential launcher creates its own logs and stops after the first
failure:

```bash
nohup bash experiments/salt_cde_validation_v1/run_sequential.sh pilot \
  > /root/data/BRIGHT-train-logs/salt_pilot8000_launcher.log 2>&1 &
```

It automatically continues a variant from `probe2000` when that checkpoint
exists; otherwise that variant starts fresh and runs directly to 8000 steps.

## Mechanism audit

```bash
for variant in d0 d0n d1 d2 d3; do
  TRAIN_TAG=pilot8000 bash experiments/salt_cde_validation_v1/run.sh audit "$variant"
  TRAIN_TAG=pilot8000 bash experiments/salt_cde_validation_v1/run.sh curve "$variant"
done

TRAIN_TAG=pilot8000 bash experiments/salt_cde_validation_v1/run.sh compare d0n_d0
TRAIN_TAG=pilot8000 bash experiments/salt_cde_validation_v1/run.sh compare d1_d0n
TRAIN_TAG=pilot8000 bash experiments/salt_cde_validation_v1/run.sh compare d2_d1
TRAIN_TAG=pilot8000 bash experiments/salt_cde_validation_v1/run.sh compare d3_d2
```

`d1_d0n` asks whether explicit local transport helps. `d2_d1` asks whether
stable-ring transport improves x8 robustness. `d3_d2` asks whether the
counterfactual gate reduces damage from a wrong SAR pair.

The completed D2/D3 runs are retained as negative controls.  The evidence-led
repair is screened separately and must start from fresh weights at the same 8k
budget:

```bash
FRESH_STAGE=1 bash experiments/salt_cde_validation_v1/run.sh smoke d2r
FRESH_STAGE=1 bash experiments/salt_cde_validation_v1/run.sh smoke d3v2

FRESH_STAGE=1 bash experiments/salt_cde_validation_v1/run.sh pilot d2r
TRAIN_TAG=pilot8000 bash experiments/salt_cde_validation_v1/run.sh audit d2r
TRAIN_TAG=pilot8000 bash experiments/salt_cde_validation_v1/run.sh compare d2r_d1

# Run this only if D2R is competitive and its x8 curve is no worse than D1.
FRESH_STAGE=1 bash experiments/salt_cde_validation_v1/run.sh pilot d3v2
TRAIN_TAG=pilot8000 bash experiments/salt_cde_validation_v1/run.sh audit d3v2
TRAIN_TAG=pilot8000 bash experiments/salt_cde_validation_v1/run.sh compare d3v2_d2r
```

D2R begins at about 1.8% ring-residual strength and uses only confident ring
matches, so an unreliable propagated offset cannot immediately replace D1.
D3V2 sees both local and building-pooled pair evidence; its correct pair is
labelled gate-on and its shuffled pair gate-off.  Unlike D3, gate-off removes
SAR from the fusion rather than merely removing transport.  Its RGB-only
fallback is trained by the base auxiliary loss.  The counterfactual weight is
reduced to 0.10 to limit the clean-accuracy cost observed in D3.
The pair-shuffle mechanism gate requires an absolute mean gate reduction of at
least 0.05, rather than accepting a numerically positive but negligible change.

For 20k/50k audits set `TRAIN_TAG=short20000` or `TRAIN_TAG=final50000`.

## Hardware arithmetic

The launcher is single-process and uses batch 4 x accumulation 4 = EBS 16.
On two GPUs, use DDP only after adding and validating a distributed launcher;
simply exposing two devices does not double the batch safely.
