# DamageFormer SRAC Joint Context V1

This isolated experiment tests one claim only:

> Does current-region context, and then Train-only historical Memory, improve a
> pixel classifier when the original A1 pixel evidence is never discarded?

It does **not** modify the old Transformer-SRAC experiment, retrain Q, change the
backbone, add a class-balanced loss, or use Test.

## Matched variants

All three decoders have identical parameters, identical initialization, see the
same image order, and share one frozen front forward.

| Variant | Pixel input | Region input | Historical input |
|---|---|---|---|
| J0 | frozen A1 `fused_roi` | zero | none |
| J1 | same | current token through learned control slots | none |
| J2 | same | current token through reader | immutable Train Memory neighbors |

The final classifier is algebraically one `Conv2d(256 + 128, 4, 1)`. It is
implemented as a 256-column pixel block plus a 128-column context block so the
original convolution is reproduced bit-for-bit. The pixel block and bias copy
A1 `clf_2`; the context block is zero. Thus before any optimizer update
J0/J1/J2 reproduce A1 exactly. This is a safe identity initialization, not a
post-hoc correction of an already produced prediction.

The reader returns an enhanced 128-D region token rather than four-class logits.
The same frozen sparse Q used for pooling unpools that token to the feature grid.
The unpooled context and A1 pixel feature are classified jointly.

V1 deliberately has no Memory-reliability gate. J1 and J2 first isolate whether
real Train neighbors outperform the same reader supplied with learned control
slots. A confidence gate is justified only if real Memory first shows useful but
selectively noisy evidence; adding it now would confound that comparison.

## Why full Train images are used

The inherited neighbor table indexes regions from deterministic 1024x1024 Train
images. Random crop/flip/rotation changes Q and region IDs, invalidating those
IDs. This experiment therefore uses full, unaugmented Train images with batch 1.
Changing to augmented crops requires a separate online-retrieval experiment.

## Required inherited artifacts

- a completed `damageformer_srac_transformer_v1/front/best.pth`;
- its exact original A1 checkpoint;
- its completed Train-only `train_memory` cache;
- its matching immutable neighbor table.

Smoke and formal artifacts must never be mixed.

## Commands

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python

bash bda_benchmark/experiments/damageformer_srac_joint_context_v1/run.sh preflight

export A1_CKPT=/path/to/original_A1.pth
export FRONT_CKPT=/path/to/completed/front/best.pth
export CACHE=/path/to/train_memory
export NEIGHBORS=/path/to/neighbors_global_cross_event
export RUN_ROOT=/a/new/path/joint_context_smoke

bash bda_benchmark/experiments/damageformer_srac_joint_context_v1/run.sh smoke
```

For the formal 5000-update matched comparison, point to formal artifacts and a
new output path:

```bash
export RUN_ROOT=/a/new/path/joint_context_formal_seed3407
nohup bash bda_benchmark/experiments/damageformer_srac_joint_context_v1/run.sh train \
  > /a/new/path/joint_context_formal_seed3407.log 2>&1 &
```

Interpret only these differences first:

- `J1 - J0`: incremental value of current-region context;
- `J2 - J1`: incremental value of real historical Memory;
- `J2 - A1`: whether the complete joint decision actually improves the baseline.

If J1 does not exceed J0, do not blame or expand Memory: current region context
has not shown complementary value. If J2 does not exceed J1, Memory retrieval or
reading has not shown value under the fixed front. Neither negative result alone
proves that Q or Memory can never work.
