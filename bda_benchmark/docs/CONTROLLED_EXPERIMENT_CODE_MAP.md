# Controlled experiment code map

This document is the canonical routing map for the active BRIGHT RGB--SAR
experiments.  Historical model files are intentionally not moved or renamed:
their paths are part of old checkpoint and launcher contracts.  New work must
extend one declared family instead of copying another trainer and inventing a
new variant tuple.

## 1. Active main line: function-preserving native SAR

| Responsibility | Canonical file |
|---|---|
| Variant specification and checkpoint migration | `model/DamageFormer_S8_P4_NativeSAR_MICB.py` |
| Radiometric input contract | `dataset/make_data_loader_p4_native_sar.py` |
| Train-only statistics | `script/standard_ML/compute_S8_P4_sar_stats.py` |
| Training | `script/standard_ML/train_S8_P4_native_sar_micb.py` |
| Independent evaluation | `script/standard_ML/eval_S8_P4_native_sar_micb.py` |
| One-GPU launcher | `run/run_S8_P4_NSMICB_5090.sh` |
| Controlled sequence | `run/run_S8_P4_NSMICB_sequence.sh` |
| Tests | `tests/test_s8_p4_native_sar.py` |

Declared variants:

- `a0`: exact historical P3 pseudo-RGB control.
- `a1`: legacy train-stat normalisation repeated to three channels.
- `a2`: legacy one-channel patch embed with channel-sum migration.  This is
  exact only for identical already-normalised channels and is **not** the
  corrected function-preserving control.
- `a3`: legacy new convolutional stem.
- `a4`: corrected one-channel affine input plus weight-and-bias patch-embed
  fold.  No clipping/nonlinear transform is allowed before its patch embed.

Required order:

1. `a0` guard.
2. `a4` exact guard at zero optimisation steps.
3. `a4` smoke.
4. Only after both guards pass, run controlled SAR adaptation.

Do not compare `a2` and `a4` as if they had the same input contract.

## 2. Structural experiments: P3-SRAC controlled family

| Responsibility | Canonical file |
|---|---|
| Model/partition/affinity/memory | `model/DamageFormer_P3_SRAC_Controlled.py` |
| Structural objectives | `script/standard_ML/p3_srac_controlled_runtime.py` |
| Training wrapper | `script/standard_ML/train_P3_SRAC_Controlled.py` |
| Launcher | `run/run_P3_SRAC_controlled_5090.sh` |
| Tests | `tests/test_p3_srac_controlled.py` |

Checkpoint selection is task specific:

- `sr1p`, `sr1pb`, `sr1pib`: include exact P3 step 0 and select by event risk.
- `sr1pa`, `sr1pai`: select `best_partition.pth` using Boundary F1, Building
  IoU, background leakage and partition entropy.
- `sr1paa`: select `best_affinity.pth` using balanced affinity accuracy and
  damage-boundary recall.
- `sr2`: must not run until its input partition/region representation passes a
  validation-only structural gate.

Partition-only variants must never be selected by unchanged P3 mIoU/event
metrics.  Validation and test samples must never update the retrieval memory.

## 3. Canonical variant registry

Public controlled variant names live in:

`configs/experiment_registry.py`

Models, datasets and command-line wrappers must import this registry.  A new
variant is incomplete until implementation dictionaries and dataset support
match the canonical tuple and the registry consistency tests pass.

## 4. Historical evidence families (do not silently reuse as new ideas)

| Function | Existing families |
|---|---|
| Prototype/memory retrieval | BGMP, SIRM, XHPI, CEAM, SR2 |
| Region/instance pooling | DARM, H5-BIQ, Q1, Q2, H13, P3-SRAC |
| Registration/correspondence | IARM, H4 flow, CCU, H10, BIOS |
| Native-SAR encoders | Heterogeneous H1--H3, P4, P5, H6 |
| Pretrained SAR transfer | H11 SARMAE, P3-RESET SAR-DINO probes |
| Safe output/feature residuals | H6B, H7, H8, H12--H14, OA-HMD, Q2, SR1P |
| Full predictor rebuilds | ExternalCD, OA-HSDNet |

A failure in a safe residual rejects that feature/permission/attachment
combination; it does not reject the complete research direction.  A failure
in a full rebuild also includes loss of P3's mature fusion and decision head.

## 5. Non-negotiable protocol invariants

1. Test is not used for model selection or threshold tuning.
2. Step 0 competes whenever a new branch is exactly zero/safe initialised.
3. Partition pretraining uses structural metrics, not unchanged segmentation
   metrics, for checkpoint selection.
4. Checkpoint `variant`, source hash, normalisation contract and completed step
   are authoritative; experiment directory names are not.
5. Smoke/partial validation metrics are never compared with full validation.
6. A radiometric transform change requires a mathematical weight/bias
   conversion or a new explicitly non-equivalent variant.
7. Region/retrieval experiments remain paused until the upstream native-SAR
   representation and structural unit each pass their own validation gate.

## 6. Image-matching evidence transfer (IM4CD-RM)

This family is a validation-only extension of RM, not another P3 residual and
not a full reproduction of IM4CD.

| Responsibility | Canonical file |
|---|---|
| Hard local response peak and dominant-translation residual | `script/standard_ML/im4cd_matching_runtime.py` |
| Optional cache export | `script/standard_ML/build_RM0_region_cache.py --matching_stats` |
| Co-located-vs-matching audit | `script/standard_ML/audit_IM4CD_region_matching.py` |
| Launcher | `run/run_IM4CD_RM_matching.sh` |
| Tests | `tests/test_im4cd_region_matching.py` |

The historical CCU family already tested soft local matching and learnable
pixel residuals. IM4CD-RM is a distinct experiment only because it tests hard
peak geometry after removal of a scene-dominant translation and uses the
result solely as region-memory evidence. Failure must stop before a pixel
residual, CRF, superpixel graph or trained retrieval classifier is added.
