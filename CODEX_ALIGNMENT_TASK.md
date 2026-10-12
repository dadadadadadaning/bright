# CODEX TASK: Alignment Experiments for BRIGHT BGCA

## Project Background

This repository is for BRIGHT building damage assessment.

Input:

* pre-disaster RGB
* post-disaster SAR

Baseline:

* DamageFormer

Current BGCA framework:

* `DamageFormer_BGCA.py`
* training script: `bda_benchmark/script/standard_ML/train_DamageFormer_BGCA_Method05_06.py`
* alignment losses: `bda_benchmark/model/modules/bgca_losses.py`

Current problem:
Old M06 used:

```text
dilated_soft ROI + damage-aware alignment lambda_align=0.001
```

It caused damaged-class collapse:

```text
best_model damaged IoU ≈ 0.00085
```

We suspect:

1. alignment weight is too strong;
2. alignment starts too early;
3. uniform building-region alignment suppresses real damaged features.

We need to implement three alignment experiment variants.

---

## Experiment M06A: Low-weight alignment

Definition:

```text
M06A = dilated_soft ROI + damage-aware alignment lambda_align=0.0001
```

No model/loss changes are required if the existing script already supports:

```bash
--use_align
--align_type damage_aware
--lambda_align
```

Required output:

* `scripts/train/run_M06A_dilatedROI_align_lam0001.sh`
* `scripts/eval/eval_M06A_checkpoints.sh`

Training core arguments:

```bash
--roi_mode dilated_soft
--roi_alpha 0.1
--roi_dilate 5
--detach_mask
--use_align
--align_type damage_aware
--lambda_align 0.0001
```

---

## Experiment M06C: Warmup alignment

Definition:

```text
M06C = dilated_soft ROI + damage-aware alignment with warmup
```

Add two training arguments:

```bash
--align_warmup_iters 10000
--align_ramp_iters 10000
```

Alignment scale should be:

```python
if current_iter < align_warmup_iters:
    align_scale = 0.0
elif current_iter < align_warmup_iters + align_ramp_iters:
    align_scale = (current_iter - align_warmup_iters) / max(1, align_ramp_iters)
else:
    align_scale = 1.0
```

Total loss should use:

```python
loss_total = loss_ce + loss_lovasz + lambda_align * align_scale * loss_align
```

Required output:

* Modify `train_DamageFormer_BGCA_Method05_06.py` minimally.
* Add CLI args:

  * `--align_warmup_iters`
  * `--align_ramp_iters`
* Log `align_scale` during training.
* Generate:

  * `scripts/train/run_M06C_dilatedROI_warmupAlign_lam0001.sh`
  * `scripts/eval/eval_M06C_checkpoints.sh`

Training core arguments:

```bash
--roi_mode dilated_soft
--roi_alpha 0.1
--roi_dilate 5
--detach_mask
--use_align
--align_type damage_aware
--lambda_align 0.0001
--align_warmup_iters 10000
--align_ramp_iters 10000
```

---

## Experiment M06D: Class-aware alignment

Definition:

```text
M06D = dilated_soft ROI + class-aware alignment
```

Motivation:
Do not pull RGB-SAR features close for all building pixels.

Desired behavior:

* intact regions: pull RGB and SAR features closer;
* damaged/destroyed regions: do not force close alignment; preserve damage-related difference.

Add a new loss function in:

```text
bda_benchmark/model/modules/bgca_losses.py
```

Suggested function name:

```python
class_aware_roi_alignment_loss(...)
```

Expected logic:

1. Resize damage label to feature resolution using nearest interpolation.
2. Use valid building labels only: 1 intact, 2 damaged, 3 destroyed.
3. Compute normalized projected features:

   * `z_rgb`
   * `z_sar`
4. Compute cosine similarity.
5. For intact pixels:

   * `loss_pull = 1 - cosine_similarity`
6. For damaged/destroyed pixels:

   * `loss_margin = relu(cosine_similarity - align_margin)`
7. Final loss:

   * `loss = pull_weight * loss_pull + margin_weight * loss_margin`
8. Weight the loss with ROI weights if available.
9. Return scalar loss and optional logging dictionary.

Add CLI args:

* `--align_margin`
* `--align_pull_weight`
* `--align_margin_weight`

Training core arguments:

```bash
--roi_mode dilated_soft
--roi_alpha 0.1
--roi_dilate 5
--detach_mask
--use_align
--align_type class_aware
--lambda_align 0.0001
--align_margin 0.3
--align_pull_weight 1.0
--align_margin_weight 0.2
--align_warmup_iters 10000
--align_ramp_iters 10000
```

Required output:

* Modify `bgca_losses.py`
* Modify `train_DamageFormer_BGCA_Method05_06.py` to dispatch `align_type == "class_aware"`
* Generate:

  * `scripts/train/run_M06D_dilatedROI_classAwareAlign_lam0001.sh`
  * `scripts/eval/eval_M06D_checkpoints.sh`

---

## General Requirements

Do not rewrite the whole project.

Make minimal changes.

Do not change dataset paths.

Do not change the model head or label mapping.

Keep existing behavior unchanged when:

* `--use_align` is not set
* `--align_type damage_aware`
* `--align_warmup_iters 0`
* `--align_ramp_iters 0`

Add a quick compile check:

```bash
python -m py_compile bda_benchmark/model/modules/bgca_losses.py
python -m py_compile bda_benchmark/script/standard_ML/train_DamageFormer_BGCA_Method05_06.py
```

Add or update documentation:

```text
docs/ALIGNMENT_EXPERIMENT_PLAN.md
```

Please output:

1. changed files list;
2. git diff;
3. how to run M06A/M06C/M06D;
4. any assumptions.
