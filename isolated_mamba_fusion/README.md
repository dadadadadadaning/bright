# isolated_mamba_fusion

`isolated_mamba_fusion` 是一个独立验证目录，用于 BRIGHT 灾前 RGB + 灾后 SAR 建筑损毁分级任务中的特征融合原型。它不 import、不修改原始 `BRIGHT-master` / `DamageFormer` 工程文件，也不包含训练、数据集或 checkpoint 逻辑。

目标是先独立验证模块的 forward、shape、NaN/backward 稳定性，后续再手动接入原始模型。

## 模块列表

### 1. BaseFusion

只做基础跨模态投影：

```text
F_base = Conv1x1(concat(F_rgb, F_sar))
```

不做 diff、不做 mul、不做 spatial mixer、不做 gate。它用于排除“只是多加参数导致提升”的可能。

### 2. DiffMambaFusion

显式构建 RGB/SAR 差异特征：

```text
Z_rgb = Norm(Proj_rgb(F_rgb))
Z_sar = Norm(Proj_sar(F_sar))
F_diff = |Z_rgb - Z_sar|
F_mul = Z_rgb * Z_sar
F_cross = Conv([Z_rgb, Z_sar, F_diff, F_mul])
F_m = SpatialMixer(F_cross)
F_base = base_mode(F_rgb, F_sar)
F_out = F_base + scale * F_m
```

### 3. ExistGuidedMambaFusion

在 DiffMambaFusion 的差异增强特征上加入损毁存在性 gate 与可选 ROI 约束：

```text
G = sigmoid(G_exist)                         # [B,1,H,W]
或
G = softmax(G_exist, dim=1)[:, 1:2, :, :]    # [B,2,H,W]

G = gate_min + (1 - gate_min) * G
G_final = G * roi_mask + bg_alpha * (1 - roi_mask)
F_out = F_base + scale * G_final * F_m
```

如果 `G_exist is None`，则 `G=1.0`，退化为不使用损毁存在性 gate 的差异融合。

## 输入输出

```text
F_rgb:   Tensor [B, C, H, W]
F_sar:   Tensor [B, C, H, W]
G_exist: None / Tensor [B, 1, H, W] / Tensor [B, 2, H, W]
roi_mask: None / Tensor [B, 1, H, W]
F_out:   Tensor [B, C, H, W]
```

如果 `G_exist` 或 `roi_mask` 的空间尺寸与特征尺寸不同，会自动插值到 `[H, W]`：

```python
F.interpolate(..., mode="bilinear", align_corners=False)
```

## bounded per-channel gamma

`DiffMambaFusion` 和 `ExistGuidedMambaFusion` 使用 per-channel gamma：

```text
gamma: [1, C, 1, 1]
scale = gamma_scale * tanh(gamma)
```

默认：

```text
gamma_init = 0.0
gamma_scale = 0.1
```

`gamma_init` 表示希望初始化得到的 effective scale。代码内部会使用：

```text
raw_gamma = atanh(gamma_init / gamma_scale)
```

因此当 `gamma_init=0.05, gamma_scale=0.1` 时，初始 effective scale 约为 `0.05`。

限制 gamma 的原因是：避免 RGB-SAR 差异分支在训练早期过强，扰乱原始特征和已有 damage head。

## G_exist 处理规则

`ExistGuidedMambaFusion` 支持：

```text
G_exist = None
G_exist = [B,1,H,W]
G_exist = [B,2,H,W]
G_exist 空间尺寸与 feature 不一致
```

规则：

```text
None       → G = 1.0
[B,1,H,W] → G = sigmoid(G_exist)
[B,2,H,W] → G = softmax(G_exist, dim=1)[:, 1:2]
```

然后：

```text
G = gate_min + (1 - gate_min) * G
```

`gate_min` 默认 `0.1`，用于避免 early-stage `G_exist` 不准时完全压制真实损毁差异。

`detach_gate=True` 时：

```python
G = G.detach()
```

这用于避免主 damage loss 的梯度反向扰乱 exist head。

## ROI 约束

`roi_mask` 表示建筑 ROI 或建筑概率图：

```text
roi_mask: [B,1,H,W]
```

如果传入 ROI：

```text
roi_mask = resize 到 [H,W]
roi_mask = clamp 到 [0,1]
G_final = G * roi_mask + bg_alpha * (1 - roi_mask)
```

默认：

```text
bg_alpha = 0.05
```

BRIGHT 是建筑损毁分级任务，差异增强应主要发生在建筑区域内；背景区域只保留弱差异增强，以减少 RGB-SAR 背景模态差异干扰。

如果没有 `G_exist` 但有 `roi_mask`：

```text
G = 1.0
G_final = roi_mask + bg_alpha * (1 - roi_mask)
```

## enable 开关

`DiffMambaFusion` 和 `ExistGuidedMambaFusion` 支持：

```python
enable=False
```

此时不执行 diff/mul/spatial mixer/gate，直接返回 base feature：

```text
base_mode="rgb"  → F_rgb
base_mode="sar"  → F_sar
base_mode="avg"  → 0.5 * (F_rgb + F_sar)
base_mode="conv" → Conv1x1(concat(F_rgb, F_sar))
```

这个开关用于后续接入 DamageFormer 后恢复原始逻辑，便于 ablation 和 debug。

## gate_warmup

`ExistGuidedMambaFusion` 预留：

```python
gate_warmup: int = 0
forward(..., cur_iter: Optional[int] = None)
```

逻辑：

```text
gate_warmup <= 0 或 cur_iter is None → 正常使用 G_exist gate
cur_iter < gate_warmup               → 不使用 G_exist gate，令 G=1.0
cur_iter >= gate_warmup              → 正常使用 G_exist gate
```

即使 warmup 阶段不使用 `G_exist`，如果传入 `roi_mask`，仍然会使用 ROI 约束。

这个功能用于后续接入训练循环时，在前若干 iter 避免 early-stage exist prediction 不准导致主特征被错误抑制。

## VMamba / VSSBlock 兼容

`LightweightVSSAdapter` 会尝试寻找：

```text
vmamba.VSSBlock
changedetection.models.vmamba.VSSBlock
```

如果没有 VMamba / VSS / selective_scan，或者 VSSBlock 初始化/forward 失败，会自动使用 fallback block：

```text
Conv + BN/GN + GELU + depthwise conv + residual
```

如果最终论文实验没有使用真正 VMamba/VSS/selective_scan，不要写成严格 Mamba，只能写：

```text
Mamba-style spatial fusion
或
lightweight spatial mixing
```

## 运行测试

在 `isolated_mamba_fusion` 目录下：

```bash
python tests/test_shapes.py
python tests/test_backward_nan.py
python demo/demo_random_forward.py
```

从项目根目录：

```bash
python isolated_mamba_fusion/tests/test_shapes.py
python isolated_mamba_fusion/tests/test_backward_nan.py
python isolated_mamba_fusion/demo/demo_random_forward.py
```

可选 CPU forward/backward smoke test：

```bash
python isolated_mamba_fusion/tests/test_forward_cpu.py
```

## 推荐消融顺序

```text
E1 baseline
E1 + BaseFusion
E1 + DiffFusion
E1 + ExistGuidedDiffFusion detach_gate=True
E1 + ExistGuidedDiffFusion detach_gate=True + roi_mask
```

重点观察：

```text
target3
DmgIoU
DesIoU
Dmg→Int
Des→Int
IntactIoU
```

## 接入建议

不要直接修改原始 DamageFormer 工程。建议后续在原工程中新建独立模块文件，例如：

```text
bda_benchmark/model/modules/mamba_fusion.py
```

再手动迁移这里的模块，并通过参数开关默认关闭：

```text
--use_base_fusion
--use_diff_mamba_fusion
--use_exist_guided_mamba_fusion
```

这样可以保证旧实验、旧 checkpoint 和已有评估流程不被破坏。
