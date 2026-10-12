# DamageFormer ROI + Cross-modal Damage-aware Soft Partition V1

## 这次只做两个顶层模块

```text
Pre-event RGB
      ↓
RGB Encoder
      ↓
clf_1
      ↓
P_building
      ↓
[模块1] A1 Soft Building ROI
      ↓
[模块2] Cross-modal Damage-aware Soft Partition
      ↓
zero-init region residual
      ↓
原 DamageFormer clf_2
      ↓
Background / Intact / Damaged / Destroyed
```

不加入 Building Reasoning、Graph、MIL、Severity、Memory、Morphology。
主方法也不使用 SLIC/SLICO。

## LCA 为什么不算第三个顶层模块

LCA 被视为 Partition 内部的“可靠跨模态关系构造步骤”：

```text
RGB feature
       \
        local 3×3 RGB→SAR match
       /
SAR feature
       ↓
aligned SAR + alignment confidence
       ↓
Partition
```

论文顶层仍然只有 ROI + Soft Partition。

## ROI

ROI 来自 RGB 建筑定位头：

```python
loc_logits_feature = clf_1(rgb_feature)
p_building = softmax(loc_logits_feature)[:, 1:2]
roi_weight = 0.1 + 0.9 * p_building
```

因此不是 GT building mask 作为推理输入。

训练时 GT location 只用于 location audit 和 partition auxiliary supervision；
推理时 partition 使用 RGB `clf_1` 的预测 `p_building`。

## Soft Partition

主方法不是按 RGB 原始颜色做 SLIC。

输入是：

```text
RGB encoder structure
+ RGB structural edge
+ predicted RGB building probability
+ locally aligned RGB-SAR relation
+ alignment confidence
```

跨模态 relation：

```text
[RGB, aligned SAR, |RGB-alignedSAR|, RGB*alignedSAR]
```

association head 输出 9 channels，每个像素只在附近 3×3 candidate region centers 中分配：

```text
Q_pk = P(pixel p -> region k)
```

然后 soft pooling 得到 Region Token，再重投影为 pixel region context，
最后通过 zero-init residual 回到原 `clf_2`。

## “在 RGB 建筑定位中生成”的实现方式

当前是 soft ROI-conditioned partition，不是 hard crop。

- `p_building` 进入 association head；
- A1 fused feature 乘 soft ROI weight；
- region residual 最终由 `building × alignment_confidence` 门控；
- 因而新 Region 对最终分类的有效影响集中在预测建筑区域。

这样比直接 `p_building > 0.5` 硬裁剪更保守，避免定位边界误差直接删掉建筑边缘。

## 训练

父 A1 完全冻结，只训练 `cmsp.*`：

```text
local alignment
relation encoder
association head
content projection
region update
region_gain
```

loss：

```text
L = L_damage
  + 0.05 L_spatial
  + 0.10 L_building
  + 0.05 L_evidence
  + 0.002 L_balance
```

不启用 boundary loss。

## 10000 optimizer steps

正式阶段固定：

```text
10000 optimizer steps
eval_every = 500
```

会在 500、1000、...、10000 共 20 个点做 Validation。

输出：

```text
Valid mIoU
mIoU4
Damaged F1
Intact IoU
Damaged IoU
Destroyed IoU
Event mean / worst
region residual magnitude
alignment confidence
alignment entropy
```

## 一步一步运行

### 1. 解压

```bash
cd /root/data
unzip -o damageformer_roi_soft_partition_v1.zip
```

### 2. 安装

```bash
bash \
/root/data/damageformer_roi_soft_partition_v1/install.sh \
/root/data/BRIGHT-master/bda_benchmark
```

### 3. 进入正确目录

```bash
cd /root/data/BRIGHT-master/bda_benchmark
pwd
```

必须看到：

```text
/root/data/BRIGHT-master/bda_benchmark
```

### 4. 环境

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bright4090
```

### 5. Preflight

```bash
bash \
experiments/damageformer_roi_soft_partition_v1/run.sh \
preflight
```

必须看到：

```text
PREFLIGHT PASSED
```

### 6. 20-step smoke

```bash
bash \
experiments/damageformer_roi_soft_partition_v1/run.sh \
smoke
```

只检查 forward/backward/BF16/gradient/checkpoint/validation。

### 7. 查看 smoke

```bash
bash \
experiments/damageformer_roi_soft_partition_v1/run.sh \
status
```

### 8. 正式 10000-step

```bash
mkdir -p \
/root/data/BRIGHT-train-logs/DAMAGEFORMER_ROI_SOFT_PARTITION_V1

nohup bash \
experiments/damageformer_roi_soft_partition_v1/run.sh \
train \
> /root/data/BRIGHT-train-logs/DAMAGEFORMER_ROI_SOFT_PARTITION_V1/train10000.log \
2>&1 &
```

### 9. 确认运行

```bash
ps -ef | \
grep damageformer_roi_soft_partition_v1.train | \
grep -v grep
```

```bash
tail -f \
/root/data/BRIGHT-train-logs/DAMAGEFORMER_ROI_SOFT_PARTITION_V1/train10000.log
```

### 10. GPU

```bash
watch -n 2 nvidia-smi
```

### 11. 训练完成后汇总

```bash
bash \
experiments/damageformer_roi_soft_partition_v1/run.sh \
summary
```

## 依赖与隔离

这个包不会复制或修改已有：

```text
experiments/damageformer_a1_cross_modal_superpixel_v1/
```

而是复用已经核过的 `association.py / losses.py / core.py / C5 implementation`，
避免重新复制底层 association 后再次引入 BF16 / scatter_add 问题。

旧实验目录必须保留。
