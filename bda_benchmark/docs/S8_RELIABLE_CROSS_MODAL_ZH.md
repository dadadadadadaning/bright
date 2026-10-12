# S8 可靠跨模态改造：中文接入与实验说明

## 1. 改动范围

本次没有覆盖已经存在的 `DamageFormer_S8_IA_MLA.py` 和
`DamageFormer_S8_MICB_DSMIL.py`，而是新增一套独立的
`DamageFormer_S8_ReliableCrossModal.py`。这样旧结果仍可复现，新机制可以逐项消融。

提示中指出的两个核心问题分别是：

1. 原 HCDM 在各层投影后直接计算 `abs(rgb-sar)` 和 `rgb*sar`。异构模态数值不在同一
   表示空间，正常建筑也可能产生较大的直接差异。
2. 原 `gamma_hcdm` 是整张图共享的标量，无法区分背景、完整建筑、局部损坏和彻底损毁。

新实现不直接把关系差异加进四分类特征，而是把流程拆成：

```text
Stage2/3 局部自相似关系
        ↓
RGB 与 SAR 的关系差异
        ↓
变化辅助监督（R1）
        ↓
有界空间可靠性门控（R2）
        ↓
SAR Adapter（R3）
        ↓
Pixel DamageExist + Severity（R4）
        ↓
Stage3 保守跨模态校正（R5）
```

## 2. 各实验的严格含义

| 变体 | 新增机制 | 主分类路径是否使用关系差异 |
| --- | --- | --- |
| R0 | 稳定双 Swin 基线；无 HCDM、无 IA；保留基础软建筑 ROI | 否 |
| R1 | Stage2/3 八邻域结构关系差异；变化辅助头 | 否 |
| R2 | R1 + `Fbase × (1 + αAchange)` | 是，但 `0≤α≤0.1` |
| R3 | R2 + 近似恒等的残差 SAR Adapter | 是 |
| R4 | R3 + 像素 DamageExist/Severity 辅助监督 | 是 |
| R5 | R4 + Stage3 轻量双向校正 | 是，校正强度不超过 0.05 |

R6 公共/私有特征分解没有实现。这是有意的：必须先证明 R1–R5 稳定有效，再引入对齐、
正交和防塌缩等额外超参数。

## 3. 关键实现

### 3.1 Stage2/3 局部结构关系

对 RGB 和 SAR 分别计算中心像素与八邻域的余弦相似度，再比较两种模态的关系图：

```text
D(x,δ) = |cos(Frgb(x), Frgb(x+δ)) - cos(Fsar(x), Fsar(x+δ))|
```

边界位置使用有效掩码，不进行循环移位。余弦归一化和关系相减固定使用 FP32，避免
bf16/fp16 下出现数值问题。R1 的关系特征只接受“完整/损毁”变化辅助监督，不进入主分类。

### 3.2 有界空间门控

R2 使用关系特征、建筑概率以及两个模态的质量估计生成空间注意力：

```text
Achange = sigmoid(φ[Drelation, Pbuilding, Qrgb, Qsar])
Fdamage = Fbase × (1 + αAchange)
```

其中 `α=0.1×sigmoid(alpha_logit)`，初始化 `alpha_logit=-6`，因此刚加载基线时几乎不改变
主路径；训练过程中乘数始终位于 `[1,1.1]`，不会负向抵消基础特征。

R2 及后续实验会同时监督关系变化头和真正用于调制主特征的 `Achange` logits，避免出现
“辅助头学会变化、实际门控却没有直接监督”的分离。

### 3.2.1 S8 基础 ROI 保持

基础权重来自 `S8_MICB_DSMIL`，其四分类头是在
`FusedFeature × SoftBuildingROI` 上训练的。因此 R0–R5 保留：

```text
ROI = 0.1 + 0.9 × stopgrad(Pbuilding)
Fbase = FusedFeature × ROI
```

只移除 MICB/DSMIL 实例池化，不移除分类头已经依赖的软建筑 ROI。否则即使权重形状完全
匹配，初始验证也会出现严重的语义不兼容。

### 3.3 SAR Adapter

R3 的 Adapter 是残差结构，最后一层卷积零初始化：

```text
SARout = SARin + Adapter(SARin)
```

所以初始化时严格等于原 SAR 输入，不会在开始训练时破坏已迁移的 S8 编码器。

### 3.4 像素层级监督

R4 保留原四分类主头，并增加：

- DamageExist：仅建筑像素参与，`Intact=0`，`Damaged/Destroyed=1`；
- Severity：仅已损毁像素参与，`Damaged=0`，`Destroyed=1`；
- 背景和无关类别使用 `ignore_index=255`，不错误监督。

这个拆分直接针对 `Damaged→Intact` 和 `Destroyed→Intact` 混淆。

### 3.5 Stage3 保守校正

R5 只在建筑概率高、原始 Stage3 关系差异低的区域增强跨模态交互。校正形式是向对方
投影特征进行小幅插值，强度严格小于 0.05。建筑概率和一致性条件在校正门中停止梯度，
避免模型为了扩大校正范围而篡改定位概率或关系能量。

## 4. 权重迁移

训练器要求 `--base_ckpt`。迁移器只加载以下共享核心：

- `encoder_1`、`encoder_2`；
- `fusion_layer_1/2/3`；
- `clf_1`、`clf_2`。

从旧 S8 checkpoint 迁移时，HCDM、IA、MICB、原型内存等不兼容键只报告、不加载；从
前一级 R 实验继续时，名称和形状兼容的关系模块、门控等会一并继承。只要任一共享核心键
缺失或形状不匹配，训练立即停止，而不是静默使用随机权重。报告保存在：

```text
<实验目录>/migration_report.json
```

## 5. 推荐执行顺序

在服务器项目根目录执行：

```bash
cd /root/shared-nvme/BRIGHT-master/bda_benchmark
conda activate bright_min

bash run/run_S8_reliable_cross_modal_5090.sh preflight

VARIANT=r0 bash run/run_S8_reliable_cross_modal_5090.sh baseline

VARIANT=r0 bash run/run_S8_reliable_cross_modal_5090.sh smoke
VARIANT=r0 bash run/run_S8_reliable_cross_modal_5090.sh pilot

VARIANT=r1 bash run/run_S8_reliable_cross_modal_5090.sh smoke
VARIANT=r1 bash run/run_S8_reliable_cross_modal_5090.sh pilot
```

先用 R0 确认稳定双 Swin 基线，再判断 R1 的结构关系监督是否真正优于 R0。R1 通过后再
运行 R2，不能同时跳到 R4/R5：

```bash
VARIANT=r2 bash run/run_S8_reliable_cross_modal_5090.sh pilot
VARIANT=r3 bash run/run_S8_reliable_cross_modal_5090.sh pilot
VARIANT=r4 bash run/run_S8_reliable_cross_modal_5090.sh pilot
VARIANT=r5 bash run/run_S8_reliable_cross_modal_5090.sh pilot
```

严格消融表中，R1–R5 必须使用同一个原始 `BASE_CKPT`、同一个 seed 和相同训练步数。把前一级
R 的最佳权重作为后一级起点属于“顺序热启动”，代码允许这样做，但它必须作为另一套协议，
不能与同起点消融结果混在一张表里。

如果基线 checkpoint 不在脚本默认位置：

```bash
BASE_CKPT=/你的绝对路径/best_val_miou.pth \
VARIANT=r1 \
bash run/run_S8_reliable_cross_modal_5090.sh pilot
```

## 6. 每一步的判断标准

不要只看最终 `valid_mIoU`，还应比较：

- `Damaged→Intact` 和 `Destroyed→Intact` 是否下降；
- R1 的 `change_iou/change_recall` 是否高于随机且随训练改善；
- R2 的 `reliability_alpha` 是否有限增长，乘数是否始终不超过 1.1；
- R3 的 `sar_adapter_delta_mean` 是否从 0 平稳增长而非突然爆炸；
- R4 的 `pixel_exist_iou`、`destroyed_vs_damaged_iou` 是否改善；
- R5 的 `rectifier_safe_fraction` 是否只覆盖可靠建筑区域。

每个变体先跑 20 步 smoke，再跑 3000 步 pilot；只有验证集正向且没有类别塌缩，才进入
10000 步 full。测试集在最终配置锁定前不得访问。

## 7. 文件清单

- 模型：`model/DamageFormer_S8_ReliableCrossModal.py`
- 模块：`model/reliable_cross_modal/blocks.py`
- 训练：`script/standard_ML/train_S8_reliable_cross_modal.py`
- 启动：`run/run_S8_reliable_cross_modal_5090.sh`
- 实验矩阵：`configs/s8_reliable_cross_modal/experiment_matrix.json`
- 合成测试：`tests/test_s8_reliable_cross_modal.py`

## 8. 串行容错训练

上传最新代码后，可让 R0–R5 自动依次执行：

```bash
nohup bash run/run_S8_reliable_cross_modal_sequence.sh pilot \
  > /root/shared-nvme/BRIGHT-train-logs/S8_ReliableCrossModal/controller.log 2>&1 &
```

控制器会为每个变体先运行 smoke，再执行 pilot；各实验使用独立目录和日志。某个实验失败
会记录在 TSV 汇总中，默认继续检查后续实验。重新运行同一命令时：

- 首先执行 R0 全验证集、0训练步的 checkpoint 兼容性预验；

- 已有 `final.pth` 的实验自动跳过；
- 已有 `latest_resume.pth` 的实验自动续训；
- 尚未保存断点的失败实验从共同基础权重重新开始。

只运行部分实验：

```bash
VARIANTS="r0 r1 r2" \
nohup bash run/run_S8_reliable_cross_modal_sequence.sh pilot \
  > /root/shared-nvme/BRIGHT-train-logs/S8_ReliableCrossModal/controller_r0_r2.log 2>&1 &
```
