# DamageFormer Building Context V2

## 1. 目标

先完整验证：

```text
A1 DamageFormer
→ LCA RGB-SAR局部对齐
→ 非颜色驱动的可学习Region
→ Region→Building
→ Building Context
→ Building→Region
→ soft unpool
→ zero-init Pixel residual
→ 原DamageFormer四分类头
```

本版本不加入 Memory、GNN、Morphology classifier。

## 2. 主方法的 Region 不是颜色超像素

主方法不使用 SLIC/SLICO，也不使用 RGB 原始颜色距离。

它复用原 `damageformer_a1_cross_modal_superpixel_v1` 的 learned association：

```text
RGB encoder structure
+ RGB structural edge
+ building probability
+ aligned RGB-SAR relation
+ alignment confidence
→ SCN-style local 9-way association
→ soft Region Tokens
```

`c2_slico` 仍会在“全部消融”里运行，但它只是固定颜色/外观区域的对照基线，不属于最终方法。

## 3. 为什么整体模型第一个跑

`run_everything.sh` 第一项就是：

```text
full_joint
```

从 A1 checkpoint 出发，同时训练 C5 Region 与 Building Context，
让你最早看到整条路线有没有正向信号。

但是 `full_joint` 不是最干净的因果证据，因为 Region 与 Building Context 同时变化。
最终必须看：

```text
old c5_aligned_assoc
vs
bir_on_frozen_c5
vs
bir_shuffle_on_frozen_c5
```

## 4. 全部实验

### 原 Region 组

|实验|意义|
|---|---|
|C1 LCA|局部对齐本身|
|C2 SLICO|固定RGB外观区域对照（不是最终方法）|
|C3 RGB association|只靠RGB结构学习Region|
|C4 Raw SAR|未对齐SAR是否干扰|
|C5 Aligned+Confidence|核心跨模态Region|
|C6 No confidence|confidence的作用|
|C7 Shifted SAR|正确跨模态对应的因果负对照|
|C8 Damage hint|是否只是复制父分类边界|
|C9 Boundary|边界弱约束是否有额外作用|

### Building Context 组

|实验|意义|
|---|---|
|full_joint|完整方案，最先跑|
|bir_on_frozen_c5|冻结C5，只训练Building Context；最关键因果实验|
|bir_shuffle_on_frozen_c5|同参数量但错误Building Context|
|rgb_only_building|没有SAR Region时Building Context表现|
|raw_sar_building|未对齐SAR + Building|
|no_conf_building|无confidence + Building|
|shifted_sar_building|错误空间SAR + Building|
|wrong_building_joint|联合训练但上下文来自错误建筑|

## 5. 为什么 Probe 先跑 5000

5000 steps 是准入预算，不是最终训练预算。

原因：

- A1 backbone被冻结；
- 只训练 `cmsp.*` / `bir.*` 小插件；
- 两个残差入口都是 zero-init，初始模型接近父模型；
- 每500步完整Validation一次，5000步约10个趋势点；
- 先判断机制是否持续提升，避免十几个消融全部直接跑2万~5万步。

阶段：

```text
smoke    20       功能/AMP/shape/gradient
probe    5000     所有变体公平筛选
confirm  20000    关键结论长程确认
```

如果你希望“所有实验都跑到长程”，也可以执行：

```bash
bash experiments/damageformer_building_context_v2/run_everything.sh confirm
```

但建议先全部5000，再决定哪些值得20000。

## 6. 安装

```bash
unzip damageformer_building_context_v2.zip
bash damageformer_building_context_v2/install.sh \
  /root/data/BRIGHT-master/bda_benchmark
```

## 7. 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bright4090

export A1_CKPT=/root/data/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/a1_long35000_seed3407/best_val_miou.pth

bash experiments/damageformer_building_context_v2/run.sh preflight
```

先全部20-step：

```bash
bash experiments/damageformer_building_context_v2/run_everything.sh smoke
```

全部5000-step，完整模型第一个：

```bash
mkdir -p /root/data/BRIGHT-train-logs/DAMAGEFORMER_BUILDING_CONTEXT_V2

nohup bash experiments/damageformer_building_context_v2/run_everything.sh probe \
  > /root/data/BRIGHT-train-logs/DAMAGEFORMER_BUILDING_CONTEXT_V2/all_probe5000.log \
  2>&1 &
```

查看：

```bash
tail -f /root/data/BRIGHT-train-logs/DAMAGEFORMER_BUILDING_CONTEXT_V2/all_probe5000.log
```

## 8. 汇总

```bash
python -m bda_benchmark.experiments.damageformer_building_context_v2.summarize \
  --old_root /root/data/BRIGHT-checkpoints/DAMAGEFORMER_A1_CROSS_MODAL_SUPERPIXEL_V1 \
  --new_root /root/data/BRIGHT-checkpoints/DAMAGEFORMER_BUILDING_CONTEXT_V2 \
  --tag probe5000 \
  --seed 3407
```

## 9. 最关键判断

首先看整体：

```text
full_joint vs A1 / old C5
```

然后看最干净因果：

```text
bir_on_frozen_c5 > old c5_aligned_assoc
```

同时要求：

```text
bir_on_frozen_c5 > bir_shuffle_on_frozen_c5
```

否则无法证明提升来自“同栋建筑上下文”。

最后看机制：

```text
full_joint > rgb_only_building
full_joint > raw_sar_building
full_joint > no_conf_building
full_joint > shifted_sar_building
```

重点指标：

```text
Valid mIoU
Damaged IoU
Damaged F1
Destroyed IoU
Event mean
Event worst
BIR gain
```
