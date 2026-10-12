# SPI Building Relational Memory V1

这是一个与旧SLIC、P3、SALT和Memory实验完全隔离的新协议。它不修改原始DamageFormer，固定读取D0初始化的ROI-LCA检查点。默认使用`a3_conf3`；只有ROI-LCA错位审计通过后，才能把A3写成正式上游模块。

## 数据流

```text
灾前RGB -> SLICO -> 预测建筑连通域切分
灾后SAR -> A2局部搜索 -> 对齐到RGB特征网格
建筑内标准化RGB区域 + 对齐SAR + 跨模态关系 + Core/Ring -> 超像素Token
同一预测建筑内多个Token -> Mean/Top-k/Quantile/Relation -> Building Key
Train-only class×event×size Memory -> 跨事件粗检索 -> 节点集合精细重排
```

质量与内容严格分工：

- `rgb/sar/relation/multiring`可以形成语义Key；
- `M1`读取未对齐`sar`，`M2-M6`明确读取单独缓存的`sar_aligned`；
- 除`M0`风格对照外，RGB先在每栋建筑内部标准化，削弱屋顶颜色和地区风格；
- A3 entropy/confidence只用于关系注意力可靠性；
- P3/D0状态不进入Key，只为后续Utility Verifier保留；
- Validation只读Memory；V1禁止Test和像素写回。

## 逐级消融

| 变体 | 作用 |
|---|---|
| `m0_rgb_mean` | RGB高层内容+整栋Mean，地区/风格偏置对照 |
| `m1_unaligned_multistat` | 未对齐RGB-SAR，多统计聚合 |
| `m2_aligned_multistat` | A2/A3对齐内容，证明对齐价值 |
| `m3_aligned_multiring` | 对齐内容+Core/Ring多尺度统计 |
| `m4_aligned_relation` | 加入建筑内部相对位置kNN关系 |
| `m5_shuffled_relation` | 错误位置关系控制，参数量保持一致 |
| `m6_aligned_relation_quality` | 正确关系+A3熵可靠性加权 |

主要比较：

```text
M2-M1：SAR对齐是否有效
M3-M2：Core/Ring统计是否有效
M4-M3：区域关系是否有效
M4-M5：正确关系是否优于错误关系
M6-M4：A3可靠性是否减少错误关系影响
Memory-Direct：历史证据是否提供分类头没有的信息
```

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash experiments/spi_building_relational_memory_v1/run.sh preflight

nohup bash experiments/spi_building_relational_memory_v1/run.sh stage_all \
  > /root/data/BRIGHT-train-logs/SPI_BUILDING_RELATIONAL_MEMORY_V1/stage_all.log 2>&1 &
```

如果A3尚未通过错位审计，可显式换成A2做机制对照：

```bash
ROI_VARIANT=a2_lca3 \
ROI_CKPT=/root/data/BRIGHT-checkpoints/DAMAGEFORMER_ROI_LCA_V1/a2_lca3_probe5000_seed3407/best_val_miou.pth \
bash experiments/spi_building_relational_memory_v1/run.sh stage m2_aligned_multistat
```

## 准入

V1只验证建筑级裸检索，不是官方像素mIoU证据。进入Utility Verifier之前必须满足：

```text
M2 > M1
M3 >= M2
M4 > M3
M4 > M5
M6 >= M4
Memory - Direct >= +0.20pp Macro-F1
正确Memory - 随机标签Memory >= +15pp
P3/D0错误建筑恢复率 >= 25%
```

任何一项失败，都保持`writeback`锁定。通过后另建V2，在Val上训练低容量Utility Verifier，再评估像素mIoU、Damaged F1、14事件和Bootstrap CI。
