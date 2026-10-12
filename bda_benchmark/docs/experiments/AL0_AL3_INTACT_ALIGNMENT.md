# AL0--AL3：Intact 条件跨模态对齐实验契约

## 科学问题

先用 Intact 建筑学习可以被“正常 RGB--SAR 对应”解释的模态差异与配准误差，
再检查无法解释的对齐残差是否能稳定区分 Damaged 与 Intact。AL2 通过前，残差
不得进入分类器或 P3。

## 代码与状态隔离

| 层 | 文件/目录 | 可写状态 |
|---|---|---|
| 冻结特征源 | H6-A2 checkpoint | 只读；仅载入 `rgb_encoder`、`sar_encoder` |
| 对齐模型 | `model/IntactAnchoredAlignment.py` | AL1 只训练 Stage2/3 aligner |
| 训练与审计 | `script/standard_ML/train_IntactAnchoredAlignment.py` | 唯一入口；无 Test 参数 |
| 运行控制 | `run/run_IntactAnchoredAlignment_5090.sh` | 阶段门控与独立输出目录 |
| AL0/AL2结果 | `BRIGHT-eval-results/ALIGNED_DAMAGE_EVIDENCE` | JSON，不含模型权重 |
| AL1/AL3权重 | `BRIGHT-checkpoints/ALIGNED_DAMAGE_EVIDENCE` | 每阶段独立目录 |

RM、SRAC、P3-reset、OA-HSDNet 的 checkpoint 不会被读取或覆盖。P3 分类器也不在
这一实验线中，避免把“对齐证据是否成立”和“融合是否成功”混成一个问题。

## 阶段

### AL0：未训练描述子基线

冻结 H6-A2 Stage2/3，使用固定局部硬匹配和主导平移去除，不训练。输出未对齐
余弦距离和匹配后几何残差的建筑区域 AUROC。

### AL1：Intact-only 对齐训练

每个尺度执行：

```text
RGB Stage-k -> RGB独立投影 --\
                               局部Correlation Volume -> soft offset
SAR Stage-k -> SAR独立投影 --/                         -> residual offset
                                                         -> warp SAR
                                                         -> aligned residual
```

只有训练标签 `Intact` 的像素进入对齐数据项。Mask以面积覆盖率下采样到
Stage2/3，避免小建筑被最近邻采样完全删除；平滑、位移幅度和方差防坍缩是正则项。
验证选权重只看 Intact 对齐残差，不查看 Damaged AUROC。

### AL2：冻结残差审计

冻结 AL1，在验证集 P3-independent 建筑连通区域代理上统计：

- Damaged vs Intact global AUROC；
- 逐事件 macro AUROC；
- 方向一致事件数；
- 相对未对齐残差的 AUROC 增量。

全部满足以下条件才写入 `passed=true`：

```text
Global Damaged AUROC >= 0.65
Event Macro AUROC    >= 0.60
方向一致事件         >= 10
相对未对齐增益       >= 0.05 AUROC
```

### AL3：Occurrence 头

运行脚本首先读取 AL2 JSON；失败时直接退出。通过后冻结编码器和全部对齐器，仅训练：

```text
[Stage2/3 residual, entropy, margin] (6 channels)
 -> Conv3x3 6->32 -> GN -> GELU
 -> Conv3x3 32->16 -> GN -> GELU
 -> Conv1x1 16->2
 -> Intact / Affected
```

AL3仍不修改 P3，也不预测 Damaged/Destroyed。只有它自身稳定后，才能设计后续与
P3 severity 的组合实验。
