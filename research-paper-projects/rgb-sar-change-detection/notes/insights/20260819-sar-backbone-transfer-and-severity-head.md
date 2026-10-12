# SAR 表征迁移与层级严重度头对当前 BRIGHT 路线的启发

- 用户研究方向：灾前高分辨率 RGB 与灾后单通道 SAR 建筑损毁评估
- 用户研究问题：H10-H3 在测试集上 Damaged gate miss 高，验证增益不能泛化；应如何替换 SAR 分支、设计严重度头并制定冻结/训练策略
- 分析日期：2026-08-19
- 当前证据：P3、H6-A2、H10-H2/H3 的验证与独立测试结果

## 核心判断

1. 原 H10-H3 不应仅从 3000 步延长到 5000 步。其最佳权重位于 2000 步，且完整 H6-A2 被冻结；增加步数只能继续优化一个已表现出事件依赖的 Stage4 小头，不能恢复上游已经丢失的弱损毁证据。
2. 若更换为 SAR 专属预训练编码器，并采用分阶段解冻，5000 步才是合理的新实验预算。
3. 严重度头不应复用同一 Lite-3D-STA。建议使用“建筑约束的多尺度有序证据聚合头”：Stage2/3 保留局部弱损毁，Stage4 表达大范围坍塌；输出 `affected` 与 `destroyed | affected` 两个有序决策。
4. RGB 编码器与定位头继续冻结；SAR 编码器不应永久冻结。先冻结做适配，再只解冻深层，最后视验证结果决定是否解冻中层。

## 可迁移论文模块

| 模块 | 论文证据 | 可迁移方式 | 主要风险 |
|---|---|---|---|
| SARMAE 的 SAR 专属 ViT 与散斑感知预训练 | CVPR 2026；SAR-1M、SARE、SARC；官方代码和权重 | 作为第一优先级 SAR 编码器；接轻量 FPN/投影以输出 Stage2/3/4 | 输入表示、极化和分辨率与 BRIGHT 不完全一致；ViT-B 显存和工程迁移成本较高 |
| SARDet-100K 的 MSFA | NeurIPS 2024；Filter Augmentation + 多阶段迁移；官方代码和权重 | 作为较轻量 CNN 对照，验证“结构滤波 + 迁移”是否改善弱损毁 | 预训练任务是检测，不能直接宣称适合分割；需重建多尺度特征接口 |
| CROMA radar encoder | NeurIPS 2023；雷达/光学独立编码、跨模态对比和重建；官方代码 | 作为多模态预训练消融或初始化来源 | 原模型雷达输入为 Sentinel-1 双通道，分辨率和 BRIGHT 单通道高分辨率差异大，不宜第一轮直接替换 |
| AnySat JEPA | CVPR 2025 Highlight；11 种传感器、尺度自适应编码与 JEPA；官方代码和权重 | 后续测试跨传感器泛化或作为冻结特征对照 | 体系较重、输入配置复杂，难以保持与 A2 的单变量对照 |

## 建议模型：H11-SATR-OH

### SAR 表征阶段

```text
外部 SARMAE ViT-B 权重
        ↓
BRIGHT train-only 短程自监督域适配
        ↓
多尺度特征适配器（输出 1/4、1/8、1/16 或兼容 A2 的层级）
        ↓
先冻结适配，随后仅解冻 SAR 深层
```

训练阶段不得访问验证/测试标签；监督训练时移除预训练重建损失。

### 层级决策阶段

```text
P3 冻结 RGB 定位掩膜
        ↓
Stage2/3：局部屋顶破坏、弱纹理与边缘异常
Stage4：整体结构消失、大范围散射紊乱
        ↓
建筑约束多尺度证据聚合
        ├─ occurrence：Intact vs Affected
        └─ severity：Damaged vs Destroyed | Affected
```

严重度头使用有序条件分类，而不是让同一个 3D 模块同时承担发生检测和严重度估计。

## 最小可行实验

| 实验 | SAR 编码器 | 可训练范围 | 目的 |
|---|---|---|---|
| T0 | 当前 H6-A2 | 全冻结 | 锁定基线 |
| T1 | SARMAE ViT-B | 仅 FPN/投影和现有损毁头 | 判断预训练特征是否可直接迁移 |
| T2 | SARMAE ViT-B | FPN/头 + 最后 2 个 block | 判断有限域适配是否改善 Damaged gate miss |
| T3 | SARMAE + BRIGHT train-only 自监督适配 | 同 T2 | 判断传感器域适配是否进一步改善跨事件泛化 |
| T4 | T2/T3 胜者 | 加多尺度有序严重度头 | 仅在 occurrence 已改善后评估严重度 |

建议每个实验总预算 5000 步、每 500 步验证；最终权重只由验证集选择。若连续三次验证无改善则提前停止。测试集仅用于锁定候选的最终一次评估，不能反复用于调参。

## 冻结策略

- 永久冻结：P3 RGB 编码器、RGB 定位头及其 BN。
- 第 0–500 步：冻结 SAR 主干，只训练接口/FPN/分类头。
- 第 500–2000 步：解冻 SAR 最后两个 block，主干学习率约为新头的 0.1 倍。
- 第 2000–5000 步：只有验证 Damaged gate miss 持续下降时才解冻中层；不解冻浅层和全部主干。
- Transformer 的 LayerNorm 仿射参数可随所属 block 更新；CNN 的小批量 BatchNorm 保持冻结或改用 GroupNorm/LayerNorm。

## 风险与停止规则

- 若 T1 初始线性/冻结迁移已明显差于 A2，先检查输入统计、极化/通道映射和空间分辨率，不进入长训练。
- 若 T2 只提升 Destroyed、Damaged gate miss 不下降，则预训练特征仍偏强破坏，停止严重度头开发。
- 若验证提高但事件非负数少于 10/14，优先检查事件域偏差，不进入测试。
- 不把“训练到 5000 步”视为贡献；贡献应来自 SAR 专属迁移和层级决策的可验证机制。

## 建议题目方案

1. **Speckle-Aware SAR Transfer for Cross-Event RGB–SAR Building Damage Assessment**
   - SAR 专属预训练、分阶段迁移与跨事件评估。
2. **Weak-to-Severe Damage Evidence Decomposition for Heterogeneous RGB–SAR Assessment**
   - 中浅层局部弱损毁与深层整体坍塌证据解耦。
3. **Optical-Anchored, SAR-Adaptive Ordinal Building Damage Assessment**
   - 冻结光学定位锚点，SAR 有限解冻，并以条件有序头输出损毁等级。

## 原文与趋势来源

- SARMAE, CVPR 2026：SAR-1M、散斑感知表示增强、光学语义锚定；官方代码 `MiliLab/SARMAE`。
- SARDet-100K, NeurIPS 2024：指出 RGB 到 SAR 的输入域、结构和迁移过程三重差距，提出 MSFA；官方代码 `zcablii/SARDet_100K`。
- CROMA, NeurIPS 2023：雷达/光学独立编码、对比学习、跨模态融合与掩码重建；官方代码 `antofuller/CROMA`。
- AnySat, CVPR 2025 Highlight：多传感器 JEPA 与尺度自适应编码；官方代码 `gastruc/AnySat`。
