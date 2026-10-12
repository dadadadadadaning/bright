# P3–S2 选择瓶颈与跨模态下一阶段路线（2026-09-07）

## 1. 研究方向与当前问题

当前父系统固定为：

```text
Pre-RGB + Post-SAR
        -> P3 像素语义预测
        -> LocCC 伪建筑
        -> A6 区域可靠性
        -> fixed S2 建筑上下文候选
```

现有结果说明：

- P3 是稳定的基础语义预测，但存在 Damaged gate-miss 和建筑内部碎片化；
- fixed S2 对 P3 有稳定净增益，因此候选生成本身有效；
- Building Oracle 相对 fixed S2 仍约有 `+0.9868pp` 的理论空间，说明 P3/S2 二选一值得研究；
- 现有像素 Gate、整栋池化 Memory、Core/Ring 特征均不能在无 GT 时可靠预测选择；
- Selective Strip-RGID 的 Split Oracle 仅约 `+0.000011pp`，说明继续分裂 LocCC 几乎没有可利用空间。

所以当前主要瓶颈不是“P3 或 S2 谁整体错误”，而是：

> fixed S2 的有益和有害修改混合存在，而现有建筑级汇总特征没有表达修改发生的位置、方向和局部跨模态依据。

## 2. 问题分解

### 2.1 P3 的责任

P3 负责生成基础类别和建筑定位。它的典型问题是漏判、建筑内部不一致和边缘不稳定。

### 2.2 fixed S2 的责任

S2 不是第二个独立模型，而是基于 LocCC 的候选修改器。它能修正一部分 P3 错误，也会把一部分 P3 正确像素改坏。

### 2.3 当前选择器的责任

Building Oracle 已证明“选谁”有较大理论价值，但 PI-M0 的跨事件 AUC 只有 `0.5176`。因此目前失败的是可观测证据和表示方式，不是 Oracle 假设本身。

### 2.4 RGID 的结论

RGID 全部分裂只改变 188 个像素，Split Oracle 仅选择 1 个父区域并改变 15 个像素。该路线缺少理论上限，应停止 Strip-RGID、Cut-band verifier 和更复杂分裂头。

## 3. 最推荐的新主线：修改模式条件下的局部跨模态验证

先运行已经隔离实现的 `S2 Modification Pattern E0`，把 S2 changed pixels 聚合为建筑内修改簇。之后的跨模态模块不处理整幅图，也不重新生成候选，只回答：

> 这一簇 `P3 -> S2` 修改，是否得到灾前 RGB 结构和灾后 SAR 局部变化的共同支持？

推荐结构：

```text
Frozen P3 logits/features
        +
fixed S2 changed clusters
        |
        +-- 修改方向、面积、连通性、距边界、P3 margin
        +-- RGB Stage3/4 局部结构
        +-- SAR Stage3/4 局部纹理
        +-- RGB->SAR 局部可位移匹配
        +-- 两模态内部关系差
        |
        -> parameter-free E0 audit
        -> 通过后训练低容量 verifier
        -> 接受该建筑/修改簇的 S2，或回退 P3
```

## 4. 跨模态证据的安全定义

### 4.1 局部可位移匹配

以灾前 RGB 的修改簇为几何锚点，在灾后 SAR 的 `±1/±2` feature-token 邻域搜索，而不是比较同位像素：

```text
RGB query token -> SAR local candidates -> best similarity / residual / offset / entropy
```

保留四类统计：最佳相似度、最佳残差、位移大小、匹配熵。匹配置信度低时应回退 P3/fixed S2，不应强制融合。

### 4.2 模态内关系变化

在同一修改簇中分别计算 RGB 和 SAR 内部的区域关系：

```text
A_rgb(i,j) = cosine(r_i, r_j)
A_sar(i,j) = cosine(s_i, s_j)
E_rel(i,j) = abs(A_rgb(i,j) - A_sar(i,j))
```

这比较的是组织关系改变，避免直接要求 RGB 与 SAR 特征处于同一数值空间。

### 4.3 方向条件化

必须按修改方向分组，例如：

- `Intact -> Damaged`：重点检查是否存在可靠的受损变化证据；
- `Background -> foreground`：重点检查 RGB 建筑结构和 SAR 支持，防止边界外溢；
- `Damaged <-> Destroyed`：暂不由该模块改变，继续交给严重度分支。

### 4.4 禁止强制全局对齐

灾前 RGB 与灾后 SAR 的真实差异既包含模态差，也包含灾害变化。若对所有位置强制特征一致，可能把真正的损毁信号当成配准或域差异消除。跨模态对齐必须是局部、置信度门控、变化条件化的。

## 5. 2024–2026 本地模块迁移分级

### A 级：适合改造后进入实验

1. **RelationAwareAttention（2025）**
   - 可迁移点：允许关系权重参与注意力；
   - 插入点：S2 修改簇或固定区域 token，而非整幅 Stage4；
   - 前提：无参数 relation E0 先通过；
   - 风险：密集使用会把关系网络容量误当作关系证据收益。

2. **HFP（2025）或 Edge-Gaussian/Scharr（2025）**
   - 可迁移点：提取灾前 RGB 的边界/高频结构和灾后 SAR 的稳健梯度；
   - 插入点：修改簇及其 halo 内，作为 verifier 证据；
   - 不应插入：P3 全局编码器或已经失败的 FSA 接口。

3. **HFA-PANet 的非孪生层级对齐思想（2025）**
   - 可迁移点：不同模态使用独立编码并逐层对齐；
   - 本项目改造：只在候选修改区做置信度门控的局部对齐；
   - 风险：整网迁移成本高，且可能抹除真实变化。

4. **M2CD/MCD-Mamba 的模态专家思想**
   - 可迁移点：RGB、SAR 使用模态专用专家，而不是共享所有参数；
   - 本项目改造：若局部证据通过，再在 SAR Stage3 增加零初始化小专家；
   - 风险：此前 S2 蒸馏和全局 SAR adapter 已失败，不能原样重复。

### B 级：只适合作为小规模对照

1. **DiffAttention（2025）**：只能用于少量修改簇 token；密集图上是 `O(N^2)`，且本身不是跨模态变化模块。
2. **KernelSelectiveFusionAttention（2025）**：可作为 SAR 多尺度专家对照，但需改成零初始化残差和稳定归一化。
3. **CrossScaleBoundaryFusion**：仅在小建筑边界 F1 诊断明确后，做 Stage3 到 Stage4 的安全残差。

### C 级：当前不采用

- `MaskAttention`：本地实现依赖随机掩码，不适合作为可复现实验；
- `Lightweight Cross Attention`：来源任务是低照度颜色/亮度交互，不能直接解释 RGB-SAR 变化；
- 本地 `DeformableInteractiveAttention`：没有完整的可变形采样机制，也不是恒等安全残差；
- 原接口 FSA/RCM：FSA 门几乎未打开，RCM 显著下降，不能继续组合；
- 全图强制 cross-attention、全局特征差、全局蒸馏：已有证据不支持。

## 6. 下一阶段消融矩阵

```text
XM0  S2 modification spatial pattern only
XM1  XM0 + RGB boundary/high-frequency support
XM2  XM0 + SAR robust texture support
XM3  XM0 + co-located direct RGB/SAR difference（负对照）
XM4  XM0 + local shiftable matching
XM5  XM0 + within-modality relation difference
XM6  XM0 + confidence-gated matching + relation
XM7  XM6 with spatial/event shuffle（因果对照）
```

E0 准入建议：

- formal cross-event AUROC `>= 0.60`；
- event-macro AUROC `>= 0.58`；
- 至少 `10/14` 事件方向正确；
- permutation `p < 0.05`；
- 相对 spatial-only 和 shuffle 至少提高 `0.02 AUC`；
- 正负样本和各修改方向具有足够覆盖。

只有 E0 通过，才训练单一两层 MLP/verifier。最终重建必须比较 P3、fixed S2、selective S2、Building Oracle，并要求相对 fixed S2 的 mIoU 增益、bootstrap 下界、Damaged F1 和事件稳定性同时通过。

## 7. 若以后修改 P3 编码器

必须作为独立支线，不与选择器同时训练：

```text
B0 frozen P3
B1 SAR Stage3 zero-init expert
B2 SAR Stage4 zero-init expert
B3 change-conditioned local alignment
B4 relation residual
B5 only independently admitted modules combined
```

第一轮仍采用短 pilot，并额外检查残差门是否真正打开、梯度是否非零、输出是否实际改变。FSA 的历史结果说明“训练完成”并不代表模块已经参与预测。

## 8. 任务匹配、创新性与风险

### 任务匹配

该路线直接对应 BRIGHT 的三类困难：异构模态差、配准残差、建筑内部预测不一致；同时保持 P3 和 fixed S2 的已验证收益。

### 潜在创新

创新点不应写成“使用 cross-attention”，而应写成：

> 以 S2 候选修改模式为条件，在伪建筑内部使用可位移对应和模态内关系变化，验证建筑上下文传播的效用，并通过安全回退机制避免有害修改。

### 主要风险

- 修改簇样本太少或跨事件分布不一致；
- SAR 斑点导致局部匹配置信度失真；
- 强制对齐抹除真实灾害变化；
- verifier 学到事件类型或建筑面积捷径；
- 同一事件进入 Memory/近邻导致泄漏。

## 9. 可执行论文题目候选

1. **Utility-Aware Cross-Modal Modification Verification for All-Weather Building Damage Assessment**
2. **Change-Conditioned Local Correspondence for Safe Building Context Propagation in Optical-SAR Damage Mapping**
3. **Relation-Consistent Pseudo-Building Verification for Heterogeneous Disaster Damage Assessment**
4. **From Candidate Correction to Reliable Acceptance: Cross-Modal Utility Learning for Building Damage Assessment**
5. **Safe Optical-SAR Context Refinement via Modification-Pattern and Local-Relation Evidence**

## 10. 外部依据与代码可用性

- M2CD 官方实现：<https://github.com/circleLZY/M2CD>
- HFA-PANet 官方实现：<https://github.com/TongfeiLiu/HFA-PANet-for-MCD>
- MCD-Mamba 官方实现：<https://github.com/Ci2Lab/MCD-Mamba>
- M2Change 官方实现：<https://github.com/Weikan0425/M2Change/>
- 3MOS 跨模态匹配数据与代码：<https://github.com/3M-OS/3MOS>
- BRIGHT 官方仓库与方法索引：<https://github.com/ChenHongruixuan/BRIGHT>
- C-DiffSET 官方实现：<https://github.com/KAIST-VICLab/C-DiffSET>

这些工作支持模态专用编码、局部对应、层级对齐和不确定性建模的研究方向，但不能直接证明它们会改善当前 P3/S2 选择。所有迁移仍需遵守 E0 无参数审计、单模块准入和跨事件验证。

## 11. 最终落地建议

```text
停止：Strip-RGID / Cut-band / 原接口 FSA / RCM / 全局 Memory
保留：P3 + LocCC + A6 + fixed S2
当前：完成 S2 Modification Pattern E0
随后：在修改簇上增加 local matching 与 relation E0
通过：训练一个低容量 Utility Verifier
失败：保留 fixed S2，并转向改进 P3 的原生 Damaged 表示，而不再训练 P3/S2 选择器
```
