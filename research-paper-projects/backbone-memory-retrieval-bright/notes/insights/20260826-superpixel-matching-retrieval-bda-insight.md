# 从 SRAC 与 IM4CD 重新理解 BRIGHT 中的超像素检索

## 结论先行

SRAC 的公开结果不能被解释为“超像素一加就提升 8.08 pp”。当前可核验的事实只有：完整系统包含可学习超像素、结构化节点嵌入、标注 Memory、元数据和 Transformer 式查询—支持联合分类，并由公开新闻稿报告 75.65% mIoU。没有公开全文消融时，任何单模块归因都不成立。

对当前项目真正可迁移的不是整套复制，而是三条原则：

1. 超像素负责形成结构一致的检索单元，不负责直接判定损毁；
2. 检索 Key 必须针对“跨事件相似损毁”设计，不能直接复用像素分类特征；
3. Memory 是支持证据，最终分类应联合读取当前区域、历史邻居与上下文，而不是把 kNN 类别硬覆盖 P3。

## 一、为什么图中的完整 SRAC 可能成功

### 1. 查询单位与任务单位一致

像素检索会把同一建筑拆成大量噪声查询。图中的 SRAC 先把像素压缩成结构区域，再以区域查询 Memory。若一幅图有 `H×W` 个像素和 `K` 个超像素，暴力查询复杂度从约 `O(HW·M)` 降到 `O(K·M)`，且邻居 ID 可以预计算。

速度只是附加收益。更重要的是，区域级 Key 能把建筑边界、面积和局部损毁范围放入同一表示，减少斑点噪声和预测碎片。

### 2. 它不是“平均池化＋kNN”

结构图显示：像素特征先经过空间注意力加权，再做超像素池化；同时保留 raw mean/std 跳连，经过主投影和归一化形成节点向量。由此，Key 同时包含：

- 学习得到的语义特征；
- 区域内重要位置权重；
- SAR 原始统计；
- 区域几何结构。

当前 RM 的 Key 主要来自冻结 P3 区域特征和少量统计，且 P3 从未为跨事件检索训练。这是两者最根本的差别。

### 3. 它不是“Memory 三分类器替代主模型”

图中 Top-R 邻居、当前超像素特征和灾害类型/地区/几何元数据一起进入 Transformer 式分类器。Cross-Attention 可以学习：

- 哪个历史邻居与当前区域真正相关；
- 哪个邻居只是同事件或同传感器捷径；
- 当前 P3/节点证据是否足以拒绝 Memory；
- 稀有类别支持样本应该贡献多大权重。

RM3E/RM6-P 则使用手工余弦相似度与全局 alpha。它只有“相似度大小”，没有 query-support 关系推理，也没有可学习的拒绝项。

### 4. 它把多个困难同时处理了

公开结构同时改变了 backbone、分区、区域嵌入、Memory、元数据与分类器。总性能可能来自这些模块的协同，也可能部分来自更强 VMamba 主干。没有消融，不能说 8.08 pp 是 Memory 或超像素单独带来的。

## 二、为什么当前 RM6-P 很接近安全，却仍失败

RM6-P 已经成功解决“权限过大”：只修改 32 个区域，12/14 事件不下降，Macro F1 仅下降 0.0563 pp。但 32 次修改中：

- Intact：2 次救援、20 次伤害，净 `-18`；
- Damaged：7 次救援、1 次伤害，净 `+6`；
- Destroyed：净 `+1`；
- 总体净纠正：`-11`。

这证明 P3 锚定公式正确，但 Key/亲和度不够好。全局 alpha 无法区分“同样是 Memory 建议 Damaged，为什么一栋是真的 Damaged，另一栋只是跨事件外观相似”。继续扫 alpha 只能改变修改数量，不能改变排序质量。

## 三、IM4CD 对当前 Key 的真正启发

IM4CD 的核心不是另一个 Transformer，而是放弃共址像素差：

```text
RGB局部模板
→ 在SAR邻域形成响应面
→ 读取峰值位置、峰值尖锐度和峰值偏移
→ 去除全局配准运动
→ 剩余局部几何残差作为变化证据
```

同一建筑未发生结构变化时，即使 RGB/SAR 数值完全不同，最佳局部对应通常仍稳定；发生损毁时，响应峰更散、偏移残差更大或双向匹配不一致。这些指标比 `RGB特征-SAR特征` 更接近“对应关系是否被破坏”。

但它主要回答 Changed/Unchanged，不直接回答 Damaged/Destroyed。因此正确迁移方式是：

- 匹配残差用于 occurrence/reliability 与检索 Key；
- P3/SAR 语义和 raw mean/std 继续负责严重度；
- 匹配结果不能单独替换四类预测。

本地代码已经实现了一个缩小版、诚实声明的迁移：

- `im4cd_matching_runtime.py`：5×5 局部余弦响应、硬峰位置、峰值增益、margin、熵、主导平移和局部残差；
- `build_RM0_region_cache.py --matching_stats`：把匹配统计写入 train-only Memory 和 validation query；
- `audit_IM4CD_region_matching.py`：比较共址差与匹配几何，且不修改 P3；
- `run_IM4CD_RM_matching.sh`：完整 validation-only 协议。

它不是完整 IM4CD 复现：原文使用稀疏控制点和稳健投影变换，本地代码只在已近似配准的 P3 特征网格上估计主导离散平移。

## 四、超像素应该怎样借鉴

### 不该再次做的版本

```text
高熵软网格
→ 区域平均
→ 强制投影回P3
```

该版本已由 SR1P/SR1PA/SR1PAI 否定：分区熵约 0.908、有效候选约 7.38、背景泄漏约 34%，本质是建筑感知软平滑。

### 合理角色

超像素应当只承担：

1. 将建筑内像素压缩为少量连通节点；
2. 阻止跨建筑和背景泄漏；
3. 保留局部 Damaged 子区域，而非整栋平均；
4. 为 Memory 提供固定、可追踪的 query ID。

如果已有 BRIGHT 实例标签，应优先用 train/val 实例作为上限和监督，不应继续让一个未通过分区 Gate 的软 SSN 控制下游分类。若没有实例标签，先用 P3 建筑连通区域内的硬 SLIC/SNIC 作为受控查询单元；可学习 SSN 后置。

### 接入条件

- 不跨越 P3/GT 建筑外边界；
- 背景泄漏低于 25%；
- 硬分配连通；
- 每栋建筑保留多个局部节点，不能只有整栋均值；
- 节点标签纯度和边界召回在 train/val 都稳定；
- 用节点后，救援/伤害 AUROC 高于建筑整区 Key，而不是只看 Boundary F1。

## 五、下一阶段模型与实验链

### IM0：无训练匹配证据审计（立即执行）

输入：冻结 P3 RGB/SAR 同尺度特征和 P3 建筑区域。

区域统计：

- 共址基线：cosine distance mean/std/q90；
- 匹配统计：1-peak、peak gain、硬偏移残差、Top1-Top2 margin、响应熵、置信度加权残差。

验证问题：匹配统计能否比共址差更好地区分 `P3漏掉的Damaged` 与 `P3正确的Intact`。

预注册 Gate 与当前代码一致：

- 最佳匹配信号 AUROC 比共址信号至少高 0.03；
- 逐事件平均 AUROC 至少 0.60；
- 匹配 Key 的 Event Macro F1 比共址 Key 至少高 0.30 pp；
- Damaged 净互补为正；
- 至少 10/14 事件不下降。

### IM1：匹配感知节点 Key（仅 IM0 通过）

```text
P3语义区域特征
+ SAR场景相对mean/std
+ 局部匹配几何统计
+ 节点几何
→ LN/MLP投影到64维
→ L2 normalize
```

Memory 仍然 train-only、排除同事件、每事件邻居上限 1。先只重跑邻域质量与 RM4 救援/伤害审计，不训练 Cross-Attention。

### IM2：结构化查询单元（仅 IM1 通过）

在 P3 建筑内使用硬、连通、低泄漏的多节点分区。比较：

- P3 整栋区域；
- 建筑内硬超像素；
- 官方实例（若可用）。

唯一变量必须是查询单位，Key 和 Memory 保持不变。

### IM3：联合查询—支持分类（仅 IM2 通过）

```text
P3当前节点Query
+ Top-R跨事件Memory节点
+ label/region geometry
+ null/reject token
→ 轻量Cross-Attention
→ 条件I/D/X校准
```

P3 Background/Building 概率保持不变；最终增量系数初始化为 0；Memory 分支必须可以选择 null，不得强制三分类。

## 六、想法评估

### 被否定的旧版本

“把当前软超像素和硬 kNN 直接接回 P3”已被本地结果反驳：SR1P 几乎无效，软分区未过 Gate，RM6-P 总体净纠正为负。按致命缺陷规则，该版本结论是 **Reject and Pivot**。

### 修订后的新版本

“用跨模态局部匹配几何构造可靠性 Key，再在建筑约束节点上做可拒绝检索增强”尚未被实验否定。

| Dimension | Score | 理由 |
|---|---:|---|
| Higher | 7 | 直接针对 RM6-P 的排序失败，但尚无 IM0 结果 |
| Faster | 7 | 节点检索显著少于像素检索，需报告真实耗时 |
| Stronger | 8 | 局部匹配对配准残差和模态差异有明确机制依据 |
| Cheaper | 8 | IM0 无训练，复用冻结 P3 和现有缓存 |
| Broader | 7 | 可迁移到其他异构变化检测，但严重度仍是 BDA 特有挑战 |

Verdict：**Accept with Revisions**。首要风险是冻结 P3 RGB/SAR 特征未必具备足够的模态不变性；IM0 若失败，应测试 RIFT/相位结构描述子，而不是直接上 Cross-Attention。

## 七、论文逻辑骨架

- Paper type：Technique Paper / cross-domain transfer。
- Research background：BRIGHT 需要灾前光学定位与灾后 SAR 损毁证据，但存在异构辐射、配准残差、长尾严重度和跨事件偏移。
- Existing limitation 1：共址特征差不能稳定表示结构对应是否被破坏。
- Existing limitation 2：像素或整栋平均不能同时保持边界和局部弱损毁。
- Existing limitation 3：硬 Memory 检索没有拒绝能力，救回 Damaged 时误伤更多 Intact。
- Key idea：在光学建筑约束下，以局部匹配几何构造节点级跨事件检索 Key，并让 Memory 作为可拒绝条件证据。
- Challenge A：从 RGB/SAR 异构响应中提取与事件无关的匹配稳定性。
- Challenge B：构建不跨建筑、不过度平均的节点查询单位。
- Challenge C：在保留 P3 的前提下只接受可靠历史支持。
- Module A：hard-peak local matching evidence，对应 Challenge A。
- Module B：building-constrained multi-node representation，对应 Challenge B。
- Module C：null-aware query-support classifier，对应 Challenge C。
- Contributions：匹配感知检索 Key、建筑约束节点 Memory、rescue/harm 与事件留出的安全协议。

四项一致性检查：Limitation→Idea、Idea→Challenges、Challenges→Modules、Modules→Contributions 均通过。

## 八、当前唯一行动

先运行现有 IM0 审计，不修改分类器、不训练融合、不访问 Test。只有 IM0 通过，超像素和 Transformer 才有继续开发的证据基础。
