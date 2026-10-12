# 面向 P3 + fixed S2 建筑效用判断的跨区域特征设计

- 用户研究方向：灾前 RGB、灾后 SAR 的建筑损毁语义分割与伪实例纠错
- 用户研究问题：如何增加跨区域特征，同时保持 P3、fixed S2 与 Building Oracle 证据链可归因
- 分析日期：2026-09-06
- 项目依据：P3 + fixed S2、BU-E0 Building Oracle、PI-M0 跨事件 Memory 审计

## 1. “跨区域”必须拆成三个层级

1. 建筑内部跨区域：核心、边界或固定子区域之间交换信息，解决同栋建筑预测碎片化。
2. 同图建筑之间：目标建筑读取邻近建筑的灾害环境与相对异常，解决局部证据弱的问题。
3. 跨图、跨事件：从训练 Memory 检索相似建筑，解决罕见模式与跨场景泛化。

三者解决的问题不同，不能用一个全局注意力模块同时验证。

## 2. 主流可迁移模块

| 路线 | 典型机制 | 对本项目的适配方式 | 主要风险 |
|---|---|---|---|
| 多尺度上下文 | 空洞卷积、金字塔池化、多尺度特征 | 作为参数量匹配的局部/全局对照 | 容易只增加感受野，不形成建筑结构 |
| 全局注意力 | Non-local、自注意力 | 只在少量区域 token 上计算 | 背景占多数、计算大、传播错误 |
| 区域 token | mask/region pooling、region proxy | LocCC 内构造 core/ring 或子区域 token | 伪实例粘连会污染 token |
| 区域图推理 | 邻接图、关系图、图卷积 | 建筑为节点，相对距离和证据差为边 | 灾害空间相关性可能成为事件捷径 |
| 数据集 Memory | 类别原型、检索增强 | 跨事件检索 S2 utility，而不是替代损毁分类 | 同事件泄漏、近重复样本、类别失衡 |
| 跨地域泛化 | 风格扰动、域不变训练、多任务约束 | 作为最终泛化增强，不应先于效用证据审计 | 可能牺牲域内性能，难以归因 |

## 3. 推荐模型：分层选择性区域上下文（HSRC）

保持 P3 与 fixed S2 冻结。对于目标 LocCC 建筑 \(B_i\)，构造：

- 自身 token：P3 语义、S2 行为、RGB/SAR 特征、形状；
- 内部 token：core、ring，必要时才增加固定 Grid/SLIC；
- 邻域 token：同图最近的 K 个建筑，仅使用相对位置和预测证据；
- Memory token：强制排除同事件后检索的训练建筑。

同图邻域聚合：

\[
n_i=\sum_{j\in\mathcal N(i)}\alpha_{ij}W_v h_j
\]

边特征只使用部署时可得信息：归一化距离、方向、面积比、P3 分布差、RGB/SAR 关系差、邻居可靠性。聚合结果不能直接改类别，而只预测：

\[
g_i=P(\text{fixed S2 对建筑 }i\text{ 有益})
\]

最终仍然二选一：

\[
\hat Y_i=g_iY_i^{S2}+(1-g_i)Y_i^{P3}
\]

第一版使用 DeepSets 或一层 attention；残差门零初始化，并在邻居不足或置信度不足时回退 P3/fixed S2 基线。

## 4. 最小可证伪实验

### E0：完全无参数审计

| 编号 | 输入 | 回答的问题 |
|---|---|---|
| C0 | geometry + semantic + S2 behavior | 当前基础上限 |
| C1 | C0 + core/ring 差异 | 建筑内部异质性是否能预测 S2 utility |
| C2 | C0 + 同图邻域统计 | 周边建筑是否提供独立信号 |
| C3 | C0 + 跨事件 Memory | 历史相似建筑是否提供独立信号 |
| C4 | C0 + 全图平均特征 | 排除普通全局场景统计即可解释收益 |
| C5 | C2 的邻居随机置乱 | 排除仅靠额外维度或事件统计 |

每条路线单独达到 AUROC≥0.60、event-macro≥0.58、至少 10/14 事件不低于随机、置换检验 p≤0.05，才进入学习阶段。

### E1：低容量效用预测器

只训练 logistic regression、两层 MLP、DeepSets、一层 masked attention。训练标签来自 meta_train 的 Building Oracle；meta_calibration 只选阈值；meta_confirm 在模型冻结后读取一次。

### E2：联合

只有 C1/C2/C3 中至少两个独立通过，才组合；组合必须分别报告相对 C0 和各单模块的增量。最终还要重建语义图，要求超过 fixed S2、Bootstrap 下界大于 0、至少 10/14 事件非负，并恢复至少 25% 的 Building Oracle 空间。

## 5. 不推荐的直接做法

- 不在 Stage4 全图直接堆 Non-local/Transformer；P3 已稳定，背景和错误区域会参与通信。
- 不让邻居直接投票 Damage 等级；当前已证明的问题是 S2 utility，不是重新分类。
- 不允许同事件 Memory；会把事件风格或重复切片误当成泛化能力。
- 不同时加入 RGID、SLIC、Memory 和图注意力；无法知道收益来源。
- 不在 meta_calibration 或 meta_confirm 上选择 K、层数、阈值以外的大量结构超参数。

## 6. 落地顺序

1. 完成当前 PI-M0；若通过，它就是跨事件层 C3。
2. 新建无参数 C1：core/ring utility audit。
3. 新建无参数 C2：同图邻域 utility audit，强制 leave-one-event-out。
4. 只有单项通过后才实现低容量 Building Verifier。
5. 如果 C1、C2、C3 都失败，先做 RGID-E0 修正 LocCC，而不是增加网络容量。

## 7. 建议题目方案

1. **Cross-Event Retrieval for Selective Pseudo-Instance Correction in Multimodal BDA**
   - 以跨事件 Memory 判断 fixed S2 utility，最终仍在 P3/S2 中安全选择。
2. **Hierarchical Region Context for Risk-Aware Building Damage Correction**
   - 分离建筑内部、同图邻域和跨事件上下文，逐层审计其独立贡献。
3. **Neighborhood-Conditioned Building Utility Verification for RGB–SAR Damage Assessment**
   - 建筑节点读取邻近建筑的相对异常，但只预测 S2 是否值得接受。
4. **Pseudo-Instance Quality Before Graph Reasoning in Multimodal Damage Mapping**
   - 研究 LocCC 粘连/碎裂如何限制跨区域图推理，并以 RGID/SLIC 进行因果对照。

## 8. 原文与趋势证据

- Non-local Networks：用位置间加权聚合直接建模长距离依赖，CVPR 2018。
- GloRe：先把像素投影为少量交互节点，在小图中推理后再广播，CVPR 2019。
- Region Proxy：把区域嵌入作为序列进行区域级 Transformer 建模，CVPR 2022。
- Mask2Former：使用预测 mask 限制 cross-attention 的区域，CVPR 2022。
- Region-Based Representations Revisited：区域表示适用于分割、检索和跨图分析，CVPR 2024。
- Mining Contextual Information Beyond Image：利用数据集级类别 Memory 增强单图特征，ICCV 2021。
- CrossEarth：通过数据级风格注入和多任务训练研究遥感语义分割跨域泛化，2024 预印本。

