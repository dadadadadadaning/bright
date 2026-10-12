# HeteCD 与 DSTCD 向 BRIGHT 建筑损毁评估迁移分析

- 用户研究方向：灾前 RGB + 灾后单通道 SAR 的建筑损毁分级
- 用户研究问题：3D 差异模块是否值得迁移；DSTCD 的少样本能力来自哪里；如何最小风险接入现有 H6-A2/CCU-BDA
- 分析日期：2026-08-18
- 论文来源：P001 HeteCD、P002 DSTCD 全文与官方代码

## 核心判断

HeteCD 的 3D-STA 是“二值变化差异提取器”，主要发力点是将光学/SAR特征按伪时间维堆叠后进行多尺度 3D 卷积和时空注意；其数据是 RGB + 三极化 SAR，目标是建设/拆除二值变化。（HeteCD PDF p.5 Sec.4.2、Fig.3；p.6 Sec.5.1）

DSTCD 的少样本增益来自“冻结的配准源任务特征 + 任务引导迁移”，不是 CDFE 输入头本身。STFTB 对规则结构变化贡献更明显，SEFTB 对依赖上下文的水体变化更明显，完整二者总体最好。（DSTCD PDF pp.3-6 Sec.2.1-2.5、Fig.2-4；pp.10-11 Table 6）

## 论文可迁移模块

| 模块 | 原文位置 | 可迁移方式 | 适配条件 |
|---|---|---|---|
| HeteCD 3D-STA | HeteCD p.5 Sec.4.2、Fig.3 | 暂不进入主实验；仅作为 stage4 小型对照 | 必须先证实比同位置差分有纠错增益，而不只是损毁相关性 |
| HeteCD 非共享编码 | HeteCD p.4 Sec.4.1、Fig.2 | 保留 RGB/SAR 模态专属参数 | 单通道 SAR 需专属输入与预训练 |
| DSTCD 配准预训练 | DSTCD pp.2-3 Sec.1.3、2.1 | 在独立阶段预训练/加载 SAR-optical 对应特征 | 源数据与单通道 SAR 成像域需要核实 |
| DSTCD STFTB | DSTCD pp.4-5 Sec.2.4.1、Fig.3 | 将冻结配准浅层特征作为 SAR 专家初始化或旁路 | 不应改变 P3 RGB 定位路径 |
| DSTCD SEFTB | DSTCD pp.5-6 Sec.2.4.2、Fig.4 | 仅在 stage4 对建筑区域做语义迁移 | 必须零初始化或受控残差接入 |

## 为什么不优先移植完整 3D-STA

1. 输入不匹配：HeteCD 使用三极化 SAR，BRIGHT 是单通道 SAR。（HeteCD p.6 Sec.5.1）
2. 目标不匹配：HeteCD 学 changed/unchanged，BRIGHT 需要 intact/damaged/destroyed；3D-STA 最容易强化强结构变化，不保证改善 Damaged。（HeteCD pp.3-4 Sec.3.2、4.1）
3. 实验已给出反证：当前 CCU E3 的局部差异信号具有损毁相关性，但对 A2 下游分割只有约 0.003 pp 级提升，说明“相关”不等于“能纠错”。
4. 完整 3D-STA 在四尺度运行且无安全锚定，可能再次破坏 P3/A2 已学决策边界；HeteCD 自身也承认计算开销与细微变化漏检。（HeteCD pp.7-8 Sec.5.3；p.10 Sec.6）

## DSTCD 少样本能力的发力模块

### 主要发力：源任务预训练

先通过配准任务学习跨模态结构/语义，目标阶段冻结，减少 BRIGHT 小样本需要学习的内容。（DSTCD pp.2-3 Sec.1.3、2.1；p.7 Sec.3.2）

### 第二发力：STFTB/SEFTB 的选择性迁移

STFTB 用余弦相似度筛选源结构信息；SEFTB 以目标变化特征为 Query，从光学和 SAR 源语义中检索信息，而不是直接拼接全部源特征。（DSTCD pp.4-6 Sec.2.4、Fig.3-4）

### 辅助发力：双输出监督

同时监督迁移前 output1 和迁移后 output2，避免目标网络只依赖源特征。（DSTCD p.6 Sec.2.5、Eq.9）

## 最小可行迁移实验

### T0：锁定 H6-A2

不修改模型，作为原生 SAR 对照。

### T1：只迁移配准预训练权重

- 为单通道 SAR 编码器 stage1-4 加载配准/跨模态预训练权重；
- 不加 STFTB、SEFTB、3D-STA；
- 保持 P3 RGB 定位冻结；
- 先验证 SAR 表征迁移本身是否改善 Damaged 和跨事件 Event mIoU。

### T2：只加深层语义迁移

- 基于 T1；
- 只在 stage4 加一个简化 SEFTB；
- 零初始化输出投影，以保证 step0 等于 T1；
- 只使用原有损毁监督，不增加新损失。

### T3：再评估浅层结构迁移

- 仅当 T2 有稳定增益，加入 stage3 STFTB；
- 建筑掩膜外禁止注入；
- 与“直接拼接”和“同位置差分”对照。

### T4：3D-STA 作为末位消融

- 只在 stage4、低通道数运行；
- 与普通 `1×1/3×3` 差异头参数量匹配；
- 若不能显著提高 Damaged、Event mean 与跨事件方向一致率，则停止。

## 关键评估门槛

- 首先看 Damaged IoU 与 Damaged→Intact；
- 同时看 Event mean/worst 和逐事件正收益数量；
- 验证集选权重，测试集只做验证锁定候选的最终审计；
- 任何迁移模块必须通过 step0 等价 guard；
- 单次提升小于常规波动时必须做多种子，不能解释为有效创新。

## 建议题目方案

1. **Registration-Initialized Modality-Specific Networks for Few-Shot RGB-SAR Building Damage Assessment**
   - 配准预训练初始化原生 SAR 专家，光学分支负责定位，深层语义负责损毁。
2. **Change-Conditioned Cross-Task Transfer for Pre-Event Optical and Post-Event SAR Damage Mapping**
   - 将 DSTCD 的全局迁移改为建筑区域内、变化条件的深层迁移。
3. **Safe Cross-Task Structural-Semantic Transfer for Cross-Event Building Damage Generalization**
   - 零初始化残差注入和跨事件评估作为主要设计与实证贡献。

## 主要风险

- DSTCD 论文对 SEN1-2/MegaDepth 源任务描述不一致，必须重新训练可追溯的源权重。
- DSTCD 仍会漏掉部分损毁建筑，不能期待配准预训练自动解决 Damaged→Intact。（DSTCD p.13 Sec.4.4）
- 如果 T1 不改善跨事件结果，说明瓶颈不只是 SAR 初始化，应停止继续添加迁移模块。
