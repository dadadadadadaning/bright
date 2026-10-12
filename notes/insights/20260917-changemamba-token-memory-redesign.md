# ChangeMamba Token 与选择性 Memory 重设计

- 用户研究方向：灾前 RGB + 灾后 SAR 的建筑损毁分级
- 用户研究问题：当前 P3 特征构建的超像素 Memory 增益较低，是否可改用 ChangeMamba 产生更独立的变化 Token，以及下一步如何验证
- 分析日期：2026-09-17
- 论文来源：ChangeMamba, IEEE TGRS 2024；官方 ChangeMamba/BRIGHT 代码
- 趋势来源：ChangeMamba 官方仓库与论文；当前项目 T0--T3 裸检索、互补性与置信度审计

## 核心判断

当前 Memory 低不是单一原因，也不能简单归结为“P3 Token 太差”。已有结果表明：

1. T3 Direct 的 Macro-F1 高于 T3 Memory，说明同一 Token 在参数分类头中比 kNN Memory 更好，Memory 建库、距离与投票机制本身存在问题。
2. P3 区域准确率为 91.63%，Direct/Memory 约 83.9%。如果直接替换 P3，绝大多数区域会被较弱专家破坏。
3. Memory 能救回 P3 错误区域的 24.38%，但未门控时会破坏 10.79% 的 P3 正确区域；Memory 的角色必须是选择性专家。
4. Memory margin、agreement 和最大相似度存在高置信错误，不能作为单独门控依据。
5. 当前局部对齐 entropy 约 0.996、margin 约 0.005，说明局部对齐没有找到明确对应位置。
6. 当前 T1--T3 虽然排除了 P3 最终概率和融合特征，但 RGB/SAR 内容仍来自 P3 的模态专用中间特征，因此与 P3 共享表征偏差；这不是严格的信息泄漏，却会限制互补性。

因此，ChangeMamba 值得尝试，但应作为独立变化证据分支，不应直接替换 P3，也不应一开始与新 Memory、门控器同时联合训练。

## 论文可迁移模块

| 模块 | 原文/代码位置 | 可迁移方式 | 适配条件 |
|---|---|---|---|
| VMamba 双编码器 | `ChangeMambaMMBDA.encoder_1/encoder_2` | 提取 RGB 与 SAR 的多尺度、分类前特征 | 使用 BRIGHT 权重并冻结，避免改变 P3 基线 |
| 层级 Change Decoder | `ChangeMambaMMBDA.decoder_damage` | 取得最终分类头之前的 128 通道变化特征 | 必须暴露中间特征，不能使用最终类别概率作为 Key |
| 多尺度时空关系 | ChangeMamba change decoder | Stage-2 保留局部细节，Stage-3 提供较高层损毁语义 | 先缓存、后训练检索头；不要端到端同时改变所有模块 |
| 长距离空间上下文 | VSS/SS2D | 给局部 Token 加入建筑周围和长距离上下文 | 它不自动解决 RGB--SAR 配准误差，需要独立对齐审计 |

## 推荐 Token 结构

固定同一套 SLICO 区域和预测建筑切分，只改变内容特征源，保证公平比较。

```text
同一个建筑内部局部区域 R
├── ChangeMamba Stage-2 RGB 特征：mean/std/max/top-k attention
├── ChangeMamba Stage-2 SAR 特征：mean/std/max/top-k attention
├── ChangeMamba Stage-3 RGB/SAR 高层特征
├── Change Decoder 分类前变化特征
├── |RGB-SAR|、乘积、余弦关系
├── Core / Ring-8 / Ring-16 差异
└── 小区域证据：top-10%异常池化，避免损毁被区域均值淹没
```

职责必须分开：

```text
检索 Key：ChangeMamba 内容和变化证据
Memory Value：类别分布、事件来源、原型/样本元数据
可靠性特征：P3 不确定性、Direct/Memory 分歧、邻域密度、OOD 分数
最终动作：Keep P3 / Use Direct / Use Memory
```

P3 概率不进入第一版 ChangeMamba Key，只用于最后的选择器。这样才能检验 ChangeMamba 是否真正提供独立证据。

## 分阶段实验

### 阶段 A：只换 Token，不回写像素

| 实验 | 区域划分 | 内容特征 | 目的 |
|---|---|---|---|
| K0 | 当前 SLICO + P3 建筑切分 | 当前 P3-T3 特征 | 已有参考 |
| K1 | 完全相同 | ChangeMamba Stage-2 | 检验局部细节 |
| K2 | 完全相同 | Stage-2 + Stage-3 | 检验局部与语义结合 |
| K3 | 完全相同 | K2 + Change Decoder 分类前特征 | 检验时空变化特征 |
| K4 | 完全相同 | P3-T3 与 K3 后期双专家 | 检验互补而非简单拼接 |

每个实验都同时报告 Direct Head 与 Memory，不能只报告 Memory。

关键指标：Macro-F1、Damaged Recall、P3 错误恢复率、Memory 独占救回率、P3 正确区域潜在伤害率、正确 Memory 与随机 Memory 差值、最差事件表现。

建议准入：K3 相对 K0 的 P3 错误恢复率至少提高 3pp；Memory 对 Direct 剩余错误的独占恢复率由当前 7.86% 提高到至少 10%；未门控伤害不高于当前 10.79%；最差事件不崩溃。

### 阶段 B：重做检索空间

如果 K3 的 Direct Head 提高而 Memory 没提高，说明 ChangeMamba Token 有效，但检索机制仍有问题。此时再做：

- 跨事件监督对比学习；
- 类别×事件均衡 Memory；
- 同一预测类别、相似面积中的困难负样本挖掘；
- 使用“同类距离与最近异类距离之差”，代替已经饱和的最大余弦相似度；
- 输出局部邻域密度和 OOD 分数；
- 对 Damaged 使用独立采样配额，防止被 Intact 淹没。

### 阶段 C：选择性回写

正式训练数据必须来自 Train 的跨事件 OOF 候选，Val450 只评估，不能用当前带 GT 的 Val 明细训练选择器。

```text
P3 确定
  -> Keep P3

P3 不确定
  -> 比较 ChangeMamba Direct 与 Memory
  -> 三动作低容量 Verifier：Keep / Direct / Memory
  -> 只在高正效用时小幅回写
```

对照：V0=P3；V1=P3+ChangeMamba Direct；V2=V1+ChangeMamba Memory；V3=V2+预测建筑 Bag 证据。只有 V2 在 Val450 显著超过 V1，Memory 才能保留为最终创新。

## 匹配与可行性评估

- 收录信息：ChangeMamba 发表于 IEEE TGRS 2024。
- 有无代码：有官方代码与 BRIGHT 训练/权重入口；本地已存在 ChangeMamba 工程和 P3-transfer 隔离实验。
- 任务匹配程度：高，官方框架包含 BDA，并提供 BRIGHT 权重。
- 创新度：单纯“换成 ChangeMamba 特征”较低；“P3 不确定性驱动的双专家选择 + 独立 ChangeMamba 检索证据”属于中等创新，需要严格互补性和机制实验支撑。
- 迁移难度：中高，主要成本是统一特征网格、缓存多尺度特征、建立 OOF 检索库与避免验证泄漏。
- 主要风险：ChangeMamba 与 P3仍学到相同标签偏差；Mamba 全局上下文可能平滑小损毁；它不天然解决跨模态错位；直接拼接会让检索维度更高、相似度更饱和。

## 建议题目方案

1. **Uncertainty-Guided Dual-Expert Retrieval for RGB--SAR Building Damage Assessment**
   - P3 作为稳定主模型，ChangeMamba Direct/Memory 作为只在不确定区域启用的独立专家。
2. **Building-Conditioned Multi-Scale Change Tokens for Cross-Event Damage Retrieval**
   - 用 ChangeMamba 多尺度分类前特征构造建筑局部 Token，重点验证跨事件检索。
3. **Risk-Aware Selective Memory for Multimodal Disaster Damage Mapping**
   - 将重点放在高置信错误检测、Keep/Direct/Memory 三动作效用选择与安全回退。

## 证据

| 类型 | 来源 | 支撑内容 |
|---|---|---|
| 原文 | https://arxiv.org/abs/2404.03425 | ChangeMamba 使用 Visual Mamba 编码器与三种时空关系机制处理 BCD/SCD/BDA |
| 代码 | https://github.com/ChenHongruixuan/ChangeMamba | 官方代码包含 MambaBDA、BRIGHT 训练入口与权重 |
| 本地代码 | `ChangeMamba-master/changedetection/models/ChangeMambaMMBDA.py` | 双编码器、building decoder、change decoder 与分类头结构 |
| 本地代码 | `bda_benchmark/experiments/spi_factorized_cross_modal_token_v1/build_records.py` | 当前 T1--T3 使用 P3 模态专用分类前特征，并排除最终概率和融合特征 |
| 当前实验 | T3 complementarity audit | Memory 有少量独占救回，但存在明显潜在伤害与置信度失真 |
