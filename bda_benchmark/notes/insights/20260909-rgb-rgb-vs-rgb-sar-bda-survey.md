# RGB–RGB 与 RGB–SAR 建筑损毁识别：从“比较同类照片”到“翻译异构证据”

## Abstract

建筑损毁识别不是单一的图像分割问题，而是一条连续的判断链：找到建筑、比较灾前与灾后、判断是否受损、区分受损等级，并把模型迁移到从未见过的灾害。本文围绕三个问题展开：RGB–RGB 和 RGB–SAR 各自最主要的错误来源是什么；常见网络模块究竟增加了什么能力；哪些设计适合迁移到当前 P3 + fixed S2 系统。检索覆盖 2018–2026 年的建筑损毁评估、遥感变化检测和异构光学–SAR 变化检测工作，优先采用论文原文、官方代码和官方挑战报告。综合证据表明，RGB–RGB 的主线已经从直接拼接发展为 Siamese、多尺度、定位–判损解耦、对象一致性和有序分类；RGB–SAR 则额外需要模态专用编码、局部配准容错、共享语义空间和可靠性控制。更大的 Transformer 或 Mamba 可以增强上下文，却不能自动消除模态鸿沟和未见事件偏移。对 P3 而言，下一步最有针对性的能力不是继续堆叠注意力，而是局部容错核对、Affected–Severity 层级概率、校准式修改验证和事件级鲁棒训练。

## 1. Introduction

同一栋建筑在灾前和灾后可能只有很少像素发生明显变化；轻度和中度损毁甚至可能无法从卫星俯视图直接看见。xBD 将任务标准化为灾前、灾后光学影像上的建筑定位与四级有序损毁判断，但数据同时包含不同国家、灾种、建筑形式和成像条件 [1]。BRIGHT 又把灾后图像换为 SAR，使问题从“比较两张照片”变成“比较照片与雷达回波” [10]。

本文回答：

- RQ1：RGB–RGB 与 RGB–SAR 最主要的问题分别是什么？
- RQ2：常见结构与训练方法为模型增加了什么具体能力？
- RQ3：当前 P3 + fixed S2 已经具备什么，还缺什么？

核心判断是：**RGB–RGB 主要学习同一视觉语言中的真实变化；RGB–SAR 必须先建立可比较且容许错位的异构证据，再学习损毁。**

## 2. Methodology

检索按四个视角进行：RGB–RGB 建筑损毁、通用双时相变化检测、光学–SAR 异构变化检测、跨事件泛化与竞赛结果。关键词包括 building damage assessment、xBD、Siamese change detection、optical SAR change detection、BRIGHT、cross-event generalization、ordinal damage 和 instance-level damage。纳入条件为：方法或数据直接关联双时相变化/建筑损毁，且标题、作者和核心结论能由论文原文、出版方页面或官方代码验证。排除只有二手转述、无法核对方法细节、与建筑或双时相判断缺少直接关系的材料。最终重点证据集包含 15 项论文或官方挑战材料。

分类轴不是 CNN、Transformer、Mamba 等网络名称，而是它们试图补齐的能力：空间定位、双时相比较、模态翻译、局部配准容错、损毁层级、对象一致性、可靠性和跨事件迁移。这种分类能避免把“更换骨干”误认为“解决了新问题”。

## 3. Taxonomy

| 判断环节 | RGB–RGB 的主要困难 | RGB–SAR 新增的困难 | 常见能力模块 |
|---|---|---|---|
| 找建筑 | 小目标、边界、密集建筑 | SAR 几何畸变使灾后定位更难 | U-Net/FPN、多尺度、灾前 RGB 定位 |
| 比较前后 | 光照、阴影、季节和视角伪变化 | 两种成像机制导致特征天然不可直接相减 | Siamese/伪 Siamese、特征对齐、局部匹配 |
| 判断受损 | 背景远多于建筑，细微损毁证据弱 | 斑点、叠掩、阴影和入射角会伪装成变化 | 建筑 ROI、模态专用分支、可靠性门控 |
| 判断等级 | Minor/Major 或 Damaged/Destroyed 易混 | 相同等级的 SAR 回波跨事件不稳定 | ordinal/层级 head、类别均衡损失 |
| 保证一致 | 一栋楼被分成多种颜色，相邻楼粘连 | 小错位会污染建筑边界 | 对象投票、软实例池化、中心/offset |
| 迁移新灾害 | 地貌、灾种、建筑风格改变 | 同时发生事件域偏移和传感器统计偏移 | 事件级划分、GroupDRO、适配、校准 |

## 4. RGB–RGB：在同一种视觉语言中找真实变化

### 4.1 从直接堆图到 Siamese 比较

早期的核心进步不是“层数更多”，而是让灾前、灾后图像分别编码后再比较。全卷积 Siamese 网络建立了共享编码、特征拼接和特征差分等基本范式 [2]；xBD 上的多时相融合研究也提示，特征级比较比把两张图直接当作六通道输入更符合任务结构 [3]。共享权重相当于让两张 RGB 图经过同一本“视觉字典”，但它仍会把阴影、季节和轻微错位当成变化。

### 4.2 从一个五分类器到“定位 + 判损”

xBD 的背景像素远多于建筑像素。如果一个 head 同时判断背景、完好和多个损毁等级，最容易先学会背景。ChangeOS 与 DamageFormer 都把建筑定位和损毁判断显式分开：前者加入对象级语义变化框架 [4]，后者使用 Siamese Transformer、时相融合和双任务解码器 [5]。这类设计增加的是“先找到楼，再给楼判级”的能力，而不只是更强的特征提取。

### 4.3 从局部卷积到全局和多尺度上下文

BIT 把整幅特征压缩成少量语义 token，再通过 Transformer 建模时相上下文并回写像素 [6]；ChangeFormer 和 DamageFormer 使用分层 Transformer 捕捉多尺度、长距离关系 [5,7]；ChangeMamba 则用状态空间模型降低大范围关系建模的成本 [8]。这些方法共同解决“只看局部像素看不清整栋结构”的问题，但它们不自动解决标签噪声、实例粘连或未见事件。

### 4.4 从像素碎片到建筑对象

ChangeOS 的对象级后处理代表了一条重要路线：先得到建筑区域，再对同一建筑内的损毁类别进行聚合 [4]。这会减少“一栋楼三种颜色”，但连通域把相邻建筑粘在一起时，硬投票反而会传播错误。因此，对象信息更适合作为软统计、辅助监督或可回退修正，而不是无条件覆盖像素预测。

### 4.5 从普通分类到长尾和有序分类

损毁等级有天然顺序：把完好错成轻损通常比错成完全毁坏更接近。常见 Focal、Dice、类别重采样解决的是少数类不被背景和完好类淹没；Ordinal Class Distance Penalty 进一步按照等级距离惩罚错误 [9]。这些方法改善“怎么学”，但如果图像本身没有可见损毁证据，它们不能凭损失函数创造证据。

## 5. RGB–SAR：先翻译和对齐，再判断损毁

### 5.1 为什么不能照搬 RGB–RGB Siamese

RGB 记录颜色和纹理，SAR 记录微波散射。完好屋顶在 RGB 中可能轮廓清楚，在 SAR 中却会因为材料、朝向、入射角、叠掩和雷达阴影呈现完全不同的亮暗。BRIGHT 明确总结了 SAR 的斑点噪声、斜视、遮挡和几何畸变 [10]。因此，直接共享浅层权重或直接计算 `|RGB-SAR|`，很可能测量的是“传感器差异”而非“灾害差异”。

### 5.2 模态专用编码：给两种传感器配不同的老师

异构变化检测通常采用 pseudo-Siamese，即 RGB 与 SAR 使用不共享或部分不共享的编码器。DA-MSCDNet 以特征变换和域适配约束对齐异构深层特征 [11]；M²CD 用混合专家显式处理不同模态，再通过光学到 SAR 引导和自蒸馏缩小特征距离 [12]。2026 BRIGHT 挑战的两个领先方案也独立采用了模态专用编码和阶段式或后期融合 [14]。这种结构增加的是“分别看懂 RGB 与 SAR”的能力。

### 5.3 配准容错：不要求证据恰好落在同一像素

BRIGHT 虽经过人工配准，仍报告约 1 像素量级的代理平均误差 [10]。对小建筑和边界像素而言，这足以把屋顶与道路错误相减。局部窗口匹配、offset、可变形采样或邻域 cross-attention 的作用，是让一个 RGB 建筑位置到附近 SAR 区域寻找最可信证据，并同时输出匹配置信度。它与全局强行对齐不同：真实倒塌区域本来就不应被拉回“与灾前一致”。

### 5.4 建筑引导与晚融合：用 RGB 管位置，用 SAR 提供灾后证据

BRIGHT 的控制实验显示，灾前 RGB 不仅帮助定位，也能改善损毁等级区分；当高质量灾后光学可用时，光学通常优于仅 SAR，但三模态融合又能进一步改善结果 [10]。Building-Guided Pseudo-Label Learning 先从灾前光学训练建筑提取器，再用集成、测试时增强和低不确定伪标签辅助跨模态损毁映射 [13]。这类方法增加的是“先把证据限制在建筑上，再谨慎读取 SAR”的能力，而非简单增加输入通道。

### 5.5 未见事件：最大的尚未解决问题

BRIGHT 的事件留一实验中，常见基线在零样本跨事件设置下平均 mIoU 低于 40%，明显低于常规监督划分的约 48%–52%；经典 UDA 方法的平均增益有限，伪标签课程方法甚至可能下降 [10]。2026 BRIGHT 实例挑战更直接：领先方案在未见事件上的 mAP 约 0.18，远低于最佳同域 holdout 的约 0.51 [14]。这说明更强骨干、注意力和融合并不能代替事件级训练与评估。

## 6. Synthesis discussion

两条路线最终都趋向“定位—比较—判级—聚合”，但起点不同。RGB–RGB 中，Siamese 主要建立时相比较；RGB–SAR 中，pseudo-Siamese 先保护模态差异，然后才允许高层语义交互。二者都从像素级走向建筑级，但硬实例规则会受粘连和定位错误影响；二者都使用类别均衡，却仍受未见事件的损毁机制变化影响。

因此，模块名称不能代表能力。多尺度模块只回答“不同大小怎么看”；Transformer/Mamba 主要回答“远处上下文怎么看”；Focal/Lovász 主要回答“少数类怎么学”；对象池化回答“同栋楼怎么一致”；局部匹配回答“错一两个像素时去哪找证据”；GroupDRO 或事件均衡回答“不要被某个大事件支配”；校准和 Verifier 回答“什么时候不该相信一次修改”。只有模块与错误机制、验证指标一一对应，才构成可信改进。

### 对当前 P3 + fixed S2 的映射

当前 P3 已具备两个独立 Swin 编码器、多尺度融合、RGB 主导建筑定位、建筑 ROI、RGB/SAR/fused 表征和建筑 bag/token 辅助 head；fixed S2 又增加了 LocCC 区域一致化。因此它已经具备“看两种模态、找建筑、限制建筑区域和做粗对象聚合”的能力。

它仍缺四项被直接验证的能力：

1. 在 LocCC 内允许局部错位的 RGB→SAR 证据匹配；
2. 将 `Building → Affected → Destroyed` 写成条件概率，而不是平面四分类；
3. 对 fixed S2 的每个修改簇作 metric-aligned 接受/回退判断；
4. 以完整事件隔离、事件均衡或 GroupDRO 验证未见事件鲁棒性。

## 7. Open problems and future directions

- 细微损毁可能在俯视 RGB 或单极化 SAR 中都没有足够可观测证据，需要更丰富极化、LiDAR、斜视影像或地面信息，而不是无限加深网络。
- “模态不变”与“保留真实变化”存在张力：对齐过弱时比较不了，对齐过强时会抹去损毁差异。高置信稳定区对齐、变化区保留差异是更安全的方向。
- 语义连通域不是可靠实例。中心/offset、距离变换或真实实例标注值得作为第二阶段，但必须比较其监督成本。
- 事件适配容易误用测试信息。应预先冻结阈值和训练规则，分开 meta-calibration、meta-confirm、Val450 与官方测试。
- 概率熵不等于可靠性。使用 entropy、margin、TTA 或伪标签前，需要报告 ECE、Brier、风险–覆盖率曲线和事件级稳定性。

## 8. Conclusion

对 RQ1，RGB–RGB 的核心是排除同模态伪变化并区分细粒度损毁；RGB–SAR 额外面对成像机制差异、斑点、几何畸变、局部错位和更强的事件域偏移。

对 RQ2，主流改法分别增加了明确能力：Siamese 增加双时相比较，多尺度增加大小兼容，定位–判损解耦减少背景干扰，对象聚合增加整栋一致，ordinal/层级 head 利用等级结构，模态专用编码保护 RGB/SAR 差异，局部匹配增加配准容错，校准与事件鲁棒训练增加可信迁移。

对 RQ3，P3 并不缺另一个通用骨干或注意力。更合理的下一步是让它“在建筑局部容错地核对 SAR、先判是否受损再判严重度、只接受有净效用的 S2 修改，并在完整未见事件上证明稳定”。这比继续堆叠模块更符合现有错误诊断，也更容易形成可归因的论文贡献。

## References

[1] R. Gupta et al., “Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery,” CVPR Workshops, 2019.

[2] R. C. Daudt, B. Le Saux, and A. Boulch, “Fully Convolutional Siamese Networks for Change Detection,” ICIP, 2018.

[3] E. Weber and H. Kané, “Building Disaster Damage Assessment in Satellite Imagery with Multi-Temporal Fusion,” arXiv:2004.05525, 2020.

[4] Z. Zheng et al., “Building Damage Assessment for Rapid Disaster Response with a Deep Object-Based Semantic Change Detection Framework: From Natural Disasters to Man-Made Disasters,” Remote Sensing of Environment, 2021.

[5] H. Chen et al., “Dual-Tasks Siamese Transformer Framework for Building Damage Assessment,” IGARSS, 2022.

[6] H. Chen, Z. Qi, and Z. Shi, “Remote Sensing Image Change Detection with Transformers,” IEEE Transactions on Geoscience and Remote Sensing, 2022.

[7] W. G. C. Bandara and V. M. Patel, “A Transformer-Based Siamese Network for Change Detection,” IGARSS, 2022.

[8] H. Chen et al., “ChangeMamba: Remote Sensing Change Detection with Spatiotemporal State Space Model,” IEEE Transactions on Geoscience and Remote Sensing, 2024.

[9] Y. Tsai and C.-H. Lin, “A Class Distance Penalty Deep Learning Method for Post-disaster Building Damage Assessment,” KSCE Journal of Civil Engineering, 2024.

[10] H. Chen et al., “Bright: A Globally Distributed Multimodal Building Damage Assessment Dataset with Very-High-Resolution for All-Weather Disaster Response,” Earth System Science Data, 2025.

[11] C. Zhang et al., “A Domain Adaptation Neural Network for Change Detection with Heterogeneous Optical and SAR Remote Sensing Images,” International Journal of Applied Earth Observation and Geoinformation, 2022.

[12] Z. Liu et al., “M2CD: A Unified MultiModal Framework for Optical-SAR Change Detection with Mixture of Experts and Self-Distillation,” arXiv:2503.19406, 2025.

[13] J. Li et al., “Building-Guided Pseudo-Label Learning for Cross-Modal Building Damage Mapping,” IGARSS, 2025.

[14] H. Chen et al., “Advancing All-Weather Building Damage Mapping to the Instance Level: Outcomes and Insights from the 2026 Bright Challenge,” arXiv:2607.22746, 2026.

[15] S. Hafner et al., “DisasterAdaptiveNet: A Robust Network for Multi-Hazard Building Damage Detection from Very-High-Resolution Satellite Imagery,” International Journal of Applied Earth Observation and Geoinformation, 2025.
