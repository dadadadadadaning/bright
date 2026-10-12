# 伪实例先例、代码迁移与 P3–S2 后续路线分析

- 用户研究方向：灾前 RGB—灾后 SAR 建筑损毁语义分割
- 用户研究问题：LocCC 伪实例与修改簇验证是否已有先例；哪些公开代码可迁移；除改进伪实例外还有哪些值得改变的环节
- 分析日期：2026-09-08
- 论文来源：BAT、MS4D-Net、ChangeOS、IRNet、BESTIE、Panoptic-DeepLab、MetaSeg、SelectiveNet、Lovász-Softmax、BRIGHT/IGARSS 方案
- 趋势来源：BRIGHT 2026 Challenge 官方页面与总结预印本

## 核心判断

1. 用语义建筑预测的连通域作为近似实例不是新方法。BAT（IEEE J-STARS 2024）与 MS4D-Net（Remote Sensing 2023）均明确使用 CCL 与对象内投票；BAT 的流程与当前 LocCC + S1 尤其接近。
2. 从语义图生成伪实例也有成熟通用先例：IRNet、BESTIE 和 Panoptic-DeepLab 分别使用边界/位移、中心/偏移、自底向上中心聚类改善纯连通域。
3. 使用 softmax 熵、几何和分布统计训练低容量段级质量模型已有 MetaSeg 先例；学习拒绝并约束覆盖率有 SelectiveNet 先例。
4. 当前更可能成立的创新不是任何单项，而是：在冻结跨模态 BDA 主干上，把 fixed-S2 的实际差分按“同伪建筑、同修改方向、8连通”切簇，以数据集 mIoU 边际效用监督低容量选择器，并允许逐簇回退 P3，同时执行跨事件确认。

## 可迁移模块

| 模块 | 来源 | 可迁移方式 | 适配条件 |
|---|---|---|---|
| CCL + 对象内投票 | BAT、MS4D-Net | 作为 S1 既有技术基线 | 不主张新颖性 |
| 边界/位移拆分 | IRNet | 只在粘连风险高的 LocCC 上预测 split evidence | 先做无参数/Oracle 上限，避免全图重训 |
| 中心/偏移实例化 | Panoptic-DeepLab、BESTIE | 使用 meta_train 语义连通域或允许的实例标签训练轻量中心/offset头 | 若坚持 no-instance，必须清楚标注伪监督来源 |
| 段级质量估计 | MetaSeg | 复用概率分位数、熵、margin、几何和边界统计；目标改成 S2-vs-P3 的 mIoU 效用 | 不能直接照搬 IoU=0 标签 |
| 拒绝/覆盖率约束 | SelectiveNet | 在 calibration 选择阈值并限制回退比例 | 需要同时报告风险—覆盖曲线和实际 mIoU |
| IoU 对齐 | Lovász-Softmax | 用于验证旧像素效用与 mIoU 目标错位；后续也可用于 P3 损毁头微调 | cluster utility 仍需基于冻结混淆矩阵定义 |
| 建筑定位增强/TTA 概率 | IGARSS 2025 Building-Guided PLL | 先提高定位概率和校准质量，再重建固定 S2 | 不能照搬使用测试伪标签的竞赛协议 |

## 除伪实例改进外的可行路线

### A. Metric-aligned cluster verifier（最高优先级）

不改变 P3、LocCC 或 S2，只改变“如何给修改簇贴有益/有害标签”。先比较 metric-aligned Oracle 与 pixel-net Oracle，再运行方向/面积规则和 Logistic Regression。成本低、因果归因清楚，最符合当前证据。

### B. 概率校准与多视图稳定性（高优先级）

S2、A6 和 verifier 均依赖 softmax 概率、entropy 与 margin。先在 meta_train 拟合 temperature/vector scaling；再用翻转、尺度或轻微光度变换产生多视图方差，比较校准前后 ECE、NLL、S2 增益和 verifier AUROC。该路线改变的是证据质量而非实例几何。

### C. 定位头/边界质量（高优先级）

S2 的所有区域均来自光学定位。改进定位往往同时减少漏楼、背景污染和粘连。优先试轻量边界辅助头、距离变换头或 UANet 式不确定区域细化；以 localization F1、边界 F1、LocCC merge/split proxy 和最终 S2 mIoU 联合验收。

### D. 损伤类别目标与有序关系（中高优先级）

Intact→Damaged→Destroyed 具有严重度顺序。可在冻结大部分 P3 后，只微调损毁分类头，比较 CE、CE+Lovász、ordinal/rank-consistent loss，并用事件平衡采样保护 Damaged。该路线直接改善 P3 的类别表征，不依赖伪实例。

### E. 训练期区域一致性正则（中优先级）

将 fixed S2 从后处理改为训练期教师：只在高置信建筑内对学生施加软分布一致性，同时保留像素 GT 损失。此前普通蒸馏失败，因此必须先验证高影响方向/大簇是否有可学习信号，不应直接重复全量 S2 蒸馏。

### F. 事件泛化训练（中高优先级）

BRIGHT 的主要瓶颈是未见事件。可比较普通 ERM、事件均衡采样、GroupDRO/最差事件损失、leave-one-event-out checkpoint selection。它改变训练分布而不是网络结构，且应同时报告 event-macro 与 worst-event。

### G. 真实例或基础模型实例（条件路线）

如果研究协议允许使用实例标注，可转向 BRIGHT 2026 的 Mask R-CNN/实例 AP 任务；若仍坚持 no-instance，可将 SAM 类模型只用作冻结 proposal，对 LocCC 做交集/拆分，并以 shuffled/proposal-size 控制验证。该路线成本与协议风险较高，不应与当前 semantic Val450 主线混写。

### H. 跨模态局部匹配/关系模块（低优先级）

项目现有 direct、matching、relation 证据没有通过准入；融合后 FSA/RCM 也失败。因此不建议马上重启大注意力或 GNN。只有在大修改簇、特定方向、Stage2 高分辨率上出现稳定无参数信号，才允许训练轻量模块。

## 建议实验顺序

1. metric targets + metric/pixel Oracle；
2. M2 方向面积规则 + M3 Logistic；
3. 概率校准/TTA 稳定性作为新增统计对照；
4. 若 verifier 空间不足，做定位边界与 LocCC merge/split Oracle；
5. 若错误主要来自 P3 严重度混淆，转向 CE+Lovász/ordinal 的损毁头微调；
6. 所有候选使用事件均衡和 meta_confirm 一次确认；
7. 最后才考虑 DeepSets、中心/offset头或基础模型实例。

## 风险与论文归因

- 不可把 CCL 伪实例、对象内多数投票、段级熵统计或 reject option 单独宣称为原创。
- “修改簇”如果只被定义为预测差异的连通块，本身创新度有限；需要 metric-aligned utility、P3 fallback 和跨事件协议共同构成方法贡献。
- Val450 已多次参与开发，不能被描述成全新未见确认集；新模型应先通过 meta_confirm。
- BRIGHT 2026 已公开实例级任务。若论文声称不需要实例，必须明确这是 semantic-only/no-instance 约束下的可部署设计，而非否认真实实例监督的价值。
- 竞赛伪标签代码可能利用测试图像做自训练，不可直接搬进严格封闭测试协议。

## 建议题目方案

1. **Metric-Aligned Selective Context Refinement for Cross-Modal Building Damage Mapping**
   - 任务：BRIGHT semantic BDA；方法：S2 修改簇 mIoU 效用与选择性回退；最小实验：M0–M7；贡献：指标对齐的安全区域传播。
2. **From Semantic Components to Safe Pseudo-Buildings for Event-Generalized Damage Assessment**
   - 任务：未见灾害事件；方法：LocCC 风险诊断、修改簇选择和事件均衡确认；最小实验：leave-one-event-out；贡献：跨事件稳定的无实例对象化后处理。
3. **Boundary-Calibrated Pseudo-Instance Refinement for RGB–SAR Building Damage Segmentation**
   - 任务：语义建筑粘连与损毁一致性；方法：轻量边界/距离头 + fixed S2；最小实验：CCL、watershed、boundary split；贡献：减少伪实例传播错误。
4. **Risk-Coverage Calibration of Structural Context for Multimodal Damage Segmentation**
   - 任务：S2 有益/有害修改选择；方法：MetaSeg 式段级统计 + SelectiveNet 式覆盖约束；最小实验：风险—覆盖曲线与 mIoU；贡献：可解释的部署安全边界。

## 证据文件

- 检索列表：`searches/20260908-pseudo-instance-cluster-verification.md`
- 结构化记录：`searches/20260908-pseudo-instance-cluster-verification.jsonl`
- 项目直接证据：`experiments/spi_cluster_detail_d1_v1/README_ZH.md`、`experiments/spi_s2_cluster_evidence_v1/README_ZH.md`
