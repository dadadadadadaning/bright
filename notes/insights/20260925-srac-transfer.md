# SRAC 对 BRIGHT 建筑损毁研究的可迁移设计

研究问题：在现有 DamageFormer/A1 上，如何用超像素聚合与历史样本增强损毁分类？

证据范围：用户提供的四阶段架构图、实验室新闻、IGARSS 2026 官方收录名单。本次未取得论文正文、官方代码或逐模块消融，因此下文区分图示机制与建议实现，不把推测写成论文事实。

## 可迁移模块

| 模块 | 图示位置 | 适配方案 | 成本与风险 |
|---|---|---|---|
| 可学习结构分区 | Stage 1：VMamba、UNet++、Differentiable SSN | 保持现有骨干，先比较固定区域与RGB结构软分区 | 中高；分区不等于建筑实例或局部损毁GT |
| 结构条件特征提取 | Stage 2：Background Dimming、Boundary Highlighting、Modulation | 对比现有特征ROI与独立结构条件分支 | 中；RGB边界不是SAR物理散射边界，不能假定解决配准 |
| 加权区域聚合与统计旁路 | Stage 3：Spatial Attention、Weighted Pooling、Raw Skip | 同一区域比较mean、mean/std、加权pool+统计旁路 | 低至中；注意力可能聚焦伪证据，需参数匹配对照 |
| 检索增强分类 | Stage 4：Memory、邻居表、metadata、Cross-Attention、FFN | 当前区域作为query，训练库邻居特征与已知标签作为support | 中；训练必须排除自匹配，验证禁止使用查询GT |

## 匹配与可行性

任务和模态与用户高度一致。官方名单确认论文编号2851；新闻报告75.65% mIoU及8.08pp提升，但无法据此确定背景是否计入、数据划分、TTA、后处理、骨干贡献或Memory净贡献。本次未找到足以核实细节的正文和官方代码。

图示更接近“学习当前区域如何利用检索到的支持样本完成分类”，没有显示Utility Verifier，不能称其采用先独立投票再安全纠错。图中Semantic Neighbors Table表示检索候选关系，不能直接解释成建筑内部空间邻接图。MMAP Gather应视为存取/检索实现标签，不是损毁语义模块。

## 建议的最小实验

固定A1、区域划分、数据协议、训练预算和输出方式，依次比较：

1. 当前Token，无邻居。
2. 当前Token，加mean/std统计旁路。
3. 当前Token，加注意力pool和统计旁路。
4. 与3同容量的查询独立分类器（无外部邻居）。
5. 真实Train Memory的Cross-Attention分类器。
6. 同一分类器评估时替换随机/打乱邻居；另做标签打乱干预。

训练样本不得检索自身；严格跨事件设置进一步排除同事件。Memory标签仅来自Train支持样本。新查询GT不能用于建区、检索、metadata注入。若骨干更新，训练库特征需按明确策略刷新。

先报告统一区域标签下的Macro-F1、Damaged precision/recall、每事件结果，再报告固定像素输出路径的Valid mIoU和mIoU4。区域分类收益不等价于像素收益。真实邻居必须超过同容量无Memory与随机邻居，才能支持历史证据增益。

## 可发展的题目方向（均为待验证方案）

1. 面向BRIGHT的保留局部差异的区域检索增强建筑损毁评估：借鉴加权pool与统计旁路，增加像素残差保留；比较平均pool、区域广播和像素保留。预期贡献是验证区域一致性与局部细节的平衡。
2. 面向跨事件RGB–SAR损毁评估的支持样本条件分类：借鉴Cross-Attention，加入事件均衡训练支持库；比较同容量无检索、真实与随机检索。预期贡献是证明跨事件历史样本的独立作用。
3. 面向配准残差的结构条件区域检索：借鉴结构条件特征提取，在保持RGB分区时容错采样SAR；比较中心采样、可靠采样及人为位移曲线。预期贡献是辨别结构先验与真实对齐鲁棒性的作用。

以上基础构件已有先例，组合本身不足以证明新颖性；独特机制和受控实验才是论文贡献依据。

## 来源

- IGARSS 2026官方收录名单：https://2026.ieeeigarss.org/papers/accepted_papers.php
- 实验室新闻：https://jianghai.ucas.ac.cn/index.php/zh/xshde/xsdte/461-igarss-2026
- 四阶段模型机制：用户本轮所附SRAC架构图。
