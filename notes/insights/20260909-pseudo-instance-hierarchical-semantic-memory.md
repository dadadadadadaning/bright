# 伪实例层级语义 Memory：论文迁移与落地分析

## 研究方向与问题

面向灾前 RGB、灾后 SAR 的建筑损毁评估，研究训练标注样本能否以外部语义 Memory 的形式，提高跨事件稀有损毁类别识别。已有实验表明整栋建筑效用检索信号弱，而修改簇具有更高理论选择空间，因此检索单位需要同时保留伪建筑父级上下文和内部子区域差异。

## 可迁移模块

1. SRAC 图示中的类别标签、灾害类型、区域和几何元数据注入，可迁移为带 provenance 的支持 token。
2. Cross-Image Pixel Contrast 的跨图像像素/区域 Memory 说明局部和区域表征可以互补；本项目采用子簇与父建筑联合查询。
3. Class-wise Memory Bank 的核心价值是按类别维护高质量样本；本项目进一步在每个类别内部平衡事件。
4. Prototype View 提醒单一类别中心会丢失类内多样性，因此每类保留多个实例条目而非一个均值中心。
5. Pin the Memory 强调跨域稳定类别知识；本项目用同事件排除直接测试这种稳定性，而不把事件 ID 当成捷径。

## 发表与代码状态

- SRAC 已出现在 IGARSS 2026 录用列表及团队介绍中；截至实现时未找到公开全文或官方代码，因此当前实现属于基于公开框架图和文字说明的独立迁移，不声称复现。
- Cross-Image Pixel Contrast（ICCV 2021）、Class-wise Memory Bank（ICCV 2021）、Prototype View（CVPR 2022）和 Pin the Memory（CVPR 2022）均有公开论文或官方代码。
- MemorySAM（2025 arXiv/代码）提供多模态语义原型 Memory 的近期参照，但其训练型原型适配与本项目推理检索库并不等价。

## 与当前任务的匹配度

高。已有 D1 缓存已经保存图像、事件、LocCC 标签、修改簇、方向、面积、RGB/SAR Stage2/边缘/纹理特征以及 GT 派生的混淆变化，能够在不重跑 P3 的前提下构建严格的训练标注库。

## 新颖性判断

潜在新颖点不是“使用 Memory”，而是三者组合：异构 RGB-SAR BDA 中的伪实例父子查询、记录级跨事件排除、Occurrence-Severity 分层语义库。如果只做普通 kNN 或类别原型，创新性不足。

## 难度与风险

- 难度：中高。离线机制筛选较容易，最终接入像素主网络需要把区域检索结果稳定地反投影。
- 主要风险：修改簇标签由 GT 只用于建库训练，推理不能访问 GT；事件过滤后稀有类可用邻居可能不足；缓存仅含面积大于既定阈值的簇；检索提升可能只体现在簇分类而不转化为最终 mIoU。
- 失败标准：Memory 相对同编码器无跨事件 macro-F1/稀有类召回增益时，应停止扩容、FAISS 或 Transformer 堆叠。

## 落地建议

先运行 PI-HSM v1 的隔离筛选，以无 Memory 的同模型为对照。如果通过，再把两个分层输出接入 P5-H 的 occurrence/severity decoder，并保持零初始化检索残差；随后做 14 事件 LOEO，最后才进行一次 Val450 评估。不要使用已经消费过的 meta_confirm 再调参。

## 可执行论文题目

1. Provenance-Aware Hierarchical Memory for Cross-Event Multimodal Building Damage Assessment
2. From Pseudo-Buildings to Change Regions: Hierarchical Retrieval for Rare Damage Recognition
3. Event-Exclusive Semantic Memory for Optical-SAR Building Damage Assessment
4. Parent-Child Pseudo-Instance Retrieval for Structurally Consistent Disaster Mapping

## 证据与趋势

语义分割 Memory 已从单一原型逐步发展为跨图像像素/区域库、类别均衡库以及多模态语义原型。共同趋势是保留多个类内样本、显式处理类别不平衡并让检索作为上下文而非硬标签。对本项目而言，关键增量是把域泛化要求写入检索规则：同灾害事件样本不可作为查询支持。
