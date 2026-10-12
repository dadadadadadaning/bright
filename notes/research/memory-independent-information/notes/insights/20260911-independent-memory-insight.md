# 跨事件建筑 Memory 独立信息迁移方案

## 用户研究方向和问题

方向为灾前 RGB、灾后 SAR 的建筑损毁评估。当前 P3 已有很强建筑语义，现有完整建筑卡虽然能做建筑级分类，但在条件于 P3 后只带来约 0.0023 AUC；真实、随机、打乱和零支持对照接近，说明 Memory 尚未产生可归因的独立病例作用。

## 核心诊断

1. **来源不独立**：病例卡仍来自同一 BRIGHT、同一 P3 伪建筑区域和相同语义目标；“不同建筑”不等于“新知识”。
2. **表示被压扁**：均值、标准差和少量角色 token 丢失屋顶局部异常、相对位置和损毁混合结构。
3. **检索目标错位**：当前空间接近“类别相似”，真正需要的是“在 P3 已知后，错误残差是否相似”。
4. **跨事件干扰占主导**：SAR视角、散斑、分辨率、材料与灾害机制可能比损毁类别更影响距离。
5. **类别过粗**：Damaged/Destroyed 都是多峰分布，单类近邻和单原型会把不同机制混在一起。
6. **P3天花板较高**：P3 AUC 约0.9355，任何外部模块都必须在条件对照下证明额外价值，而不是只提高未校准基线的F1。

## 可迁移模块

- msGFM/CROMA/SkySense：先进行跨传感器自监督预训练，再构建病例 key。
- SkySense V2：共享语义主干与模态提示/专家并存，不强制完全同构。
- UniMAP：每类多个域/机制原型，保留类内多样性。
- MixMAE/SOMA-1M：把配准扰动与质量清洗放进数据和预训练。
- ProMM-RS：显式输出表示不确定性，低可信检索回退 P3。
- TransOSS：采用跨模态对比学习训练用于“找同一语义对象”的嵌入，而不是直接使用分类特征。

## 建议模型：Residual-Conditioned Factorized Memory

病例卡拆成三个 key 和一个 value：

- Structure key：灾前 RGB 建筑形态与局部 patch token；
- Damage key：灾后 SAR 散射异常与模态内相对变化；
- Quality key：配准、有效面积、SAR噪声与预测不确定性；
- Residual value：训练病例的 `GT分布 - P3分布`，而不是硬类别。

检索分两级：先以 Structure/Quality 找候选，再用 Damage key 重排。读取时只估计 P3 尚未解释的残差；邻居分歧大、相似度间隔小或质量不兼容时回退 P3。

## 落地顺序

1. 运行本地 V2 条件筛选，公平比较 P3-only、真实卡、打乱卡和残差检索。
2. 若仍失败，不调容量；生成保留空间 patch 的新建筑卡，并对外部编码器做冻结筛选。
3. 第一优先筛选 CROMA，但必须处理 BRIGHT 3通道RGB/1通道SAR 与 CROMA 12/2通道输入不匹配；不能复制通道冒充传感器波段。
4. 若外部权重不兼容，采用 BRIGHT meta_train 上的双分支 MAE+对比预训练：RGB/SAR独立重建，共享部分只做弱对齐，加入小范围错位增强。
5. 只有 `P3+真实外部特征 > P3-only` 且 `真实 > 打乱` 后，才做 token 交叉注意力并进入像素级评估。

## 最小实验

- E0：条件独立性与层级残差检索（本轮已提供代码）。
- E1：Frozen encoder screen：P3、CROMA-RGB、CROMA-SAR、CROMA-joint、P3+CROMA、P3+shuffled-CROMA。
- E2：共享/特有分解：完全对齐、弱对齐、概率不确定性门控。
- E3：单原型与每类3/5个机制原型。
- E4：真实token支持、随机支持、打乱支持、零支持与同参数无Memory网络。

统一准入：宏AUC相对 P3-only 至少 +0.01，目标 +0.02；真实支持至少超过打乱支持 +0.01 AUC；至少10/14事件不下降。通过后才评估像素mIoU和fixed S2组合。

## 创新度与迁移难度

- 单独替换外部编码器：创新低，难度中。
- 条件残差检索 + key/value解耦：创新中高，难度中。
- 模态共享/特有概率表征 + 分层病例读取：创新高，难度高。
- 大规模外部数据预训练：创新取决于新任务设计，计算和数据许可风险高。

## 可能问题

- Sentinel预训练权重与高分辨率BRIGHT数据的通道和尺度不匹配；
- 灾前RGB与灾后SAR并非同一时刻，跨模态“同语义”与真实变化相互冲突；
- P3伪建筑掩码仍可能让所谓独立卡继承P3定位错误；
- 训练病例的GT残差只能作为Memory value，绝不能出现在查询key或测试阶段构造中；
- 2026预印本和代码需在正式写作前再次核验收录与可复现性。

## 建议题目

1. **Residual-Conditioned Cross-Event Memory for Optical–SAR Building Damage Assessment**：以P3条件残差为检索value，验证跨事件病例是否提供独立纠错信息。
2. **Factorized Structure–Scattering Memory for Misregistration-Robust Disaster Damage Mapping**：RGB结构粗检索、SAR散射重排、质量门控，面向配准残差。
3. **Uncertainty-Gated Multi-Prototype Case Retrieval for Rare Building Damage Recognition**：每类多机制原型与概率不确定性拒绝，针对稀有损毁。
4. **Foundation-Encoded Spatial Case Memory for Multimodal Building Damage Segmentation**：使用跨传感器预训练patch token构建空间病例库，避免整栋类别广播。

