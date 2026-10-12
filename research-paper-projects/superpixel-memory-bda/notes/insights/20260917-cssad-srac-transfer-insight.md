# CSSAD 与 SRAC 超像素机制迁移到 RGB–SAR BDA 的分析

## 用户方向

以原始 DamageFormer 为父模型，在灾前 RGB + 灾后 SAR 建筑损毁四分类中研究：
局部配准容错、结构区域组织、训练集历史证据检索与安全残差写回。

## 两种超像素用法

### CSSAD（Liu et al., 2020）

- 固定 SLIC 在震后高分辨率光学影像上形成颜色相近、空间邻近的区域。
- 以超像素几何中心提取 20/40/60 多尺度图像块，形成 AlexNet 训练样本库。
- AlexNet先产生像素损毁结果，再以超像素多数投票细化，并去除植被和阴影。
- 这里的“样本数据库”是训练样本集合，不是推理时检索邻居的 Memory Bank。

### SRAC（IGARSS 2026，依据公开框架图和录用消息）

- 由 VMamba backbone、UNet++ 与 differentiable SSN 生成可学习超像素。
- 超像素掩膜作为结构先验，配合建筑掩膜、背景抑制和边界强调，调制像素特征。
- 空间注意力对每个超像素进行加权池化，并保留 raw statistics skip path。
- 区域特征查询预计算 Memory Bank/semantic-neighbor table；图中还显示标签、灾害类型、区域几何等元数据注入，再由 Cross-Attention + FFN 分类。
- 完整论文和代码未从公开搜索中检得，因此Memory条目的精确字段、训练损失和更新方式不能仅凭框架图断言。

## 对用户方案的建议

不直接复制“VMambaSSN + Memory + Transformer classifier”。建议把独立贡献放在：

1. 先用建筑约束局部相关对齐获得配准容错的 SAR 特征；
2. 再用灾前 RGB 结构形成建筑内部局部区域；
3. 对 `[RGB, aligned SAR, abs difference, product]` 做区域池化；
4. Memory 只提供历史类别/残差建议，不直接覆盖 DamageFormer；
5. 由低容量效用 Verifier 综合父模型不确定性、对齐质量、区域质量和检索质量，决定零初始化残差写回。

## Memory契约

- Key：归一化区域内容Embedding，不含GT、事件ID和父模型最终概率。
- Value：区域类别分布或小幅类别残差建议。
- Metadata：类别、事件、图像ID、几何、类别纯度、对齐质量、父模型错误/效用标签。
- 只从Train写入；Val/Test只读；按类别和事件均衡；同图/同事件排除用于审计。

## 最小验证链

1. ROI-LCA必须先证明错位曲线和偏移恢复优于D0。
2. 对齐后Token必须优于未对齐Token。
3. SLICO/学习区域必须分别优于规则网格和打乱区域。
4. 正确Memory必须优于Direct Head、随机标签Memory和打乱邻居。
5. Verifier写回必须提高像素mIoU，并降低D0正确→新模型错误的伤害率。

## 证据

- Liu et al., 2020: https://doi.org/10.5194/isprs-annals-V-3-2020-45-2020
- SRAC录用消息: https://jianghai.ucas.ac.cn/index.php/zh/xshde/xsdte/461-igarss-2026
- IGARSS 2026 accepted papers（paper 2851）: https://2026.ieeeigarss.org/papers/accepted_papers.php
