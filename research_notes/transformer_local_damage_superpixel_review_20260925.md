# 面向 RGB–SAR 建筑损毁的局部 Transformer 与损毁证据超像素：文献与本地实现审计

## 摘要

本报告回答三个问题：第一，为什么当前 DamageFormer/Swin 路线容易遗漏小面积损毁；第二，怎样改造 Transformer，使其同时保留局部细节和建筑级语义；第三，在 BRIGHT 只有建筑级损毁标签的条件下，怎样生成和验证“建筑内部损毁证据超像素”。综合文献与本地代码，问题并非 Transformer 天生不适合局部损毁，而是当前网络在早期将影像压缩到低分辨率 Token，随后用插值恢复尺寸，却没有独立的高分辨率损毁证据流；现有超像素分区又主要受到建筑结构重建和建筑级 Bag 标签监督，无法唯一确定建筑内部哪个局部区域对应损毁。推荐采用“高分辨率局部证据流 + 深层建筑语义流 + 建筑约束的可微软超像素 + 区域/建筑双层 Transformer”，并以逐级消融而不是一次性叠加完成验证。

## 1. 研究问题

- RQ1：Transformer 的什么结构导致小损毁信息衰减，问题是否来自注意力本身？
- RQ2：怎样在现有 DamageFormer 上恢复局部细节，而不放弃全局语义？
- RQ3：怎样生成由 RGB 结构约束、由对齐 RGB–SAR 变化证据细化的建筑内部区域，并在缺少局部损毁 GT 时进行可信验证？

## 2. 文献证据形成的判断

### 2.1 Transformer 不是天然不适合局部任务，低分辨率 Token 才是主要风险

HRFormer 明确指出，大 Patch 和单尺度低分辨率表示会损失密集预测所需的细粒度空间信息；其解决办法是贯穿全网保留高分辨率支路、并行维护多分辨率表示，同时在局部窗口注意力后的 FFN 中加入 3×3 depth-wise convolution。SegFormer 和 ChangeFormer采取另一条路线：编码器输出多尺度特征，再由轻量解码器聚合局部与全局信息。三者共同说明，Transformer 可以服务局部预测，但必须提供高分辨率信息通路和多尺度重建，不能只依赖最深层 Token。

另一方面，Plain-DETR 的结果说明“全局注意力一定不适合小目标”也不成立；足够强的预训练和合适的位置偏置可以部分补偿多尺度与局部先验。因此，对 BRIGHT 更准确的判断是：在训练数据、跨事件泛化和细粒度损毁信号有限的条件下，显式局部与多尺度先验更容易训练，而不是 Transformer 在原理上做不到。

### 2.2 局部与全局应分工，而不是只选择其中之一

HRFormer 保持高分辨率并以低分辨率流补充语义；ChangeFormer从四个尺度计算差异并统一解码；Fully Transformer Network 则指出高层特征提供全局语义、低层特征保留局部细节，并使用金字塔聚合和边界监督。Deformable Attention Transformer进一步表明，注意力不必访问全部位置，可以由数据决定少量 Key/Value 采样位置，从而减少无关背景干扰。

这些方法的共同结构可归纳为：

```text
高分辨率局部流：边缘、散射异常、小范围差异
                    ↓
             跨尺度选择性融合
                    ↑
低分辨率语义流：建筑类别、整体损毁、事件上下文
```

这比“让深层特征同时负责定位小损毁和判断整栋建筑类别”更符合任务结构。

### 2.3 可学习超像素的核心不是换成深特征，而是让分区目标与下游任务一致

SSN 将深网络特征输入可微 SLIC，使任务损失能够反向影响超像素；SpixelFCN 将超像素生成转化为像素到规则网格种子的局部关联预测；CLUSTSEG 把区域中心视为查询，以 Cross-Attention 交替更新像素归属和聚类中心。三者都采用“局部候选归属 + 可微聚合”，而不是先生成不可改变的硬分区。

更关键的是，Tu 等人的 segmentation-aware affinity 研究发现，把通用深特征直接送入传统超像素算法并不足以得到更好的语义边界，必须用与分割误差相关的 affinity loss。这个结论直接适用于当前实验：RGB–SAR 深特征本身不等于损毁分区目标。

## 3. 本地 DamageFormer 与超像素实现审计

### 3.1 DamageFormer 已使用多尺度特征，但没有真正的高分辨率损毁解码器

本地 `DamageFormer_S3_BGMP_K3.py` 使用两个 Swin-T 编码器，提取四层特征，将深层特征双线性插值到最浅 Transformer 层的分辨率后拼接，再经 ResBlock 和 1×1 分类头输出，最终直接插值到原图。

优点是模型并非只使用最深层特征；局限是最浅可用特征已经经历约 4 倍 Patch 压缩，随后所有深层特征的上采样只能恢复空间尺寸，不能恢复早期丢失的局部纹理。最终分类头前也没有独立的 1/2 或全分辨率细节解码支路。因此，小损毁可能在进入多尺度融合以前就已经弱化。

### 3.2 V1 可学习超像素更接近“特征同质区域”，不是有监督损毁区域

`damageformer_a1_cross_modal_superpixel_v1/model.py` 在融合特征分辨率上预测每个像素对邻近 3×3 种子的归属；区域内容由 RGB、SAR、绝对差和乘积构成。默认 `region_stride=8`，若特征约为原图 1/4，则种子间距约为 32 个原图像素，小于这一尺度的损毁容易被一个区域吸收。

其 `losses.py` 接收的是建筑定位标签，而不是局部损毁标签。损失包括坐标紧凑性、建筑掩膜重建、关系强度重建、区域均衡和可选建筑边界 affinity。所谓 evidence loss 只要求区域能够重建自身的关系强度，并不要求该关系对应 Damaged 或 Destroyed。因此它能够学习结构同质或变化强度同质区域，却不能据此声称学习到了损毁语义超像素。

### 3.3 V2 加入了建筑级 Bag 监督，但局部归属仍欠定

`damageformer_building_internal_change_partition_v2` 已做出重要改进：预测建筑是硬几何约束；RGB 结构产生基础归属；对齐后的变化证据只以有界残差修改归属；Token 保留 mean/std/soft-peak；建筑损毁 GT 只用于预测建筑 Bag 标签。

但 BRIGHT 的监督主要给出整栋建筑的损毁等级，而不是建筑内部局部损毁掩膜。对于一个 Damaged Bag，模型只知道“至少有某些区域应当支持 Damaged”，却不知道具体是哪一个。Bag 聚合能够训练建筑级分类，却不能唯一确定内部损毁边界；如果对所有区域传播近似相同梯度，就会出现区域语义趋同或只选择最容易的伪证据。

因此，本地实现目前验证的是“区域集合能否帮助建筑级分类”，尚未严格证明“区域边界对应真实局部损毁边界”。

## 4. 推荐模型：建筑约束的局部—全局损毁 Token Transformer

### 4.1 两条特征流

保留现有 Swin 主干作为语义流，同时新增轻量高分辨率证据流：

```text
灾前 RGB ── RGB浅层Stem（1/2分辨率） ─┐
                                      ├─ 安全局部对齐 ─ 局部关系特征 D
灾后 SAR ── SAR浅层Stem（1/2分辨率） ─┘

RGB/SAR Swin多尺度主干 ── 深层建筑语义 S

D ← S进行语义调制，但D始终保留残差直通
```

局部流使用独立模态 Stem，避免用相同早期卷积强迫 RGB 与 SAR 共享低级统计。关系特征至少包含：对齐后 SAR、绝对差、乘积、局部对比、对齐置信度；RGB 外观只提供结构，不应在检索或损毁判别中占据主导。

### 4.2 适合局部损毁的 Transformer Block

在 1/2 或 1/4 分辨率局部流上使用少量局部窗口注意力，而不是全局稠密注意力：

1. 5×5 或 7×7 window attention 建模建筑内部邻域；
2. FFN 中加入 3×3 depth-wise convolution，补充平移与局部连续性先验；
3. 每隔若干层由建筑 Token 与区域 Token 做一次 Cross-Attention，实现局部到整体和整体到局部的信息交换；
4. 使用有效建筑掩膜阻止注意力跨到道路、植被和邻近建筑；
5. 输出只作为零初始化残差加到原模型 logits，避免一开始破坏基线。

这不是把 Transformer 改成 CNN，而是让注意力承担关系建模，让卷积和高分辨率支路承担局部连续性。

### 4.3 RGB结构初始化、RGB–SAR变化细化的软超像素

对建筑内像素 p 和邻近区域种子 k，定义归属 logit：

\[
a_{pk}=a^{struct}_{pk}+\beta_p a^{change}_{pk}-\lambda\|x_p-c_k\|^2.
\]

- `struct`：灾前 RGB 的浅层结构、边缘与建筑内坐标；
- `change`：高分辨率对齐关系特征，而不是原始 SAR；
- `β`：对齐可靠度；低可靠时变化分支自动退回；
- 空间项保证紧凑性；候选仅限同一预测建筑内部的邻近种子。

结构分支决定“不能跨越哪些稳定边界”，变化分支只负责“是否需要在建筑内部进一步拆分”。不建议让 SAR 单独决定初始边界，因为散斑、阴影和叠掩会制造大量伪边界。

### 4.4 区域 Token 与建筑 Token

每个软区域不只做 Mean Pooling，而应保存：

- RGB 结构 mean/std；
- 对齐 SAR mean/std；
- 关系特征 mean/std；
- soft top-k 或 soft-peak；
- Core/Ring 对比；
- 区域面积、质心、与建筑中心距离；
- 对齐置信度和熵。

区域 Token 之间只在建筑内部按邻接图或局部窗口交互；另设一个建筑 Token 聚合整体类别。最终输出分两路：区域残差回写像素；建筑 Token 提供建筑级类别先验。这样既保留小损毁，又维持同一建筑的语义一致性。

## 5. 在 BRIGHT 弱监督下如何训练

### 5.1 可直接使用的监督

1. 官方像素四分类损失：CE + Lovasz，训练最终输出；
2. 建筑定位损失：约束建筑范围；
3. 结构重建与紧凑性：保证区域连通、紧凑并尊重 RGB 建筑边缘；
4. 变化证据重建：防止 pooling 洗掉局部异常；
5. 扰动一致性：SAR 人为平移 ±1/±2 像素后，反向对齐的区域 Token 应与原样本一致。

### 5.2 建筑级 MIL 只能提供弱约束

- Intact 建筑：所有有效区域的损毁分数应低；
- Damaged/Destroyed 建筑：Top-k 或 Noisy-OR 聚合后应支持相应类别；
- 使用区域多样性与覆盖正则，避免所有 Token 完全相同；
- 不应强行把 Damaged 建筑的每个区域都标成 Damaged。

Destroyed 是否意味着更大损毁覆盖率属于额外先验，不能在没有标注核验的情况下当作事实。

### 5.3 证明“损毁感知超像素”需要额外局部证据标注

若论文要声称超像素边界对应建筑内部真实损毁，建议从 Val 中选择跨事件、跨尺度的 100–300 栋建筑，人工标注局部损毁点、粗多边形或损毁核心区域。没有这份小型审计集，只能声称“变化证据感知区域”，不能严格声称“真实损毁区域”。

## 6. 最小而严谨的消融顺序

| 编号 | 设计 | 回答的问题 |
|---|---|---|
| T0 | 冻结协议下的 A1 | 统一基线 |
| T1 | A1 + 1/2 高分辨率卷积细节流 | 小损毁问题是否来自分辨率 |
| T2 | T1 + 局部窗口 Transformer + DWConv-FFN | 局部关系建模是否增益 |
| T3 | T2 + 深层语义对局部流的 Cross-Attention | 全局语义是否帮助排除伪变化 |
| S0 | T3 + 固定 RGB SLICO | 仅区域平滑是否有用 |
| S1 | T3 + 可学习 RGB 结构软区域 | 学习结构是否优于固定 SLICO |
| S2 | S1 + 可靠变化残差细化 | RGB–SAR 变化是否改善内部区域 |
| S3 | S2，但打乱 SAR 或跨图像配对 | 正确跨模态对应是否必要 |
| S4 | S2 去掉 Bag MIL / soft-peak /邻接交互 | 分别验证区域语义、显著证据和关系建模 |

准入顺序必须是 `T1 > T0`、`T2/T3 > T1`、`S1 > S0`、`S2 > S1`、`S2 > S3`。如果前一步不成立，不应继续把后续 Memory 接入。

## 7. 评价指标

最终任务：Valid mIoU、mIoU4、Damaged F1/Recall/Precision、Destroyed IoU、每事件结果、事件宏平均与最差事件。

小损毁：按建筑面积分箱、按受损建筑面积分箱、最小 25% 建筑的 Damaged F1；如有局部审计集，再报告局部损毁 Recall 和 Boundary Recall。

区域质量：建筑边界 Recall、Under-segmentation Error、区域紧凑度、有效 Token 数、区域特征重建误差、Token 多样性。由于 BRIGHT 建筑内部通常只有一个建筑级标签，不能用原始类别纯度证明内部损毁分区正确。

配准鲁棒性：原始 Val 和 SAR ±1/±2/±4 像素扰动下的性能曲线、偏移恢复误差、零偏移中心保持率。

## 8. 结论

RQ1：当前问题主要来自早期空间压缩、缺少高分辨率解码和弱局部监督，而非 Transformer 注意力本身。  
RQ2：应保留独立高分辨率证据流，以局部窗口注意力和卷积 FFN 建模局部连续性，再由深层建筑语义选择性调制。  
RQ3：超像素应由 RGB 结构初始化、由可靠的对齐变化证据做有界细化，并在同一预测建筑内形成软区域。BRIGHT 只提供建筑级标签，因此训练应采用结构约束、证据保持、扰动一致性和 MIL；若要证明区域就是局部损毁，需要额外小规模局部审计标注。

## 参考文献

1. Yuan et al., “HRFormer: High-Resolution Transformer for Dense Prediction,” 2021. https://arxiv.org/abs/2110.09408
2. Xie et al., “SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers,” NeurIPS, 2021. https://papers.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html
3. Liu et al., “Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows,” ICCV, 2021. https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.html
4. Xia et al., “Vision Transformer With Deformable Attention,” CVPR, 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Xia_Vision_Transformer_With_Deformable_Attention_CVPR_2022_paper.html
5. Bandara and Patel, “A Transformer-Based Siamese Network for Change Detection,” IGARSS, 2022. https://arxiv.org/abs/2201.01293
6. Yan et al., “Fully Transformer Network for Change Detection of Remote Sensing Images,” ACCV, 2022. https://openaccess.thecvf.com/content/ACCV2022/html/Yan_Fully_Transformer_Network_for_Change_Detection_of_Remote_Sensing_Images_ACCV_2022_paper.html
7. Jampani et al., “Superpixel Sampling Networks,” ECCV, 2018. https://openaccess.thecvf.com/content_ECCV_2018/html/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.html
8. Tu et al., “Learning Superpixels With Segmentation-Aware Affinity Loss,” CVPR, 2018. https://openaccess.thecvf.com/content_cvpr_2018/html/Tu_Learning_Superpixels_With_CVPR_2018_paper.html
9. Yang et al., “Superpixel Segmentation With Fully Convolutional Networks,” CVPR, 2020. https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Superpixel_Segmentation_With_Fully_Convolutional_Networks_CVPR_2020_paper.html
10. Liang et al., “CLUSTSEG: Clustering for Universal Segmentation,” ICML, 2023. https://proceedings.mlr.press/v202/liang23h.html
11. Chen et al., “Advancing All-Weather Building Damage Mapping to the Instance Level: Outcomes and Insights from the 2026 BRIGHT Challenge,” 2026. https://arxiv.org/abs/2607.22746
