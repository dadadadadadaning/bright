# RGB–SAR建筑损毁任务中的超像素生成与本地代码审计

## 1. 最终结论

你的目标可以写成：

> 先由灾前RGB建立稳定的建筑结构分区，再由局部对齐后的RGB–SAR变化证据，在可靠性约束下细化像素到区域的归属，使区域既遵守建筑结构，又能保留局部损毁差异。

这个目标是合理的，但不应理解为顺序生成两套彼此独立的硬超像素。推荐只维护一套共享的软区域归属：

\[
Q=\operatorname{softmax}(L_{\text{RGB-structure}}+
\alpha(x)\,\Delta L_{\text{RGB-SAR-change}})
\]

- `L_RGB-structure`：RGB结构分支给出的基础归属；
- `ΔL_RGB-SAR-change`：跨模态变化分支对归属的修正；
- `α(x)`：由对齐置信度、匹配熵和建筑ROI决定的有界门控；
- `Q`：最终共享给RGB、SAR和Memory使用的区域归属概率。

超像素本身仍然不是损毁类别。“损毁感知超像素”是指区域边界和归属受到损毁任务监督，能尽量避免把Intact与Damaged/Destroyed证据混在同一区域中。

## 2. 文献中的主要生成路线

### 2.1 固定、任务无关的超像素

SLIC/SLICO在颜色与空间坐标的联合空间中聚类。SLICO只自适应调节紧凑度，仍不学习损毁语义。它适合提供稳定、低成本的结构先验。

CSSAD属于这一类：先在震后高分辨率RGB图像上运行SLIC，再以超像素中心构造多尺度AlexNet样本，最后用超像素多数统计细化粗分类。超像素不由分类损失学习。

### 2.2 学习外观特征，再运行超像素算法

SEAL学习适合边界的像素嵌入，但官方实现仍使用ERS等离散算法产生最终超像素。它是“学习特征 + 非端到端分区”。

### 2.3 端到端可学习软超像素

- SSN：网络学习像素嵌入，再进行可微软聚类，可由下游任务损失训练出任务特定超像素。
- SpixelFCN：在规则种子网格上直接预测每个像素属于周围3×3种子的9路概率。
- AINet：进一步显式建模像素与邻近网格种子的关系，并加入边界感知损失。

这类方法的核心并不是“先输出一个类别再切块”，而是预测软归属矩阵 `Q(pixel, local_seed)`。

### 2.4 多时相/变化感知超像素

遥感变化检测中常见三种方式：

1. 在单幅结构较清晰的影像上生成超像素，再把同一区域映射到另一时相；
2. 从双时相构造图、差异图或合成图，在合成表示上进行共分割；
3. 固定超像素，后续在区域图上学习变化关系。

SAR共分割工作会用强度、位置和边缘改造SLIC距离，并在构造图上同时分割SAR影像对。另一些变化检测工作直接在差异图上生成超像素。它们说明变化证据可以参与分区，但SAR散斑和跨模态差异使“直接把原始RGB和SAR拼接后做SLIC”风险很高。

## 3. 本地代码到底实现了什么

### 3.1 固定RGB SLIC/SLICO路线

#### SPI Superpixel Relation Occurrence V1

- 输入：灾前RGB；
- 生成：`skimage.segmentation.slic`；
- 是否学习：否；
- SAR如何使用：同一RGB区域ID用于汇总P3的RGB/SAR关系特征；
- 输出用途：Occurrence残差，不改变四类标签定义。

#### SPI Superpixel Exemplar Memory V2

- 分区仍来自灾前RGB SLIC；
- RGB与SAR先做局部容错对齐；
- 在固定区域内统计关系特征、均值、标准差、最大值、分位数、core-ring差异、几何和对齐质量；
- GT只用于给可靠训练区域赋Intact/Damaged/Destroyed标签并按纯度筛选；
- Memory保存学习后的Key、类别、事件、图像、纯度和面积，不反向改变超像素。

因此它是“RGB结构区域 + 跨模态区域描述符 + Memory”，不是“RGB–SAR学习损毁超像素”。

#### SPI SRAC Mechanism Ablation V1

- 仍由灾前RGB生成SLICO；
- 再用预测建筑概率切开建筑与背景；
- 固定相同区域，对B2/ChangeMamba变化特征测试mean、mean/std/max、注意力和异常尾部池化；
- Memory和Router发生在区域生成之后。

### 3.2 RGB引导的软区域，但不学习损毁边界

`DamageFormer_RGBSAR_H13_SPER.py`中的 `RGBGuidedSoftSuperpixels`：

- 像素只在周围3×3规则种子中选择；
- 权重由标准化RGB颜色距离和空间距离计算；
- RGB在入口处被 `detach()`；
- 没有可学习的区域生成网络参数；
- 随后才在这些RGB区域内汇总RGB–SAR关系的mean/std/max。

这比硬SLIC更适合GPU池化，但仍是RGB定义区域、SAR提供内容。

### 3.3 一步式RGB–SAR联合软分区

`damageformer_a1_cross_modal_superpixel_v1`：

- `c2_slico`：固定灾前RGB SLICO；
- `c3_rgb_assoc`：只用RGB结构学习软分区；
- `c4_raw_sar_assoc`：RGB结构与未对齐SAR关系直接共同预测9路归属；
- `c5_aligned_assoc`：RGB结构与置信度门控后的对齐SAR关系共同预测9路归属。

在`c5`中，`rgb_structure + relation + edge + building`被一次性拼接送入`association_head`。它不是“RGB先生成基础超像素，再由RGB–SAR修正”，而是“一步联合决定分区”。

`damageformer_a1_joint_soft_partition_v1`和`damageformer_roi_soft_partition_v1`都锁定为`c5_aligned_assoc`，机制没有改变，只是训练策略不同。

### 3.4 真正接近目标的两阶段结构—变化分区

`damageformer_building_internal_change_partition_v2`最接近你的正确设想：

1. 从P3预测建筑概率得到建筑实例约束；
2. RGB结构、RGB边缘和建筑内部坐标生成`structural_logits`；
3. RGB与局部对齐SAR形成关系特征；
4. 变化分支输出`delta_logits`和门控；
5. 使用零初始化、有上限的残差修正结构归属：

\[
L=L_{struct}+\alpha_{max}\tanh(g)\sigma(G(x))\Delta L_{change}
\]

6. 最终软归属用于汇总RGB、对齐SAR、关系均值/标准差/峰值、对齐质量和区域面积。

但必须注意：当前V2并没有像素级“损毁边界重建”监督。变化阶段主要使用关系证据保持、残差正则和建筑bag弱标签；因此它是“损毁任务相关的弱监督细化”，还不能直接宣称已经学到了准确的Damaged/Destroyed局部边界。

### 3.5 本地SRAC-Controlled路线

`DamageFormer_P3_SRAC_Controlled.py`中的可学习区域主要由冻结RGB特征生成；边界版本额外加入预测建筑先验和RGB边缘。区域生成后，才把RGB、SAR、差异、乘积和融合特征池化并分类或检索。因此它更接近公开SRAC图中的“结构超像素 + 跨模态区域表征”，并不是两阶段损毁边界细化。

## 4. 为什么不能简单生成两套硬超像素

如果先得到RGB区域 `S_rgb`，再独立生成一套RGB–SAR区域 `S_change`，会出现：

- 两套区域数量和编号无法一一对应；
- Memory中同一位置的Key失去稳定地址；
- SAR噪声可能把建筑内部切得很碎；
- 训练早期跨模态特征不稳定，区域本身随迭代剧烈漂移；
- 很难证明收益来自结构、变化证据还是区域数改变。

推荐让RGB结构分区作为锚，RGB–SAR只修正共享软归属。这样既能保持可解释性，也能用`alpha=0`严格复现父模型或结构基线。

## 5. 推荐的新统一设计

### Stage A：RGB结构初始化

输入：灾前RGB编码特征、RGB边缘、预测建筑ROI、建筑内归一化坐标。

输出：局部3×3候选种子的结构归属logits。

监督：空间紧凑、RGB结构重建、建筑边界不跨越、连通性。

### Stage B：对齐后的损毁感知细化

先在建筑ROI内做3×3局部RGB–SAR相关搜索，得到对齐SAR、匹配熵和置信度。变化分支输入不应只用绝对差，还应包含：

- RGB结构特征；
- 对齐SAR特征；
- 有符号差与绝对差；
- 共同响应/相关性；
- 局部变化梯度；
- 匹配置信度与熵；
- 父模型不确定性（只作门控，不直接作为检索Key）。

变化分支只输出对结构归属的有界残差。

### Stage C：任务监督

要真正称为“损毁感知超像素”，至少需要一项直接证明损毁语义边界被保留的目标：

- 四类GT的软重建损失，但降低Background/Intact多数像素权重；
- Intact–Affected边界的邻接对比损失；
- 小损毁区域加权的混合块重建损失；
- 区域纯度/Oracle上限作为评估，而不是把多数投票当最终预测。

建议第一阶段只监督`Intact`与`Affected`边界，严重度仍由像素分类头决定，避免小片Damaged被区域多数类别抹掉。

### Stage D：统一Token与Memory

最终共享归属`Q`同时池化：RGB结构、对齐SAR、跨模态关系、max/Top-k/P90小面积异常、对齐质量和几何。Memory只读取这些Token，不参与第一版区域生成。只有区域消融通过后，再让检索损失以较小权重反向影响Token编码器，而不要立刻反向改变分区。

## 6. 必须执行的因果消融

| 编号 | 分区方式 | 目的 |
|---|---|---|
| P0 | 规则方格 | 排除普通局部平滑 |
| P1 | 固定RGB SLICO | 固定结构基线 |
| P2 | 学习RGB结构分区 | 验证可学习结构是否优于SLICO |
| P3 | RGB+未对齐SAR一步联合分区 | 证明原始SAR是否污染边界 |
| P4 | RGB+对齐SAR一步联合分区 | 一步联合对照 |
| P5 | RGB结构→对齐变化残差细化 | 推荐主模型 |
| P6 | P5但打乱/平移SAR | 因果负对照 |
| P7 | P5但关闭置信门控 | 验证可靠性作用 |
| P8 | P5但无损毁边界监督 | 验证任务监督作用 |

分区指标必须先于最终mIoU报告：Boundary Recall、Under-segmentation Error、Achievable Segmentation Accuracy/Oracle Region Upper Bound、Region Purity、小损毁覆盖率、平均区域数与连通分量数。然后再报告四类mIoU、Damaged F1、Destroyed IoU、Affected Recall、每事件/建筑尺寸结果，以及人工SAR平移下的稳健性。

判定逻辑：

- P2>P1：学习RGB结构有价值；
- P5>P2：RGB–SAR确实改善了分区，而不只是分类；
- P5>P4：结构锚定优于直接混合；
- P5>P6且P5>P7：真实对齐证据和可靠性门控必要；
- 分区Oracle提高但最终mIoU不提高：问题在区域表征/读出；
- 分区Oracle不提高：不要继续扩Memory，先修正区域生成。

## 7. 最准确的一句话介绍

> 本方法不是分别生成RGB超像素和SAR超像素，而是以灾前RGB建立建筑内部结构锚点，再利用局部对齐且置信度受控的RGB–SAR变化证据，对像素—区域软归属进行有限修正，从而形成兼顾结构完整性与局部损毁敏感性的共享区域Token。

