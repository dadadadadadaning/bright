# H13-SPER：面向 RGB–SAR 建筑损毁的超像素证据推理

## 摘要

H6-A2 与 H12-D1 的结果说明，当前瓶颈位于 `Intact → Affected` 判断，尤其是弱、中度 `Damaged` 被漏判。H12-D1 的 Stage2 逐像素旁路在测试集只带来约 `+0.0012 pp` Final mIoU，说明浅层证据存在，但逐像素比较仍受到 SAR 斑点、RGB–SAR 成像差异和局部不对应干扰。H13-SPER 将超像素作为介于像素与建筑之间的软证据单元：它聚合局部统计，但不强制整栋建筑或整个超像素输出一个最终类别；最终仍是像素级四类预测。

## 1. 研究问题

1. 超像素区域统计能否比逐像素 RGB–SAR 差异更稳定地识别弱损毁？
2. RGB 引导的边界感知超像素是否优于相同规模的规则网格区域？
3. 一层保守邻接传播是否能补充上下文，而不导致弱损毁过平滑？

## 2. 文献证据与迁移边界

| 方法 | 已验证机制 | 对 H13 的迁移 | 不直接迁移的部分 |
|---|---|---|---|
| [SSN, ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.php) | 可微 SLIC 和软像素—超像素关联 | 每个像素只连接邻近 3×3 网格种子；软池化和回填 | 第一轮不联合学习分区，避免分区与分类共同过拟合验证事件 |
| [SpixelFCN, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_Superpixel_Segmentation_With_Fully_Convolutional_Networks_CVPR_2020_paper.pdf) | 学习边界保持与紧凑超像素 | RGB 颜色距离与空间紧凑性共同决定关联 | 不迁移完整分割网络和额外超像素重建损失 |
| [ESCNet, TNNLS 2021](https://github.com/Bobholamovic/ESCNet) | 超像素可作为变化检测网络的中间表示 | 区域证据进入变化判断后再回填像素 | ESCNet面向同模态二值变化，不能直接承担三级损毁严重度 |
| [CGSL, ISPRS JPRS 2025](https://github.com/TongfeiLiu/CGSL-for-MCD) | 异构影像可在超像素图结构层面比较 | 使用共享分区和稀疏邻接关系，避免全连接注意力 | 不迁移其无监督重建/KL/commonality复合损失 |
| [Structured graph regression, ISPRS JPRS 2022](https://www.sciencedirect.com/science/article/pii/S0924271622000089) | 异构变化可利用局部/全局结构而非直接像素值 | 区域节点保留局部结构与上下文 | 不做跨模态图回归，避免重建损失抹去真实损毁 |
| [SAR superpixel cosegmentation, 2019](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/joe.2019.0193) | 普通光学 SLIC 在 SAR 上会受斑点影响 | 分区由灾前 RGB 主导，RGB/SAR共用同一分区 | 不直接在原始 SAR 上独立生成超像素 |

这些工作共同支持“区域表示有利于噪声抑制和结构比较”，但没有直接证明其能提高 BRIGHT 的 `Damaged`。因此 H13 必须先运行分区审计和规则网格对照，不能把其他数据集的二值变化结果当作本任务增益。

## 3. 方法

### 3.1 共享软超像素

在 Stage2 分辨率上建立规则种子，每个像素只与周围九个种子关联。`S1` 只使用中心网格；`S2/S3` 根据灾前 RGB 颜色与空间距离产生软权重：

\[
q_{pi}=\operatorname{softmax}_i\left(
-\frac{\|c_p-c_i\|_2^2}{\tau_c}
-\lambda_s\frac{\|x_p-x_i\|_2^2}{s^2}
\right).
\]

同一个 \(Q=[q_{pi}]\) 同时用于 RGB 和 SAR 关系证据，避免两种模态各自产生不兼容边界。分区覆盖完整场景；建筑概率只在最终残差注入时作为支持掩膜，因此这里不是“建筑内统一分类”。

### 3.2 超像素证据聚合

RGB/SAR先通过非共享投影与 GroupNorm，再计算：

\[
[R,S,|R-S|,R\odot S,\mu_{3\times3}(|R-S|),
\sigma_{3\times3}(|R-S|)].
\]

每个区域聚合 `mean + std + max`。其中均值抑制 SAR 斑点，标准差描述区域内部不一致，最大值保留小面积强响应，避免弱损毁被均值完全稀释。区域特征还包含 H6-A2 原始 occurrence logit 与建筑重叠率。

### 3.3 保守图传播

`S3` 只在四邻接区域间传播一次。边权由 RGB 区域颜色相似度和建筑重叠兼容度决定；强 RGB 边界或建筑/背景差异会降低传播。图残差有独立零初始化系数，避免在训练初始破坏 S2。

### 3.4 层级安全残差

区域输出回填到像素后，只同时修正 `Damaged` 和 `Destroyed` logits：

\[
z'_d=z_d+\alpha M\delta,\qquad
z'_x=z_x+\alpha M\delta.
\]

因此：

\[
z'_x-z'_d=z_x-z_d,
\]

即 `Damaged/Destroyed` 严重度边界严格不变，H13只研究 `Intact/Affected`。输出层零初始化使 S1/S2/S3 在训练前均严格等价于 H6-A2。

## 4. 受控实验链

| 变体 | 唯一变化 | 回答的问题 |
|---|---|---|
| S0 | 锁定 H6-A2 | 指标与权重加载是否精确 |
| S1 | 规则网格区域池化 | 收益是否仅来自区域聚合/新增参数 |
| S2 | RGB引导软超像素 | 边界感知分区是否优于规则网格 |
| S3 | S2 + 一层边界门控图 | 邻域上下文是否提供额外收益 |

先运行 validation-only 分区审计，再运行 S1/S2。只有 S2 明显优于 S1 才能把贡献归因于“超像素边界”，只有 S2通过准入门槛才运行 S3。

准入门槛：`Damaged IoU +0.30 pp`、`Affected Recall +0.50 pp`、`Damaged gate-miss -0.50 pp`、Event mean/worst不下降、至少10/14事件非负、Final mIoU下降不超过0.10 pp。测试集不参与结构、超参数或权重选择。

## 5. 风险与反证

1. 若 RGB 超像素经常跨越真实 Intact/Affected 边界，区域聚合会产生标签污染；分区审计应直接停止该路线。
2. 若 S1 与 S2相同，收益来自区域统计而不是超像素边界，论文不能宣称边界感知融合有效。
3. 若 S2有效而 S3下降，说明邻域传播发生过平滑，应保留无图的 S2。
4. 若所有变体仍在最优权重中关闭残差，说明当前 Stage2 RGB/SAR关系特征缺少可泛化的新证据，而不是超像素数量不足。
5. 当前实现是 SSN-inspired 的固定 RGB 引导软关联，不是完整可学习 SSN。只有固定分区上限明确后，才值得增加可学习分区和对应消融。
