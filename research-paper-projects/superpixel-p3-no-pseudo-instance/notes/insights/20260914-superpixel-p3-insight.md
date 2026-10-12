# 超像素如何接入P3而不变成伪实例

- 用户研究方向：灾前RGB—灾后SAR建筑损毁像素级分割
- 用户研究问题：使用超像素改善空间一致性和边界，但不把超像素当建筑实例、不生成伪标签
- 分析日期：2026-09-14

## 核心判断

第一版采用固定SLIC作为**无标签的关系图**，不要让每个超像素输出一个类别。P3仍输出每个像素的四类logits，CE+Lovasz和官方mIoU都保持不变。超像素只提供一个辅助残差和一个低权重一致性损失。

## 建议结构：Superpixel Relation Regularizer（SPR）

```text
灾前RGB ──> SLIC区域ID（不含类别、不含实例ID）
                    │
P3 fused_roi特征 ──> 区域内加权平均 ──> 区域上下文
                    │                    │
                    └── 像素原特征 <── 零门控广播残差
                                      │
                                  原P3 clf_2
                                      │
                                 像素级4类输出
```

推荐计算：对区域s中的像素特征`f_i`求均值`r_s`，再把`r_s-f_i`经过两层1×1卷积形成残差。输出为`f'_i=f_i+gamma*g_i*Delta_i`，其中`gamma`初始化为0，`g_i`由建筑概率、超像素边界距离和预测置信度组成。因为`gamma=0`时严格等于P3，可以安全加载P3权重。

辅助一致性损失只约束高可信区域：

```text
L_sp = sum_i w_i * KL(stopgrad(mean_probability_of_region), p_i)
```

`w_i`在建筑边界、超像素边界、预测高熵处降低。它不会产生任何新的标签，也不把区域判成某一类。

## 为什么不属于伪实例

以下四项必须同时满足：

1. 超像素由灾前RGB外观产生，不读取损毁GT；
2. 不给超像素分配Intact/Damaged/Destroyed标签；
3. 不使用超像素多数投票覆盖像素预测；
4. 训练、输出和评测始终是像素级。

## 准备工作

准备量属于中等，主要困难不在模型，而在数据同步。

1. 在完整灾前RGB图上离线生成SLIC ID图，保存为无损`uint16 .npz`；建议先做100、200、400三个粒度。
2. 数据集裁剪、左右翻转、上下翻转、90度旋转时，必须对ID图执行完全相同的几何变换；ID图只能用nearest方式缩放。
3. 离线审计超像素质量：建筑边界召回BR、欠分割误差UE、跨GT类别率、平均区域面积、每图区域数。
4. 超像素仅由train/val各自图像生成；允许为test生成无标签超像素，但在方法冻结前禁止访问test指标。

当前数据加载器把pre/post/label一起增强，尚未携带第四张superpixel ID图，因此需要独立数据集包装器，不能直接修改所有主实验。

## 最小实验

| 实验 | 内容 | 回答问题 |
|---|---|---|
| SP0 | 冻结P3 | 60.502 Val450基线复现 |
| SP1 | P3+参数量匹配1×1残差，无超像素 | 提升是否只是参数变多 |
| SP2 | P3+超像素特征残差，gamma=0初始化 | 区域上下文是否有效 |
| SP3 | SP2+高可信一致性损失 | 关系正则是否进一步有效 |
| SP4 | SP3但打乱超像素ID | 模块是否真正利用空间分组 |
| SP5 | SP3但使用规则网格 | 超像素边界是否优于普通窗口 |

先只跑SP0—SP3的20步smoke和5000步probe。SP2相对SP1、SP3相对SP2有效后，再做SP4/SP5机制对照。

## 准入标准

- Val450相对P3提升至少0.30pp；
- Damaged F1不下降；
- 至少10/14事件不下降；
- bootstrap CI下界大于0；
- 超像素打乱后收益显著减弱；
- 规则网格不能获得同等收益；
- 建筑边界2像素带IoU/F1不得下降。

## 不建议第一版开展

- 不使用超像素多数投票作为最终结果；
- 不把超像素当建筑实例或MIL bag；
- 不一开始使用GNN、Transformer或可学习SSN；
- 不在SAR强度图上单独生成SLIC后强行与RGB一一对应；SAR散斑和错位容易产生不一致边界；
- 不与fixed S2同时加入，否则无法区分收益来源。

## 后续升级

若固定SLIC有效，再把硬ID替换为SpixelFCN/SSN的软分配矩阵；若无效，先检查边界召回和跨类别率，而不是立刻增加GNN。RegionProxy表明区域上下文有价值，但也明确提示固定低层区域噪声可能成为强模型瓶颈，因此“可拒绝、零门控、保留像素支路”是本项目必须保留的安全设计。

## 建议题目

1. **Boundary-Safe Superpixel Relation Regularization for RGB–SAR Building Damage Segmentation**
   - 任务：BRIGHT像素级损毁分割。
   - 方法：灾前RGB超像素关系正则、边界降权、零门控回写。
   - 最小贡献：证明不需要实例标签即可利用区域一致性，并保持官方像素评测。
2. **Building-Aware Region-to-Pixel Context for Cross-Modal Disaster Damage Assessment**
   - 任务：未见事件RGB–SAR损毁识别。
   - 方法：仅在soft建筑ROI内进行区域池化，保留像素主路径。
   - 最小贡献：区域结构与跨事件稳定性的机制验证。
3. **Uncertainty-Gated Superpixel Context for Misregistered Optical–SAR Damage Mapping**
   - 任务：存在局部错位的光学—SAR损毁制图。
   - 方法：高熵/边界位置拒绝区域传播，并做x/y位移审计。
   - 最小贡献：证明超像素上下文在配准误差下何时应被拒绝。

## 证据

- SSN（ECCV 2018）：https://openaccess.thecvf.com/content_ECCV_2018/html/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.html
- SpixelFCN（CVPR 2020）：https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Superpixel_Segmentation_With_Fully_Convolutional_Networks_CVPR_2020_paper.html
- RegionProxy（CVPR 2022）：https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Semantic_Segmentation_by_Early_Region_Proxy_CVPR_2022_paper.pdf
- 异构遥感超像素引导变化检测（JSTARS 2024）：https://doi.org/10.1109/JSTARS.2024.3479703
- 超像素级震后建筑损毁（JAG 2022）：https://doi.org/10.1016/j.jag.2022.102899
- 边缘约束PolSAR超像素（ISPRS JPRS 2025）：https://doi.org/10.1016/j.isprsjprs.2025.08.006

