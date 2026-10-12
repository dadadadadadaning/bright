# 伪实例、对象级建筑损毁与选择性验证论文检索结果

- 检索日期：2026-09-08
- 检索关键词：pseudo instance segmentation；connected components building damage；object-wise voting；segment quality estimation；selective prediction；IoU-aligned loss
- 数据源：IEEE Xplore、CVF Open Access、PMLR、ESSD、IGARSS、arXiv、论文官方 GitHub

## P001 Bitemporal Attention Transformer for Building Change Detection and Building Damage Assessment

- 年份：2024
- 收录于：IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
- 作者：Wen Lu，Lu Wei，Minh Nguyen
- 中文摘要：采用双时相网络与局部交叉注意力处理建筑变化和损毁；其对象级 BDA 流程把灾前建筑提取图做连通域编号，再在每个连通建筑内对像素损毁类别多数投票。
- 有无代码：未证实
- 原文链接：https://doi.org/10.1109/JSTARS.2024.3354310
- PDF：开放获取，未下载
- 相关性说明：这是与 LocCC + hard voting 最直接的正式先例，和当前 S1 高度相似。

## P002 MS4D-Net: Multitask-Based Semi-Supervised Semantic Segmentation Framework with Perturbed Dual Mean Teachers for Building Damage Assessment

- 年份：2023
- 收录于：Remote Sensing
- 作者：Yongjun He，Jinfei Wang，Chunhua Liao，Xin Zhou，Bo Shan
- 中文摘要：以双教师半监督、多任务孪生网络学习建筑定位和损毁分类；推理时使用连通域标记与对象内投票修正损毁类别。
- 有无代码：未证实
- 原文链接：https://doi.org/10.3390/rs15020478
- PDF：开放获取，未下载
- 相关性说明：再次证明“语义建筑图转伪实例，再进行对象级一致性修正”已有先例。

## P003 Building Damage Assessment via Deep Object-based Semantic Change Detection (ChangeOS)

- 年份：2021
- 收录于：Remote Sensing of Environment
- 作者：Zhuo Zheng，Yanfei Zhong，Junjue Wang，Ailong Ma，Liangpei Zhang
- 中文摘要：把对象化分析与深度语义变化检测结合，联合建筑定位和损毁分类，并强调对象先验对建筑级损毁结果的一致化作用。
- 有无代码：有：https://github.com/Z-Zheng/ChangeOS
- 原文链接：https://doi.org/10.1016/j.rse.2021.112636
- PDF：未下载
- 相关性说明：可作为建筑对象先验和对象级损毁分类的领域内基线。

## P004 Weakly Supervised Learning of Instance Segmentation With Inter-Pixel Relations (IRNet)

- 年份：2019
- 收录于：CVPR 2019
- 作者：Jiwoon Ahn，Sunghyun Cho，Suha Kwak
- 中文摘要：从弱监督响应图学习像素位移和类别边界，用连通域、边界与位移线索生成伪实例掩膜。
- 有无代码：有：https://github.com/jiwoon-ahn/irn
- 原文链接：https://openaccess.thecvf.com/content_CVPR_2019/html/Ahn_Weakly_Supervised_Learning_of_Instance_Segmentation_With_Inter-Pixel_Relations_CVPR_2019_paper.html
- PDF：开放获取，未下载
- 相关性说明：适合迁移“拆分粘连 LocCC”的边界/位移思想，而不是直接迁移其 CAM 监督。

## P005 Beyond Semantic to Instance Segmentation (BESTIE)

- 年份：2022
- 收录于：CVPR 2022
- 作者：Beomyoung Kim，YoungJoon Yoo，Chae Eun Rhee，Junmo Kim
- 中文摘要：从语义分割图和点线索生成伪实例，再用中心图、偏移图和在线自修正改进实例掩膜，并分析错误伪标签造成的语义漂移。
- 有无代码：有：https://github.com/clovaai/BESTIE
- 原文链接：https://openaccess.thecvf.com/content/CVPR2022/html/Kim_Beyond_Semantic_to_Instance_Segmentation_Weakly-Supervised_Instance_Segmentation_via_Semantic_CVPR_2022_paper.html
- PDF：开放获取，未下载
- 相关性说明：说明连通域只是伪实例起点，中心和偏移可以用于拆分粘连对象。

## P006 Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation

- 年份：2020
- 收录于：CVPR 2020
- 作者：Bowen Cheng，Maxwell D. Collins，Yukun Zhu，Ting Liu，Thomas S. Huang，Hartwig Adam，Liang-Chieh Chen
- 中文摘要：同时预测语义图、实例中心和像素到中心的偏移，把属于同一中心的像素聚成实例。
- 有无代码：有：https://github.com/bowenc0221/panoptic-deeplab
- 原文链接：https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Panoptic-DeepLab_A_Simple_Strong_and_Fast_Baseline_for_Bottom-Up_Panoptic_CVPR_2020_paper.pdf
- PDF：开放获取，未下载
- 相关性说明：是替代纯连通域、学习可分实例几何的成熟底层方案。

## P007 Prediction Error Meta Classification in Semantic Segmentation

- 年份：2020
- 收录于：IJCNN 2020
- 作者：Matthias Rottmann 等
- 中文摘要：把 softmax 熵、离散度、几何尺寸等像素信息聚合到预测连通段，使用 Logistic/LASSO 判断该段是否与真值相交或回归段级 IoU。
- 有无代码：有：https://github.com/mrottmann/MetaSeg
- 原文链接：https://doi.org/10.1109/IJCNN48605.2020.9206659
- PDF：arXiv 开放版本，未下载
- 相关性说明：与“对预测修改簇提取统计，再用低容量验证器判断质量”最相近，但其目标不是 S2 相对 P3 的 mIoU 边际效用。

## P008 SelectiveNet: A Deep Neural Network With an Integrated Reject Option

- 年份：2019
- 收录于：ICML 2019
- 作者：Yonatan Geifman，Ran El-Yaniv
- 中文摘要：联合训练预测器和选择头，让模型在固定覆盖率约束下拒绝高风险样本，研究风险—覆盖率权衡。
- 有无代码：有：https://github.com/geifmany/selectivenet
- 原文链接：https://proceedings.mlr.press/v97/geifman19a.html
- PDF：开放获取，未下载
- 相关性说明：可迁移覆盖率约束，解释“最多回退 60%–70%”的选择性预测思想。

## P009 The Lovász-Softmax Loss

- 年份：2018
- 收录于：CVPR 2018
- 作者：Maxim Berman，Amal Rannen Triki，Matthew B. Blaschko
- 中文摘要：提出可优化 Jaccard/IoU 的凸扩展替代损失，并指出像素交叉熵、图像 IoU 与数据集级 mIoU并不等价。
- 有无代码：有：https://github.com/bermanmaxim/LovaszSoftmax
- 原文链接：https://openaccess.thecvf.com/content_cvpr_2018/papers/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.pdf
- PDF：开放获取，未下载
- 相关性说明：支撑把簇监督从 corrected-minus-harmed 改成 mIoU 边际效用，但不能直接替代簇级目标构建。

## P010 Building-Guided Pseudo-Label Learning for Cross-Modal Building Damage Mapping

- 年份：2025
- 收录于：IGARSS 2025
- 作者：Jiepan Li，He Huang，Yu Sheng，Yujun Guo，Wei He
- 中文摘要：针对灾前光学—灾后 SAR，先增强建筑提取，再以多模型、测试时增强和低不确定性伪标签训练损毁模型，使用建筑先验约束跨模态损毁映射。
- 有无代码：有：https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping
- 原文链接：https://doi.org/10.1109/IGARSS55030.2025.11243835
- PDF：arXiv 开放版本，未下载
- 相关性说明：与 BRIGHT 模态和任务完全一致，可迁移建筑定位增强、TTA概率与低不确定性伪标签，但数据使用协议需重新审查。

## P011 Advancing All-Weather Building Damage Mapping to the Instance Level: Outcomes and Insights from the 2026 BRIGHT Challenge

- 年份：2026
- 收录于：预印本：arXiv（未证实正式收录）
- 作者：未获取
- 中文摘要：总结 BRIGHT 实例级挑战；最终阶段使用训练中未见灾害事件，领先方法偏向模态专用编码、分阶段或晚融合、光学主导定位，并显示明显跨事件性能下降。
- 有无代码：部分可用：https://github.com/ChenHongruixuan/BRIGHT/tree/master/cvprw26
- 原文链接：https://arxiv.org/abs/2607.22746
- PDF：开放获取，未下载
- 相关性说明：说明如果允许实例标注，任务已经转向真实实例级；如果保持 no-instance 协议，则跨事件验证仍是主要贡献点。

## P012 Insta-BDA: Instance-Aware Building Damage Assessment and Counting via Foundation Model Fusion

- 年份：2026
- 收录于：Remote Sensing
- 作者：未获取
- 中文摘要：将变化检测与基础模型生成的零样本建筑实例结合，通过置信度引导融合获得建筑级损毁和计数，不需要实例级训练标注。
- 有无代码：论文页面关联代码，但当前检索到的仓库主要为其变化检测基础实现，完整 Insta-BDA 代码对应关系未证实
- 原文链接：https://doi.org/10.3390/rs18142347
- PDF：开放获取，未下载
- 相关性说明：提供“用基础模型实例替代 LocCC”的最新路线，但模型版本、计算成本和可复现性风险较高。

---

需要下载开放论文时，可指定编号，例如：`P001,P004-P009`。
