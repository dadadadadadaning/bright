# P3 未解决漏判的相关论文检索结果

- 检索日期：2026-09-14
- 检索关键词：optical SAR change detection, change detection Mamba, deformable alignment, boundary segmentation, point refinement
- 数据源：CVF Open Access、arXiv、官方 GitHub、IEEE/IGARSS 论文代码页、期刊页面

## P001 ChangeMamba: Remote Sensing Change Detection With Spatiotemporal State Space Model

- 年份：2024
- 收录于：IEEE Transactions on Geoscience and Remote Sensing
- 作者：Hongruixuan Chen, Jian Song, Chengxi Han, Junshi Xia, Naoto Yokoya
- 中文摘要：提出面向遥感变化检测的状态空间模型，并给出二值、语义与建筑损毁评估版本，以多尺度时空关系建模变化信息。
- 有无代码：有：https://github.com/ChenHongruixuan/ChangeMamba
- 原文链接：https://arxiv.org/abs/2404.03425
- PDF：开放（arXiv）
- 相关性说明：可借鉴其多尺度时空变化解码，不应直接替换当前 P3 主干。

## P002 A Transformer-Based Siamese Network for Change Detection

- 年份：2022
- 收录于：IGARSS 2022
- 作者：Wele Gedara Chaminda Bandara, Vishal M. Patel
- 中文摘要：以孪生 Transformer 编码两时相影像并融合多尺度差异，用于遥感变化检测。
- 有无代码：有：https://github.com/wgcban/ChangeFormer
- 原文链接：https://github.com/wgcban/ChangeFormer
- PDF：未获取
- 相关性说明：可作为“多尺度变化证据进入主网络”的结构参照与对照基线。

## P003 Deformable ConvNets v2: More Deformable, Better Results

- 年份：2019
- 收录于：CVPR 2019
- 作者：Xizhou Zhu, Han Hu, Stephen Lin, Jifeng Dai
- 中文摘要：通过学习采样偏移和调制系数，使卷积的有效采样区域适应目标的几何变化。
- 有无代码：有：https://github.com/msracver/Deformable-ConvNets
- 原文链接：https://openaccess.thecvf.com/content_CVPR_2019/html/Zhu_Deformable_ConvNets_V2_More_Deformable_Better_Results_CVPR_2019_paper.html
- PDF：开放（CVF）
- 相关性说明：可作为 RGB 锚定的局部 SAR 对齐模块候选，专门处理经审计确认的错位问题。

## P004 PointRend: Image Segmentation as Rendering

- 年份：2020
- 收录于：CVPR 2020
- 作者：Alexander Kirillov, Yuxin Wu, Kaiming He, Ross Girshick
- 中文摘要：优先在不确定且高频的点位细化预测，以较低成本改善边界细节。
- 有无代码：有：https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend
- 原文链接：https://openaccess.thecvf.com/content_CVPR_2020/papers/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.pdf
- PDF：开放（CVF）
- 相关性说明：可用于边界与小损毁的稀疏细化，不应直接广播区域类别。

## P005 Boundary Loss for Highly Unbalanced Segmentation

- 年份：2019
- 收录于：MIDL 2019；扩展版发表于 Medical Image Analysis 2021
- 作者：Hoel Kervadec, Jihene Bouchtiba, Christian Desrosiers, Eric Granger, Jose Dolz, Ismail Ben Ayed
- 中文摘要：将边界距离写成可与常规区域损失组合的损失项，以缓解前景极少时的训练失衡。
- 有无代码：有：https://github.com/LIVIAETS/boundary-loss
- 原文链接：https://arxiv.org/abs/1812.07032
- PDF：开放（arXiv）
- 相关性说明：可作为训练期 ring/small-damage 监督候选；不构成新的推理期证据。

## P006 M²CD: A Unified MultiModal Framework for Optical-SAR Change Detection with Mixture of Experts and Self-Distillation

- 年份：2025
- 收录于：预印本：arXiv（未证实正式收录）
- 作者：Ziyuan Liu, Jiawei Zhang, Wenyu Wang, Yuantao Gu
- 中文摘要：针对光学与 SAR 的异质分布，以模态专家和光学到 SAR 引导路径进行跨模态变化检测。
- 有无代码：未证实
- 原文链接：https://arxiv.org/abs/2503.19406
- PDF：开放（arXiv）
- 相关性说明：为“模态非对称编码与自适应融合”提供直接的光学-SAR参考。

## P007 Joint Learning of Frequency-Spatio-Temporal Dependencies and Structural Context for Optical-SAR Change Detection

- 年份：2026
- 收录于：International Journal of Applied Earth Observation and Geoinformation
- 作者：Yujie Li, Xuanguang Liu, Chenguang Dai, Zhenchao Zhang, Lei Ding, Han Zhu, Ruihua Lu, Hanyun Wang
- 中文摘要：面向光学-SAR变化检测，联合频率、时空依赖与结构上下文，并用双路径自适应融合校准语义和结构贡献。
- 有无代码：未证实（论文页面称将提供：https://github.com/liuxuanguang/FSTS-Net）
- 原文链接：https://doi.org/10.1016/j.jag.2026.105534
- PDF：未获取
- 相关性说明：与 BRIGHT/RGB-SAR任务接近；可用于理解频率与结构证据的分工，但不能直接假定代码可用。

---

需要下载开放论文时，请输入编号，例如：`P001,P003` 或 `P002-P006`。
