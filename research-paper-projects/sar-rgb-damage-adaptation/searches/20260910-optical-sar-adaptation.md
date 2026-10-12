# 光学-SAR适配与变化检测论文检索结果

- 检索日期：2026-09-10
- 检索关键词：optical-SAR change detection, native SAR encoder, filter augmentation, cross-sensor alignment, multimodal representation
- 数据源：CVF Open Access、NeurIPS Proceedings、ECVA、arXiv、期刊页面和论文官方GitHub

## P001 M2CD: A Unified MultiModal Framework for Optical-SAR Change Detection with Mixture of Experts and Self-Distillation

- 年份：2025
- 收录于：预印本：arXiv（本次未从出版社页面核实正式收录）
- 作者：Ziyuan Liu, Jiawei Zhang, Wenyu Wang, Yuantao Gu
- 中文摘要：针对灾前光学、灾后SAR变化检测，使用模态专家和仅训练期使用的光学到SAR桥接路径进行自蒸馏。
- 有无代码：有：https://github.com/circleLZY/M2CD
- 原文链接：https://arxiv.org/abs/2503.19406
- PDF：开放：https://arxiv.org/pdf/2503.19406
- 相关性说明：与BRIGHT的灾前RGB、灾后SAR输入形式最接近。

## P002 SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection

- 年份：2024
- 收录于：NeurIPS 2024
- 作者：Yuxuan Li, Xiang Li, Weijie Li, Qibin Hou, Li Liu, Ming-Ming Cheng, Jian Yang
- 中文摘要：指出RGB预训练到SAR存在数据域和结构域差异，并提出多阶段预训练与滤波增强输入。
- 有无代码：有：https://github.com/zcablii/SARDet_100K
- 原文链接：https://proceedings.neurips.cc/paper_files/paper/2024/hash/e7eb8128eb26eafbe901348df1dbacdc-Abstract-Conference.html
- PDF：开放：https://arxiv.org/pdf/2403.06534
- 相关性说明：是当前固定SAR滤波思路的直接参考，但其有效方法不仅包含滤波。

## P003 CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders

- 年份：2023
- 收录于：NeurIPS 2023
- 作者：Anthony Fuller, Koreen Millard, James R. Green
- 中文摘要：分别编码空间时间对齐的雷达与光学数据，再通过对比、掩码重建和联合编码器学习单模态及融合表示。
- 有无代码：有：https://github.com/antofuller/CROMA
- 原文链接：https://openreview.net/forum?id=ezqI5WgGvY
- PDF：开放：https://openreview.net/pdf?id=ezqI5WgGvY
- 相关性说明：可借鉴独立编码和联合表示，但其Sentinel输入与BRIGHT单通道显示SAR不同。

## P004 Decoupling Common and Unique Representations for Multimodal Self-supervised Learning

- 年份：2024
- 收录于：ECCV 2024
- 作者：Yi Wang, Conrad M. Albrecht, Nassim Ait Ali Braham, Chenying Liu, Zhitong Xiong, Xiao Xiang Zhu
- 中文摘要：把跨模态表示拆成公共与模态独有部分，公共维度对齐、独有维度解耦，并用模态内学习避免独有特征坍塌。
- 有无代码：有：https://github.com/zhu-xlab/DeCUR
- 原文链接：https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4236_ECCV_2024_paper.php
- PDF：开放：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04236.pdf
- 相关性说明：解释了为什么不能强迫全部SAR特征直接模仿RGB特征。

## P005 AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities

- 年份：2025
- 收录于：CVPR 2025
- 作者：Guillaume Astruc, Nicolas Gonthier, Clement Mallet, Loic Landrieu
- 中文摘要：使用尺度自适应空间编码器与JEPA，在11种传感器和多种分辨率上学习统一地球观测表示。
- 有无代码：有：https://github.com/gastruc/AnySat
- 原文链接：https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html
- PDF：开放：https://openaccess.thecvf.com/content/CVPR2025/papers/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.pdf
- 相关性说明：适合作为尺度/模态适配设计参考，不宜未经探针验证直接替换当前编码器。

## P006 Cross-sensor Self-supervised Training and Alignment for Remote Sensing

- 年份：2025
- 收录于：IEEE JSTARS
- 作者：Valerio Marsocci, Nicolas Audebert
- 中文摘要：通过多传感器稠密对齐损失把不同传感器和分辨率的特征对齐，并支持持续预训练适配新传感器。
- 有无代码：有：https://github.com/VMarsocci/X-STARS
- 原文链接：https://arxiv.org/abs/2405.09922
- PDF：开放：https://arxiv.org/pdf/2405.09922
- 相关性说明：可借鉴投影后对齐而不是直接让完整SAR特征等于旧P3特征。

## P007 GLCD-DA: Change Detection from Optical and SAR Imagery Using a Global-Local Network with Diversified Attention

- 年份：2025
- 收录于：ISPRS Journal of Photogrammetry and Remote Sensing
- 作者：Jie Li, Meiru Wu, Liupeng Lin, Qiangqiang Yuan, Huanfeng Shen
- 中文摘要：先进行模态同质化，再用边缘注入、全局局部交互与多层特征融合完成光学-SAR变化检测。
- 有无代码：有：https://github.com/WHU-SGG-RS-Pro-Group/ChangeDetection-GLCD-DA
- 原文链接：https://doi.org/10.1016/j.isprsjprs.2025.05.029
- PDF：未获取
- 相关性说明：边缘辅助与局部/全局分工可迁移，但图像翻译链路成本和伪变化风险较高。

---

需要下载开放论文时，可提供编号，例如：`P001,P002,P004`。

