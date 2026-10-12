# RGB–SAR轻微错位与跨模态匹配论文检索结果

- 检索日期：2026-09-17
- 检索关键词：RGB-SAR misregistration、SAR-optical matching、building change detection alignment、cross-modal feature matching
- 数据源：CVF Open Access、IEEE Xplore、AAAI Proceedings、ISPRS/Elsevier、GitHub官方仓库

## P001 Exploring Context Alignment and Structure Perception for Building Change Detection

- 年份：2025
- 收录于：IEEE TGRS
- 作者：Qi Wang, Mingwei Zhang, Jiawei Ren, Qiang Li
- 中文摘要：针对双时相建筑影像轻微错位导致的伪变化，提出双向参考引导的上下文聚合、差异诱导对齐和结构感知融合。
- 有无代码：有：https://github.com/ptdoge/CASP
- 原文链接：https://doi.org/10.1109/TGRS.2025.3540013
- PDF：开放作者版本
- 相关性说明：任务最接近建筑变化与轻微错位，但原任务是同模态双时相，需要改造成RGB锚定的SAR局部对齐。

## P002 SOMA: Feature Gradient Enhanced Affine-Flow Matching for SAR-Optical Registration

- 年份：2026
- 收录于：AAAI 2026
- 作者：Haoyu Wang, Tianyang Zhuo, Xiang Zhang, Hongcheng Yin, Hui Wu, Yanning Zhang
- 中文摘要：针对SAR–光学配准，将多尺度多方向梯度先验注入深层特征，并采用全局仿射与局部Flow的粗到细匹配。
- 有无代码：未证实
- 原文链接：https://doi.org/10.1609/aaai.v40i12.37939
- PDF：出版社开放
- 相关性说明：模态最匹配、对齐能力最完整，但对于几像素局部残差和分割任务可能过重。

## P003 XoFTR: Cross-modal Feature Matching Transformer

- 年份：2024
- 收录于：CVPR Workshop 2024
- 作者：Önder Tuzcuoğlu, Aybora Köksal, Buğra Sofu, Sinan Kalkan, A. Aydın Alatan
- 中文摘要：面向可见光–热红外匹配，通过跨模态预训练、粗匹配、精细匹配和亚像素修正提高局部对应可靠性。
- 有无代码：有：https://github.com/OnderT/XoFTR
- 原文链接：https://openaccess.thecvf.com/content/CVPR2024W/IMW/html/Tuzcuoglu_XoFTR_Cross-modal_Feature_Matching_Transformer_CVPRW_2024_paper.html
- PDF：开放
- 相关性说明：粗到细思想可迁移，但模态不是SAR且完整匹配器计算较重。

## P004 MINIMA: Modality Invariant Image Matching

- 年份：2025
- 收录于：CVPR 2025
- 作者：Jiangwei Ren, Xingyu Jiang, Zizhuo Li, Dingkang Liang, Xin Zhou, Xiang Bai
- 中文摘要：通过合成多模态数据引擎为现有匹配器提供大规模准确对应监督，并验证多种跨模态匹配情形，包括Optical–SAR。
- 有无代码：有：https://github.com/LSXI7/MINIMA
- 原文链接：https://openaccess.thecvf.com/content/CVPR2025/html/Ren_MINIMA_Modality_Invariant_Image_Matching_CVPR_2025_paper.html
- PDF：开放
- 相关性说明：最适合借鉴“人为位移产生准确监督”，不必完整移植其大规模匹配器。

## P005 HOMO-Feature: Cross-Arbitrary-Modal Image Matching with Homomorphism of Organized Major Orientation

- 年份：2025
- 收录于：ICCV 2025
- 作者：Chenzhong Gao, Wei Li, Desheng Weng
- 中文摘要：使用主方向图、多尺度和广义极坐标描述子构造跨模态稳定结构特征，减少强度差异的影响。
- 有无代码：有：https://github.com/MrPingQi/HOMO_Feature_ImgMatching
- 原文链接：https://openaccess.thecvf.com/content/ICCV2025/html/Gao_HOMO-Feature_Cross-Arbitrary-Modal_Image_Matching_with_Homomorphism_of_Organized_Major_Orientation_ICCV_2025_paper.html
- PDF：开放
- 相关性说明：梯度方向对RGB–SAR比原始强度更稳定，适合作为轻量局部匹配输入。

## P006 Robust change detection for remote sensing images based on temporospatial interactive attention module

- 年份：2024
- 收录于：International Journal of Applied Earth Observation and Geoinformation
- 作者：Jinjiang Wei, Kaimin Sun, Wenzhuo Li, Wangbin Li, Song Gao, Shunxia Miao, Qinhui Zhou, Junyi Liu
- 中文摘要：使用时空交互注意力查询并重建双时相语义相关性，以缓解视角旋转和时相风格干扰。
- 有无代码：有：https://github.com/wjj282439449/CDNeXt
- 原文链接：https://doi.org/10.1016/j.jag.2024.103767
- PDF：开放
- 相关性说明：说明交互式局部关系优于简单自注意力，但未专门针对RGB–SAR。

---

需要下载论文时，请输入编号，例如：`P001,P003` 或 `P002-P006`。
