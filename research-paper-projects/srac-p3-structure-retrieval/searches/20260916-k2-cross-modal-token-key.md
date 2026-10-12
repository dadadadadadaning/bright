# K2独立跨模态Token与检索Key论文检索结果

- 检索日期：2026-09-16
- 检索关键词：cross-modal feature matching、optical-SAR、superpixel token、set aggregation、local-global retrieval descriptor
- 数据源：CVF Open Access、PMLR、NeurIPS Proceedings、论文官方GitHub/项目页

## P001 XoFTR: Cross-modal Feature Matching Transformer

- 年份：2024
- 收录于：CVPR Workshop 2024
- 作者：Önder Tuzcuoğlu、Aybora Köksal、Buğra Sofu、Sinan Kalkan、A. Aydin Alatan
- 中文摘要：面向可见光—热红外跨模态局部匹配，通过跨模态预训练、粗到细匹配和亚像素细化提升尺度变化与模态差异下的对应可靠性。
- 有无代码：有：https://github.com/OnderT/XoFTR
- 原文链接：https://openaccess.thecvf.com/content/CVPR2024W/IMW/html/Tuzcuoglu_XoFTR_Cross-modal_Feature_Matching_Transformer_CVPRW_2024_paper.html
- PDF：开放
- 相关性说明：支持将RGB–SAR关系Token拆成模态专用描述、局部候选对应、匹配置信度，而不是直接做同位置特征相减。

## P002 MINIMA: Modality Invariant Image Matching

- 年份：2025
- 收录于：CVPR 2025
- 作者：Jiangwei Ren、Xingyu Jiang、Zizhuo Li、Dingkang Liang、Xin Zhou、Xiang Bai
- 中文摘要：通过可扩展的多模态合成数据引擎生成准确对应标签，使通用匹配管线学习跨模态不变特征，并在包括光学—SAR在内的多种跨模态场景中验证泛化。
- 有无代码：有：https://github.com/LSXI7/MINIMA
- 原文链接：https://openaccess.thecvf.com/content/CVPR2025/html/Ren_MINIMA_Modality_Invariant_Image_Matching_CVPR_2025_paper.html
- PDF：开放
- 相关性说明：支持用人工SAR位移和已知对应关系训练Key的一致性，而不是仅依赖损毁类别监督。

## P003 HOMO-Feature: Cross-Arbitrary-Modal Image Matching with Homomorphism of Organized Major Orientation

- 年份：2025
- 收录于：ICCV 2025
- 作者：Chenzhong Gao、Wei Li、Desheng Weng
- 中文摘要：使用主方向图、广义极坐标描述符和多尺度策略提取跨任意模态更稳定的结构特征，降低纹理和强度差异对匹配的影响。
- 有无代码：有：https://github.com/MrPingQi/HOMO_Feature_ImgMatching
- 原文链接：https://openaccess.thecvf.com/content/ICCV2025/html/Gao_HOMO-Feature_Cross-Arbitrary-Modal_Image_Matching_with_Homomorphism_of_Organized_Major_Orientation_ICCV_2025_paper.html
- PDF：开放
- 相关性说明：支持在K2中加入RGB边缘方向、SAR局部梯度方向等跨模态稳定结构，而不是过度依赖原始幅值。

## P004 Superpixel Sampling Networks

- 年份：2018
- 收录于：ECCV 2018
- 作者：Varun Jampani、Deqing Sun、Ming-Yu Liu、Ming-Hsuan Yang、Jan Kautz
- 中文摘要：将SLIC式超像素采样改造成可微模块，使超像素能够针对下游任务学习，而不是固定依赖手工颜色距离。
- 有无代码：有：https://github.com/NVlabs/ssn_superpixels
- 原文链接：https://openaccess.thecvf.com/content_ECCV_2018/html/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.html
- PDF：开放
- 相关性说明：支持把固定SLICO保留为首轮低成本基线，并把学习式任务相关区域放到第二阶段，而不是一开始就替换全部区域生成流程。

## P005 Deep Sets

- 年份：2017
- 收录于：NeurIPS 2017
- 作者：Manzil Zaheer、Satwik Kottur、Siamak Ravanbakhsh、Barnabas Poczos、Ruslan Salakhutdinov、Alexander J. Smola
- 中文摘要：给出集合函数的置换不变结构，用共享元素编码与聚合构造不依赖元素排列顺序的集合表示。
- 有无代码：未证实
- 原文链接：https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html
- PDF：开放
- 相关性说明：适合把一栋建筑内数量不固定的局部证据Token聚合为建筑级Key。

## P006 Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks

- 年份：2019
- 收录于：ICML 2019
- 作者：Juho Lee、Yoonho Lee、Jungtaek Kim、Adam Kosiorek、Seungjin Choi、Yee Whye Teh
- 中文摘要：通过集合自注意力建模集合元素间交互，同时保持对输入顺序不敏感，并用诱导点降低注意力计算成本。
- 有无代码：有：https://github.com/juho-lee/set_transformer
- 原文链接：https://proceedings.mlr.press/v97/lee19d.html
- PDF：开放
- 相关性说明：适合第二阶段建模建筑内多个局部损毁证据之间的关系，但首版成本和过拟合风险高于Deep Sets。

## P007 NetVLAD: CNN Architecture for Weakly Supervised Place Recognition

- 年份：2016
- 收录于：CVPR 2016
- 作者：Relja Arandjelovic、Petr Gronat、Akihiko Torii、Tomas Pajdla、Josef Sivic
- 中文摘要：把局部描述符相对学习中心的残差进行可学习聚合，形成适合大规模检索的紧凑全局描述符。
- 有无代码：有：https://www.di.ens.fr/willow/research/netvlad/
- 原文链接：https://openaccess.thecvf.com/content_cvpr_2016/html/Arandjelovic_NetVLAD_CNN_Architecture_CVPR_2016_paper.html
- PDF：开放
- 相关性说明：提示Token聚合不应只有mean/std/max，还可以编码局部证据相对多个损毁模式中心的残差。

## P008 Patch-NetVLAD: Multi-Scale Fusion of Locally-Global Descriptors for Place Recognition

- 年份：2021
- 收录于：CVPR 2021
- 作者：Stephen Hausler、Sourav Garg、Ming Xu、Michael Milford、Tobias Fischer
- 中文摘要：从NetVLAD残差中构造多尺度patch级局部—全局描述符，在条件和视角变化下兼顾局部对应与全局检索。
- 有无代码：有：https://github.com/QVPR/Patch-NetVLAD
- 原文链接：https://openaccess.thecvf.com/content/CVPR2021/html/Hausler_Patch-NetVLAD_Multi-Scale_Fusion_of_Locally-Global_Descriptors_for_Place_Recognition_CVPR_2021_paper.html
- PDF：开放
- 相关性说明：支持为建筑同时保留核心、Ring-8、Ring-16等多尺度局部Token，再形成建筑级Key，而不是过早压成单一平均向量。

---

需要下载论文时，请输入编号，例如：`P001,P003` 或 `P002-P006`。
