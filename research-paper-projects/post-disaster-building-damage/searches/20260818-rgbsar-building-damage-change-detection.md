# 2023—2026 RGB–SAR 建筑损伤评估与异构变化检测论文检索

- 检索日期：2026-08-18
- 范围：2023—2026；最多 20 篇
- 检索主题：灾前 optical/RGB + 灾后 SAR 建筑损伤评估、optical–SAR 异构变化检测、模态独立编码、非共享双流、变化感知对齐、后期融合
- 说明：会议/期刊、代码和开放 PDF 分别核验；“预印本”不写成正式会议论文；未发现合法开放 PDF 的论文不下载。

## A. 直接建筑损伤评估

1. **BRIGHT: A Globally Distributed Multimodal Building Damage Assessment Dataset with Very-High-Resolution for All-Weather Disaster Response**
   - 年份/出处：2025，*Earth System Science Data* 17，6217–6243
   - 摘要：构建全球分布的灾前光学、灾后 SAR 建筑损毁数据集及像素/实例评估协议。
   - 代码：[ChenHongruixuan/BRIGHT](https://github.com/ChenHongruixuan/BRIGHT)
   - 开放 PDF：[出版社 PDF](https://essd.copernicus.org/articles/17/6217/2025/essd-17-6217-2025.pdf)；本地已有 `D:\software\文献\A01-BRIGHT-2025.pdf`

2. **Advancing All-Weather Building Damage Mapping to the Instance Level: Outcomes and Insights from the 2026 BRIGHT Challenge**
   - 年份/出处：2026，预印本：arXiv 2607.22746（未证实正式收录）
   - 摘要：总结 BRIGHT 实例级挑战、参赛方案和跨事件泛化难点。
   - 代码：[BRIGHT 官方仓库](https://github.com/ChenHongruixuan/BRIGHT)
   - 开放 PDF：[arXiv](https://arxiv.org/pdf/2607.22746)；本地已有 `D:\software\文献\A09-BRIGHT-Challenge-2026.pdf`

3. **M3ICNet: A Cross-Modal Resolution Preserving Building Damage Detection Method with Optical and SAR Remote Sensing Imagery and Two Heterogeneous Image Disaster Datasets**
   - 年份/出处：2025，*ISPRS Journal of Photogrammetry and Remote Sensing* 221，224–250；DOI: 10.1016/j.isprsjprs.2025.02.004
   - 摘要：以多模态、多分辨率和多层级交互保持 SAR 原始结构，并引入 WBD/EBD 两个灾害数据集。
   - 作者：Haiming Zhang，Guorui Ma，Di Wang，Yongxian Zhang
   - 代码：未证实；开放 PDF：未发现

4. **Building-Guided Pseudo-Label Learning for Cross-Modal Building Damage Mapping**
   - 年份/出处：2025，IGARSS 2025
   - 摘要：以建筑先验约束低不确定性伪标签，处理灾前光学与灾后 SAR 的跨模态损毁映射。
   - 代码：[Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping)
   - 开放 PDF：[arXiv](https://arxiv.org/pdf/2505.04941)

5. **QuickQuakeBuildings: Post-Earthquake SAR-Optical Dataset for Quick Damaged-Building Detection**
   - 年份/出处：2024，*IEEE Geoscience and Remote Sensing Letters* 21；DOI: 10.1109/LGRS.2024.3406966
   - 摘要：提供灾后 SAR–optical 建筑损毁数据与快速检测基线。
   - 代码：[ya0-sun/PostEQ-SARopt-BuildingDamage](https://github.com/ya0-sun/PostEQ-SARopt-BuildingDamage)
   - 开放 PDF：[arXiv](https://arxiv.org/pdf/2312.06587)

6. **Multiclass Post-Earthquake Building Assessment Integrating High-Resolution Optical and SAR Satellite Imagery, Ground Motion, and Soil Data with Transformers**
   - 年份/出处：2025，*Earthquake Spectra* 41(5)，3365–3392；DOI: 10.1177/87552930251377778
   - 摘要：以 Transformer 融合灾后高分辨率影像、SAR 损伤代理、地震动与土壤元数据，完成建筑级多分类。
   - 代码：未证实
   - 开放 PDF：[arXiv](https://arxiv.org/pdf/2412.04664)

7. **A Deep Learning Framework for Building Damage Assessment Using VHR SAR and Geospatial Data: Demonstration on the 2023 Turkiye Earthquake**
   - 年份/出处：2025，预印本：arXiv 2506.22338（未证实正式收录）
   - 摘要：融合单时相 VHR SAR、OSM 建筑轮廓、DSM 与 GEM 暴露属性进行建筑损毁判别。
   - 代码：论文仅声明录用后公开，当前未获取
   - 开放 PDF：[arXiv](https://arxiv.org/pdf/2506.22338)

## B. Optical–SAR / 异构变化检测

8. **Prior-Guided Fusion of Multimodal Features for Change Detection from Optical-SAR Images (STSF-Net)**
   - 年份/出处：2026，预印本：arXiv 2604.05527（未证实正式收录）
   - 摘要：区分模态特有与时空共有特征，再用语义先验引导 optical–SAR 变化融合。
   - 代码：[liuxuanguang/STSF-Net](https://github.com/liuxuanguang/STSF-Net)（检索时仓库未见完整训练实现）
   - 开放 PDF：[arXiv](https://arxiv.org/pdf/2604.05527)

9. **M²CD: A Unified MultiModal Framework for Optical-SAR Change Detection with Mixture of Experts and Self-Distillation**
   - 年份/出处：2025，*IEEE Geoscience and Remote Sensing Letters* 22；DOI: 10.1109/LGRS.2025.3590959
   - 摘要：在 CNN/Transformer 主干中加入模态专家，使用 optical-to-SAR 辅助路径和自蒸馏缩小特征差距。
   - 代码：[circleLZY/M2CD](https://github.com/circleLZY/M2CD)
   - 开放 PDF：[arXiv](https://arxiv.org/pdf/2503.19406)

10. **HeteCD: Feature Consistency Alignment and Difference Mining for Heterogeneous Remote Sensing Image Change Detection**
    - 年份/出处：2025，*ISPRS Journal of Photogrammetry and Remote Sensing* 223，317–327；DOI: 10.1016/j.isprsjprs.2025.03.008
    - 摘要：使用非共享双流 Transformer、特征一致性对齐和时空差异挖掘处理异构变化检测。
    - 代码：[weiAI1996/HeteCD](https://github.com/weiAI1996/HeteCD)
    - 开放 PDF：未发现

11. **GLCD-DA: Change Detection from Optical and SAR Imagery Using a Global-Local Network with Diversified Attention**
    - 年份/出处：2025，*ISPRS Journal of Photogrammetry and Remote Sensing* 226，396–414；DOI: 10.1016/j.isprsjprs.2025.05.029
    - 摘要：组合图像转换、边缘注入、CNN–Transformer 全局局部交互与多层差异融合。
    - 代码：[WHU-SGG-RS-Pro-Group/ChangeDetection-GLCD-DA](https://github.com/WHU-SGG-RS-Pro-Group/ChangeDetection-GLCD-DA)
    - 开放 PDF：未发现

12. **Refined Change Detection in Heterogeneous Low-Resolution Remote Sensing Images for Disaster Emergency Response (HRSICD)**
    - 年份/出处：2025，*ISPRS Journal of Photogrammetry and Remote Sensing* 220，139–155；DOI: 10.1016/j.isprsjprs.2024.12.010
    - 摘要：利用结构关系和语义信息提升低分辨率、多源灾害变化检测及跨场景泛化。
    - 代码：[Lucky-DW/HRSICD](https://github.com/Lucky-DW/HRSICD)
    - 开放 PDF：未发现

13. **Adversarial Feature Equilibrium Network for Multimodal Change Detection in Heterogeneous Remote Sensing Images (AFENet)**
    - 年份/出处：2024，*IEEE Transactions on Geoscience and Remote Sensing* 62，4512617
    - 摘要：以多尺度对抗域适配缩小模态差距，并用差异感知对比约束维持变化可分性。
    - 代码：[PuYan79/AFENet_MCDpytorch](https://github.com/PuYan79/AFENet_MCDpytorch)
    - 开放 PDF：未证实

14. **Fourier Domain Structural Relationship Analysis for Unsupervised Multimodal Change Detection (FD-MCD)**
    - 年份/出处：2023，*ISPRS Journal of Photogrammetry and Remote Sensing* 198，99–114；DOI: 10.1016/j.isprsjprs.2023.03.004
    - 摘要：在 Fourier/图 Fourier 域比较模态无关的局部和非局部结构关系，并进行频率解耦融合。
    - 代码：[ChenHongruixuan/FDMCD](https://github.com/ChenHongruixuan/FDMCD)
    - 开放 PDF：未找到可确认的作者或出版社开放直链

15. **Similarity and Dissimilarity Relationships Based Graphs for Multimodal Change Detection (SDIR)**
    - 年份/出处：2024，*ISPRS Journal of Photogrammetry and Remote Sensing* 208，70–88；DOI: 10.1016/j.isprsjprs.2024.01.002
    - 摘要：分别以 KNN/KFN 图建模低频相似关系和高频不相似关系，再执行异构图像回归与 MRF 融合。
    - 代码：[yulisun/SDIR](https://github.com/yulisun/SDIR)
    - 开放 PDF：未发现

16. **A Graph Contrastive Learning Network for Change Detection with Heterogeneous Remote Sensing Images (GCLN)**
    - 年份/出处：2026，*Pattern Recognition* 172，112394；DOI: 10.1016/j.patcog.2025.112394
    - 摘要：在超像素图中联合学习近邻/远邻结构，并以图对比目标分离相似与不相似特征。
    - 代码：[ImgSciGroup/2024-GCLN](https://github.com/ImgSciGroup/2024-GCLN)
    - 开放 PDF：未发现

17. **COMIC: An Unsupervised Change Detection Method for Heterogeneous Remote Sensing Images Based on Copula Mixtures and Cycle-Consistent Adversarial Networks**
    - 年份/出处：2024，*Information Fusion* 106，102240；DOI: 10.1016/j.inffus.2024.102240
    - 摘要：以 CycleGAN 生成同域表示，再用 Copula 混合建模跨模态特征依赖并进行变化假设检验。
    - 代码：未证实
    - 开放 PDF：[arXiv](https://arxiv.org/pdf/2304.00721)

## C. 可迁移的独立编码、融合与预训练

18. **CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders**
    - 年份/出处：2023，NeurIPS 2023
    - 摘要：雷达与光学独立编码，以对比学习和掩膜重建联合预训练单模态/跨模态表征。
    - 代码：[antofuller/CROMA](https://github.com/antofuller/CROMA)
    - 开放 PDF：[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/file/11822e84689e631615199db3b75cd0e4-Paper-Conference.pdf)；本地已有 `D:\software\文献\C02-CROMA-2023.pdf`

19. **Keep the Balance: A Parameter-Efficient Symmetrical Framework for RGB+X Semantic Segmentation**
    - 年份/出处：2025，CVPR 2025，10587–10598
    - 摘要：以对称模态适配和动态稀疏跨模态融合避免 RGB 主导及全局注意力噪声。
    - 代码：[imcjx/KTB](https://github.com/imcjx/KTB)
    - 开放 PDF：[CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Cai_Keep_the_Balance_A_Parameter-Efficient_Symmetrical_Framework_for_RGBX_Semantic_CVPR_2025_paper.pdf)；本地已有 `D:\software\文献\D01-Keep-the-Balance-2025.pdf`

20. **MM-OVSeg: Multimodal Optical-SAR Fusion for Open-Vocabulary Segmentation in Remote Sensing**
    - 年份/出处：2026，CVPR 2026，42202–42212
    - 摘要：使用 optical/SAR 独立编码、跨模态统一表示和双编码器融合完成开放词汇遥感分割。
    - 代码：[Jimmyxichen/MM-OVSeg](https://github.com/Jimmyxichen/MM-OVSeg)
    - 开放 PDF：[CVF](https://openaccess.thecvf.com/content/CVPR2026/papers/Wei_MM-OVSeg_Multimodal_Optical-SAR_Fusion_for_Open-Vocabulary_Segmentation_in_Remote_Sensing_CVPR_2026_paper.pdf)

## 下载状态

- 已存在、未重复下载：P001、P002、P018、P019。
- 已确认开放并完成下载：P004、P005、P006、P007、P008、P009、P017、P020；均已校验文件头为 `%PDF`。
- 未下载：P003、P010–P016，原因是未发现可确认的合法开放 PDF；检索记录保留出版社和代码入口。
