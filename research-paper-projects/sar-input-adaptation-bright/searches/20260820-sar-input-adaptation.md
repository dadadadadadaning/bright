# SAR 输入适配与 RGB–SAR 迁移检索（2026-08-20）

1. **P001｜SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection**
   - 年份：2024
   - 收录于：NeurIPS 2024
   - 作者：Yuxuan Li, Xiang Li, Weijie Li, Qibin Hou, Li Liu, Ming-Ming Cheng, Jian Yang
   - 原文：[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e7eb8128eb26eafbe901348df1dbacdc-Abstract-Conference.html)
   - 中文摘要：论文系统指出 RGB 预训练迁移至 SAR 时同时存在输入域、预训练域和模型迁移差距，并提出多阶段、滤波增强的 MSFA 迁移框架。
   - 有无代码：有：[官方代码](https://github.com/zcablii/SARDet_100K)
   - 一句相关性：最直接说明“只换第一层”不足，适合作为 BRIGHT SAR 分支分阶段迁移的依据。
   - PDF 状态：open，NeurIPS

2. **P002｜CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders**
   - 年份：2023
   - 收录于：NeurIPS 2023
   - 作者：Anthony Fuller, Koreen Millard, James Green
   - 原文：[NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html)
   - 中文摘要：使用独立雷达和光学编码器、跨模态对比学习、掩码重建及联合跨注意力编码器学习单模态和多模态表征。
   - 有无代码：有：[官方代码](https://github.com/antofuller/CROMA)
   - 一句相关性：支持“模态独立输入与编码、只在高层共同语义上交互”，但其 Sentinel-1 双通道契约不能直接套到 BRIGHT 单极化 VHR SAR。
   - PDF 状态：open，NeurIPS

3. **P003｜SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery**
   - 年份：2024
   - 收录于：CVPR 2024
   - 作者：Xin Guo 等
   - 原文：[CVPR](https://openaccess.thecvf.com/content/CVPR2024/html/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.html)
   - 中文摘要：在大规模光学和 SAR 时序数据上预训练因子化多模态时空编码器，并使用多粒度对比学习与地理上下文原型学习。
   - 有无代码：有：[官方代码](https://github.com/Jack-bo1220/SkySense)
   - 一句相关性：证明 SAR 不必伪装成 RGB，而可拥有独立编码路径并在多粒度语义空间联合学习。
   - PDF 状态：open，CVF

4. **P004｜AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities**
   - 年份：2025
   - 收录于：CVPR 2025
   - 作者：Guillaume Astruc, Nicolas Gonthier, Clément Mallet, Loic Landrieu
   - 原文：[CVPR](https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html)
   - 中文摘要：为不同传感器设计模态专属投影器和尺度自适应 patch encoding，再以共享空间编码器及跨注意力组合器获得统一表征。
   - 有无代码：有：[官方代码](https://github.com/gastruc/AnySat)
   - 一句相关性：其“模态专属投影器 + 共享高层编码”比随机更换单通道 PatchEmbed 更符合你的输入适配问题。
   - PDF 状态：open，CVF

5. **P005｜SkySense V2: A Unified Foundation Model for Multi-modal Remote Sensing**
   - 年份：2025
   - 收录于：ICCV 2025
   - 作者：Yingying Zhang, Lixiang Ru, Kang Wu, Lei Yu, Lei Liang, Yansheng Li, Jingdong Chen
   - 原文：[ICCV](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_SkySense_V2_A_Unified_Foundation_Model_for_Multi-modal_Remote_Sensing_ICCV_2025_paper.html)
   - 中文摘要：用统一 Transformer 处理多模态，借助自适应 patch merging、可学习模态提示和 MoE 适配不同分辨率与模态特征。
   - 有无代码：有：[官方项目](https://github.com/Jack-bo1220/SkySense)
   - 一句相关性：说明共享主干可行，但前提是大规模多模态预训练与显式模态适配，而不是在小型 BRIGHT 数据上临时共享。
   - PDF 状态：open，CVF

6. **P006｜SARMAE: Masked Autoencoder for SAR Representation Learning**
   - 年份：2026
   - 收录于：CVPR 2026
   - 作者：Danxu Liu, Di Wang, Hebaixu Wang, Haoyang Chen, Wentao Jiang, Yilin Cheng, Haonan Guo, Wei Cui, Jing Zhang
   - 原文：[CVPR](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_SARMAE_Masked_Autoencoder_for_SAR_Representation_Learning_CVPR_2026_paper.html)
   - 中文摘要：基于 SAR-1M 进行掩码自编码预训练，引入斑点感知增强，并以配对光学编码提供语义锚点约束。
   - 有无代码：有：[官方代码](https://github.com/MiliLab/SARMAE)
   - 一句相关性：是最直接的 SAR 专属预训练来源；你的 T1/T1E 只否定了深层残差接入方式，尚未否定建筑 ROI 级迁移。
   - PDF 状态：open，CVF

7. **P007｜HeteCD: Feature Consistency Alignment and Difference Mining for Heterogeneous Remote Sensing Image Change Detection**
   - 年份：2025
   - 收录于：ISPRS Journal of Photogrammetry and Remote Sensing
   - 作者：Wei Jing, Haichen Bai, Binbin Song, Weiping Ni, Junzheng Wu, Qi Wang
   - 原文：[DOI](https://doi.org/10.1016/j.isprsjprs.2025.03.008)
   - 中文摘要：采用非共享 Transformer 编码器、特征一致性对齐和 3D 时空注意力差异模块处理异构遥感变化检测。
   - 有无代码：有：[官方代码](https://github.com/weiAI1996/HeteCD)
   - 一句相关性：最贴近 RGB–SAR 异构变化，但二值建筑变化与 BRIGHT 三级损毁、跨事件泛化并不等价。
   - PDF 状态：未获取

8. **P008｜Building-Guided Pseudo-Label Learning for Cross-Modal Building Damage Mapping**
   - 年份：2025
   - 收录于：IGARSS 2025
   - 作者：Jiepan Li, He Huang, Yu Sheng, Yujun Guo, Wei He
   - 原文：[arXiv](https://arxiv.org/abs/2505.04941)
   - 中文摘要：先从灾前光学提取建筑，再训练跨模态损毁变化网络，并利用建筑先验和低不确定性伪标签优化损毁预测。
   - 有无代码：有：[官方代码](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping)
   - 一句相关性：任务最一致，支持光学主导定位、独立损毁阶段和建筑约束；其测试伪标签属于传导式协议，必须与纯监督结果分开。
   - PDF 状态：open，arXiv

9. **P009｜Neural Plasticity-Inspired Foundation Model for Observing the Earth Crossing Modalities (DOFA)**
   - 年份：2024
   - 收录于：预印本：arXiv（未证实正式收录）
   - 作者：Zhitong Xiong, Yi Wang, Fahong Zhang, Adam J. Stewart, Joëlle Hanna, Damian Borth, Ioannis Papoutsis, Bertrand Le Saux, Gustau Camps-Valls, Xiao Xiang Zhu
   - 原文：[arXiv](https://arxiv.org/abs/2403.15356)
   - 中文摘要：通过动态超网络根据输入波段信息生成或调整 patch embedding，使一个 Transformer 处理不同传感器与通道配置。
   - 有无代码：有：[官方代码](https://github.com/zhu-xlab/DOFA)
   - 一句相关性：提供“动态输入适配”思路，但 BRIGHT 单极化幅度图缺少可靠波长描述，不能无修改直接套用。
   - PDF 状态：open，arXiv

数据来源：NeurIPS Proceedings、CVF Open Access、arXiv、DOI 出版记录及论文官方 GitHub。

如需下载开放 PDF，可指定编号，例如 `P001,P002,P004-P006`。
