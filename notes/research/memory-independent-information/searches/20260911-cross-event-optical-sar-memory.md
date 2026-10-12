# 跨事件 Optical–SAR Memory 与表征学习论文检索

1. **P001 — Bridging Remote Sensors with Multisensor Geospatial Foundation Models**
   - 年份/收录：2024，CVPR 2024
   - 作者：Boran Han, Shuai Zhang, Xingjian Shi, Markus Reichstein
   - 中文摘要：利用约两百万幅多传感器影像进行跨传感器掩码预训练，同时支持配对与非配对数据，学习可联合使用的多传感器表征。
   - 原文：https://openaccess.thecvf.com/content/CVPR2024/html/Han_Bridging_Remote_Sensors_with_Multisensor_Geospatial_Foundation_Models_CVPR_2024_paper.html
   - 代码：https://github.com/boranhan/Geospatial_Foundation_Models
   - 相关性：适合替换当前手工均值卡，先提供真正独立于 P3 的跨传感器特征。

2. **P002 — SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery**
   - 年份/收录：2024，CVPR 2024
   - 作者：Xin Guo 等
   - 中文摘要：通过因子化多模态时空编码、多粒度对比学习和地理上下文原型，在大规模 Optical–SAR 数据上学习通用遥感表征。
   - 原文：https://openaccess.thecvf.com/content/CVPR2024/html/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.html
   - 代码：https://github.com/Jack-bo1220/SkySense
   - 相关性：说明事件/区域上下文应作为检索质量变量，而非与损毁语义直接混合。

3. **P003 — SkySense V2: A Unified Foundation Model for Multi-modal Remote Sensing**
   - 年份/收录：2025，ICCV 2025
   - 作者：Yingying Zhang, Lixiang Ru, Kang Wu, Lei Yu, Lei Liang, Yansheng Li, Jingdong Chen
   - 中文摘要：以统一 Transformer、模态提示、分辨率自适应 patch 合并和 MoE 处理多模态遥感数据。
   - 原文：https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_SkySense_V2_A_Unified_Foundation_Model_for_Multi-modal_Remote_Sensing_ICCV_2025_paper.html
   - 代码：https://github.com/Jack-bo1220/SkySense
   - 相关性：支持“共享语义主干 + 模态特有适配器”，避免把 RGB/SAR 强行压进同一手工向量。

4. **P004 — Universal Domain Adaptation for Semantic Segmentation (UniMAP)**
   - 年份/收录：2025，CVPR 2025
   - 作者：Seun-An Choe, Keon-Hee Park, Jinwoo Choi, Gyeong-Moon Park
   - 中文摘要：用域特定多原型区分类内多样性，并按目标内容匹配训练图像，以减轻域偏移和负迁移。
   - 原文：https://openaccess.thecvf.com/content/CVPR2025/html/Choe_Universal_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2025_paper.html
   - 代码状态：仓库存在，但当前页面标注“code will be released soon”：https://github.com/KU-VGI/UniMAP
   - 相关性：直接支持“每类多个机制/质量原型”，不应只保存单一损毁类别中心。

5. **P005 — Cross-modal Ship Re-Identification via Optical and SAR Imagery: A Novel Dataset and Method**
   - 年份/收录：2025，ICCV 2025
   - 作者：Han Wang, Shengyang Li, Jian Yang, Yuxuan Liu, Yixuan Lv, Zhuang Zhou
   - 中文摘要：提出 Optical–SAR 跨模态目标重识别数据与 TransOSS，通过模态信息嵌入、共享 Transformer 和大规模配对对比预训练学习模态不变特征。
   - 原文：https://openaccess.thecvf.com/content/ICCV2025/html/Wang_Cross-modal_Ship_Re-Identification_via_Optical_and_SAR_Imagery_A_Novel_ICCV_2025_paper.html
   - 代码：https://github.com/Alioth2000/Hoss-ReID
   - 相关性：与“建筑找到跨事件相似建筑”的检索目标最接近，但需把身份监督改为损毁机制/残差监督。

6. **P006 — Enhancing Remote Sensing Representations Through Mixed-Modality Masked Autoencoding**
   - 年份/收录：2025，WACV Workshops 2025
   - 作者：Ori Linial, George Leifman, Yochai Blau, Nadav Sherman, Yotam Gigi, Wojciech Sirko, Genady Beryozkin
   - 中文摘要：通过 Optical 重建、SAR 预测以及跨模态 patch 混合进行预训练，以缓解空间错配并学习互补表征。
   - 原文：https://openaccess.thecvf.com/content/WACV2025W/GeoCV/html/Linial_Enhancing_Remote_Sensing_Representations_Through_Mixed-Modality_Masked_Autoencoding_WACVW_2025_paper.html
   - 代码：未证实
   - 相关性：提示当前配准误差不应只作为卡片统计量，而应进入预训练扰动机制。

7. **P007 — ProMM-RS: Exploring Probabilistic Learning for Multi-Modal Remote Sensing Image Representations**
   - 年份/收录：2025，WACV Workshops 2025
   - 作者：Nicolas Houdré, Diego Marcos, Dino Ienco, Laurent Wendling, Camille Kurtz, Sylvain Lobry
   - 中文摘要：使用概率嵌入显式表示不同输入与模态之间的不确定性，以判断各模态在受遮挡或异常条件下的可靠程度。
   - 原文：https://openaccess.thecvf.com/content/WACV2025W/GeoCV/html/Houdre_ProMM-RS_Exploring_Probabilistic_learning_for_Multi-Modal_Remote_Sensing_Image_Representations_WACVW_2025_paper.html
   - 代码：未证实
   - 相关性：适合把相似度置信、SAR质量和邻居分歧变成拒绝条件。

8. **P008 — SARMAE: Masked Autoencoder for SAR Representation Learning**
   - 年份/收录：2026；正式会议收录未独立核验，按预印本记录
   - 作者：未获取
   - 中文摘要：构建大规模 SAR 预训练数据，显式注入散斑噪声，并借助冻结的光学语义编码器约束 SAR 表征。
   - 原文：https://arxiv.org/abs/2512.16635
   - 代码：https://github.com/MiliLab/SARMAE
   - 相关性：比手工 SAR 均值/MAD 更可能提供 P3 尚未学习的散射物理信息。

9. **P009 — SOMA-1M: A Large-Scale SAR-Optical Multi-resolution Alignment Dataset for Multi-Task Remote Sensing**
   - 年份/收录：2026，预印本 arXiv（未证实正式收录）
   - 作者：Peihao Wu, Yongxiang Yao, Yi Wan, Wenfei Zhang, Ruipeng Zhao, Jiayuan Li, Yongjun Zhang
   - 中文摘要：提供约 130 万对多分辨率 SAR–Optical 对，并通过粗到细配准和质量清洗支持匹配、融合与跨模态任务。
   - 原文：https://arxiv.org/abs/2602.05480
   - 项目：https://github.com/PeihaoWu/SOMA-1M
   - 相关性：外部数据应优先增加“高质量配对覆盖”，而不是继续增加同分布建筑卡数量。

10. **P010 — CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders**
    - 年份/收录：2023，NeurIPS 2023
    - 作者：Anthony Fuller, Koreen Millard, James R. Green
    - 中文摘要：对空间与时间对齐的 Radar–Optical 影像同时使用跨模态对比学习和掩码重建，输出单模态与联合 patch/GAP 表征。
    - 原文：https://arxiv.org/abs/2311.00566
    - 代码：https://github.com/antofuller/croma
    - 相关性：代码成熟，适合作为第一项外部编码器筛选；但原模型要求 2 通道 Sentinel-1 与 12 通道 Sentinel-2，不能直接伪造为 BRIGHT 的 1 通道 SAR/3 通道 RGB。

如需下载开放 PDF，可指定编号，例如 `P001,P003,P005`。

