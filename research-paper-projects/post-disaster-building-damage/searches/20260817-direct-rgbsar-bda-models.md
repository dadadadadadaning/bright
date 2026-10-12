# 可替换现有基础模型的 RGB–SAR 建筑损毁相关模型

- 检索日期：2026-08-17
- 检索关键词：pre-event optical post-event SAR building damage；BRIGHT Challenge winning solution；optical-SAR multimodal segmentation；heterogeneous change detection
- 数据源：CVPR Open Access、NeurIPS Proceedings、IEEE IGARSS 官方日程、arXiv、论文官方 GitHub

## P001 MM-OVSeg: Multimodal Optical–SAR Fusion for Open-Vocabulary Segmentation in Remote Sensing

- 年份：2026
- 收录于：CVPR 2026
- 作者：Yimin Wei, Aoran Xiao, Hongruixuan Chen, Junshi Xia, Naoto Yokoya
- 中文摘要：使用独立 RGB/SAR DINO 编码器、跨模态统一预训练和双编码器融合，实现恶劣天气下的光学–SAR开放词汇分割。
- 有无代码：有：https://github.com/Jimmyxichen/MM-OVSeg
- 原文链接：https://openaccess.thecvf.com/content/CVPR2026/papers/Wei_MM-OVSeg_Multimodal_Optical-SAR_Fusion_for_Open-Vocabulary_Segmentation_in_Remote_Sensing_CVPR_2026_paper.pdf
- PDF：开放
- 相关性说明：正式顶会、代码和SAR预训练权重完整，是完全替换DamageFormer时最可执行的双编码基础模型，但原任务不是时相损毁，需要改造成变化条件融合。

## P002 Advancing All-Weather Building Damage Mapping to the Instance Level: Outcomes and Insights from the 2026 Bright Challenge

- 年份：2026
- 收录于：预印本：arXiv（未证实正式收录）
- 作者：Hongruixuan Chen 等
- 中文摘要：总结BRIGHT实例级挑战，两个领先方案均采用模态专属编码、分阶段或后期融合，并以光学主导建筑定位。
- 有无代码：部分：官方基线完整；第一名链接失效，第二名公开仓库当前为空：https://github.com/ChenHongruixuan/BRIGHT
- 原文链接：https://arxiv.org/abs/2607.22746
- PDF：开放
- 相关性说明：与用户任务的数据、输入模态和损毁等级最一致，可作为架构设计依据，但获奖实现当前不能直接复用。

## P003 Building-Guided Pseudo-Label Learning for Cross-Modal Building Damage Mapping

- 年份：2025
- 收录于：IGARSS 2025
- 作者：Jiepan Li, He Huang, Yu Sheng, Yujun Guo, Wei He
- 中文摘要：先利用灾前光学提取建筑，再以建筑先验和低不确定伪标签训练跨模态损毁变化模型。
- 有无代码：未证实
- 原文链接：https://arxiv.org/abs/2505.04941
- PDF：开放
- 相关性说明：完全相同的灾前光学/灾后SAR像素级损毁任务，但会议层级和代码可复现性弱于P001。

## P004 CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders

- 年份：2023
- 收录于：NeurIPS 2023
- 作者：Anthony Fuller, Koreen Millard, James Green
- 中文摘要：分别编码雷达和光学数据，通过跨模态对比学习与掩膜重建获得单模态和联合表征。
- 有无代码：有：https://github.com/antofuller/CROMA
- 原文链接：https://proceedings.neurips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html
- PDF：开放
- 相关性说明：适合作为SAR/光学表征初始化，不是现成损毁预测模型，且其时空对齐预训练假设不能直接作用于真实变化区域。

## P005 SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection

- 年份：2024
- 收录于：NeurIPS 2024
- 作者：Yuxuan Li, Xiang Li, Weijie Li, Qibin Hou, Li Liu, Ming-Ming Cheng, Jian Yang
- 中文摘要：提出大规模SAR检测基准与多阶段滤波增强迁移，分别处理输入域、结构和迁移过程差距。
- 有无代码：有：https://github.com/zcablii/SARDet_100K
- 原文链接：https://proceedings.neurips.cc/paper_files/paper/2024/hash/e7eb8128eb26eafbe901348df1dbacdc-Abstract-Conference.html
- PDF：开放
- 相关性说明：最适合强化SAR编码器和迁移流程，不是完整RGB–SAR损毁模型。

## P006 Prior-guided Fusion of Multimodal Features for Change Detection from Optical-SAR Images

- 年份：2026
- 收录于：预印本：arXiv（未证实正式收录）
- 作者：Xuanguang Liu, Lei Ding, Yujie Li, Chenguang Dai, Zhenchao Zhang, Mengmeng Li, Ziyi Yang, Yifan Sun, Yongqi Sun, Hanyun Wang
- 中文摘要：STSF-Net使用模态特有与时空共有特征建模，并以语义先验自适应融合光学和SAR变化特征，在BRIGHT上报告实验。
- 有无代码：代码仓库已建立但当前仅有README：https://github.com/liuxuanguang/STSF-Net
- 原文链接：https://arxiv.org/abs/2604.05527
- PDF：开放
- 相关性说明：像素级任务与BRIGHT高度匹配，但尚非正式顶会且实现未公开，不能作为当前可执行基础代码。

---

需要下载论文时，可输入编号，例如：`P001,P004,P005`。

