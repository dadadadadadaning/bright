# P3–SRAC结构区域与检索增强论文检索（2026-08-25）

1. **P001 — SRAC: Superpixel Retrieval-Augmented Classification for Multi-Modal Disaster Building Damage Assessment**
   - 年份：2026
   - 收录于：IGARSS 2026
   - 作者：Siheng Ning, Yuan Li, Xiaodan Zhang, Jianbin Jiao（来自用户已有项目笔记；会议页未展示作者）
   - 原文链接：https://2026.ieeeigarss.org/papers/accepted_papers.php
   - 中文摘要：当前仅能确认其面向光学–SAR建筑损毁评估，结合可学习超像素、区域嵌入和检索增强分类；完整摘要未获取。
   - 有无代码：未证实
   - 相关性：与目标任务、数据模态和BRIGHT基准完全一致，是最接近工作。
   - PDF状态：unavailable

2. **P002 — BRIGHT: A Globally Distributed Multimodal Building Damage Assessment Dataset with Very-High-Resolution for All-Weather Disaster Response**
   - 年份：2025
   - 收录于：Earth System Science Data
   - 作者：Hongruixuan Chen 等
   - 原文链接：https://essd.copernicus.org/articles/17/6217/2025/
   - 中文摘要：提出全球分布的灾前光学–灾后SAR建筑损毁数据集，并系统评估标准划分和跨事件迁移。
   - 有无代码：有: https://github.com/ChenHongruixuan/BRIGHT
   - 相关性：确定官方类别、划分、指标和跨事件风险，是所有实验协议依据。
   - PDF状态：open

3. **P003 — Superpixel Sampling Networks**
   - 年份：2018
   - 收录于：ECCV 2018
   - 作者：Varun Jampani, Deqing Sun, Ming-Yu Liu, Ming-Hsuan Yang, Jan Kautz
   - 原文链接：https://openaccess.thecvf.com/content_ECCV_2018/html/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.html
   - 中文摘要：将局部SLIC式分配松弛为可微过程，使超像素可由下游任务损失端到端学习。
   - 有无代码：有: https://github.com/NVlabs/ssn_superpixels
   - 相关性：支撑SR1中“可学习结构分区”，区别于H13固定RGB距离分区。
   - PDF状态：open

4. **P004 — SpixelFCN: Superpixel Segmentation with Fully Convolutional Networks**
   - 年份：2020
   - 收录于：CVPR 2020
   - 作者：未获取
   - 原文链接：https://github.com/fuy34/superpixel_fcn
   - 中文摘要：以全卷积网络预测局部超像素关联，形成高效、规则邻域受限的超像素分区。
   - 有无代码：有: https://github.com/fuy34/superpixel_fcn
   - 相关性：为局部九候选分配与高分辨率实现提供公开工程参考。
   - PDF状态：open

5. **P005 — Regional Semantic Contrast and Aggregation for Weakly Supervised Semantic Segmentation**
   - 年份：2022
   - 收录于：CVPR 2022
   - 作者：Tianfei Zhou, Meijie Zhang, Fang Zhao, Jianwu Li
   - 原文链接：https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Regional_Semantic_Contrast_and_Aggregation_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2022_paper.pdf
   - 中文摘要：通过类别区域记忆库保存区域语义嵌入，并进行区域级对比和聚合。
   - 有无代码：有: https://github.com/maeve07/RCA
   - 相关性：支撑从少量全局原型转向多样化区域记忆，但其弱监督目标不能直接照搬。
   - PDF状态：open

6. **P006 — Pin the Memory: Learning to Generalize Semantic Segmentation**
   - 年份：2022
   - 收录于：CVPR 2022
   - 作者：Jin Kim, Jiyoung Lee, Jungin Park, Dongbo Min, Kwanghoon Sohn
   - 原文链接：https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_Pin_the_Memory_Learning_To_Generalize_Semantic_Segmentation_CVPR_2022_paper.pdf
   - 中文摘要：用类别记忆和元学习模拟域外测试，学习跨域稳定的类别概念。
   - 有无代码：有: https://github.com/Genie-Kim/PintheMemory
   - 相关性：支撑后续“事件留出记忆”而非把验证/测试特征写入记忆库。
   - PDF状态：open

7. **P007 — HeteCD: Feature Consistency Alignment and Difference Mining for Heterogeneous Remote Sensing Image Change Detection**
   - 年份：2025
   - 收录于：ISPRS Journal of Photogrammetry and Remote Sensing
   - 作者：未获取
   - 原文链接：https://doi.org/10.1016/j.isprsjprs.2025.03.008
   - 中文摘要：采用非共享编码器、特征一致性对齐与差异挖掘处理异构变化检测。
   - 有无代码：有: https://github.com/weiAI1996/HeteCD
   - 相关性：支持保留模态独立编码；用户现有反证不支持继续迁移全图FCA或3D差异头。
   - PDF状态：unknown

8. **P008 — Building-Guided Pseudo-Label Learning for Cross-Modal Building Damage Mapping**
   - 年份：2025
   - 收录于：IGARSS 2025
   - 作者：Jiepan Li, He Huang, Yu Sheng, Yujun Guo, Wei He
   - 原文链接：https://arxiv.org/abs/2505.04941
   - 中文摘要：先从灾前光学提取建筑，再用建筑先验和低不确定伪标签优化跨模态损毁映射。
   - 有无代码：有: https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping
   - 相关性：支持保留P3的光学建筑定位锚点，但其竞赛伪标签协议需与clean设置分开报告。
   - PDF状态：open

9. **P009 — Advancing All-Weather Building Damage Mapping to the Instance Level: Outcomes and Insights from the 2026 BRIGHT Challenge**
   - 年份：2026
   - 收录于：预印本：arXiv（未证实正式收录）
   - 作者：未获取
   - 原文链接：https://arxiv.org/abs/2607.22746
   - 中文摘要：总结实例级BRIGHT挑战，指出模态独立编码、光学主导定位、后期融合以及跨事件严重度泛化的重要性。
   - 有无代码：有: https://github.com/ChenHongruixuan/BRIGHT
   - 相关性：支持将最终决策单位从像素转向建筑/区域，并要求未知事件评估。
   - PDF状态：open

10. **P010 — CGSL: Commonality Graph Structure Learning for Unsupervised Multimodal Change Detection**
   - 年份：2025
   - 收录于：ISPRS Journal of Photogrammetry and Remote Sensing
   - 作者：Jianjian Xu, Tongfei Liu, Tao Lei, Hongruixuan Chen, Naoto Yokoya, Zhiyong Lv, Maoguo Gong
   - 原文链接：https://github.com/TongfeiLiu/CGSL-for-MCD
   - 中文摘要：将异构影像构造成超像素图，在共享结构空间中比较跨模态变化。
   - 有无代码：有: https://github.com/TongfeiLiu/CGSL-for-MCD
   - 相关性：说明超像素可承载异构结构，但用户BIOS关系审计已否定“关系差本身预测Damaged漏检”，故不进入首轮模型。
   - PDF状态：unknown

可提供编号如 `P003,P005-P006` 以下载开放PDF；本轮未自动下载。
