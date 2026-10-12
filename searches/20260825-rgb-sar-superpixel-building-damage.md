# RGB-SAR 建筑损毁、异构变化检测与可学习超像素论文检索

检索日期：2026-08-25  
检索范围：BRIGHT/DFC 建筑损毁、Optical-SAR 异构变化检测、可微超像素、SAR 表征预训练。优先原论文、官方项目页和作者官方代码。

## 检索结果

1. **Bright: a globally distributed multimodal building damage assessment dataset with very-high-resolution for all-weather disaster response** — Hongruixuan Chen et al. — *Earth System Science Data*, 2025 — DOI: 10.5194/essd-17-6217-2025 — [论文](https://essd.copernicus.org/articles/17/6217/2025/) — 定义灾前光学、灾后 VHR SAR、三级建筑损毁和 14 个灾害事件的 BRIGHT 数据协议。
2. **Advancing All-Weather Building Damage Mapping to the Instance Level: Outcomes and Insights from the 2026 Bright Challenge** — Hongruixuan Chen et al. — arXiv:2607.22746, 2026 — [论文](https://arxiv.org/abs/2607.22746) · [官方资源](https://github.com/ChenHongruixuan/BRIGHT/tree/master/cvprw26) — 汇总 46 支队伍和两个优胜方法，报告模态独立编码、晚期融合、光学定位锚点及跨事件泛化差距。
3. **Building-Guided Pseudo-Label Learning for Cross-Modal Building Damage Mapping** — Jiepan Li, He Huang, Yu Sheng, Yujun Guo, Wei He — *IGARSS 2025*, pp. 228–232 — DOI: 10.1109/IGARSS55030.2025.11243835 — [论文](https://arxiv.org/abs/2505.04941) · [官方代码](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping) — 两阶段建筑提取与建筑引导低不确定性伪标签训练。
4. **HeteCD: Feature Consistency Alignment and difference mining for heterogeneous remote sensing image change detection** — Wei Jing et al. — *ISPRS Journal of Photogrammetry and Remote Sensing* 223, 2025 — DOI: 10.1016/j.isprsjprs.2025.03.008 — [论文](https://www.sciencedirect.com/science/article/pii/S0924271625001066) · [官方代码](https://github.com/weiAI1996/HeteCD) — 非共享 Siamese Transformer、FCA 和 3D 时空注意力差异模块用于光学-SAR 二值变化检测。
5. **FlowMamba: Building Damage Assessment via Optics Flow-based State Space Model** — Xin Guo et al. — *IEEE TCSVT*, accepted 2025 — [官方代码与论文入口](https://github.com/flying318/FlowMamba) — 在 BDA 中引入多尺度状态空间建模、差分光流对齐和类别距离监督。
6. **M²CD: A Unified MultiModal Framework for Optical-SAR Change Detection with Mixture of Experts and Self-Distillation** — Ziyuan Liu, Jiawei Zhang, Wenyu Wang, Yuantao Gu — arXiv:2503.19406, 2025 — [论文](https://arxiv.org/abs/2503.19406) · [官方代码](https://github.com/circleLZY/M2CD) — 在骨干阶段加入 MoE，并用 Optical-to-SAR 路径和训练期自蒸馏处理模态分布差异。
7. **SARMAE: Masked Autoencoder for SAR Representation Learning** — Danxu Liu et al. — *CVPR 2026* — arXiv:2512.16635 — [论文与官方代码](https://github.com/MiliLab/SARMAE) — 以 SAR MAE、斑点感知增强和冻结光学语义锚点进行 SAR 预训练。
8. **Superpixel Sampling Networks** — Varun Jampani, Deqing Sun, Ming-Yu Liu, Ming-Hsuan Yang, Jan Kautz — *ECCV 2018*, pp. 352–368 — [论文](https://openaccess.thecvf.com/content_ECCV_2018/html/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.html) — 将局部聚类和超像素采样写成可微模块，可由下游任务联合训练。
9. **Learning Superpixels With Segmentation-Aware Affinity Loss** — Wei-Chih Tu et al. — *CVPR 2018*, pp. 568–576 — [论文](https://openaccess.thecvf.com/content_cvpr_2018/html/Tu_Learning_Superpixels_With_CVPR_2018_paper.html) — 以分割错误感知的邻接亲和度监督弱边界，而非直接使用通用深特征生成超像素。
10. **Superpixel Segmentation With Fully Convolutional Networks** — Fengting Yang, Qian Sun, Hailin Jin, Zihan Zhou — *CVPR 2020*, pp. 13964–13973 — [论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Superpixel_Segmentation_With_Fully_Convolutional_Networks_CVPR_2020_paper.html) — 直接预测像素到规则网格单元的关联分数，并用可微下采样/上采样接入稠密预测。
11. **Boundary-Responsive Differentiable Gating for Superpixel-Based Segmentation** — Fatmaelzahraa Ahmed et al. — *CVPR 2026*, pp. 42137–42146 — [论文](https://openaccess.thecvf.com/content/CVPR2026/html/Ahmed_Boundary-Responsive_Differentiable_Gating_for_Superpixel-Based_Segmentation_CVPR_2026_paper.html) — 将区域生成、边界门控和像素级精修分开，并用邻接区域对比监督保持边界。
12. **SRAC: Superpixel Retrieval-Augmented Classification for Multi-Modal Disaster Building Damage Assessment** — 第一作者思衡宁；完整作者表公开来源尚未获取 — *IGARSS 2026*, accepted paper 2851 — [IGARSS 录用列表](https://2026.ieeeigarss.org/papers/accepted_papers.php) · [机构介绍](https://jianghai.ucas.ac.cn/index.php/zh/xshde/xsdte/461-igarss-2026) — 公开介绍称其使用 VMambaSSN 超像素和标注库检索增强处理 BRIGHT 稀有损毁类别；截至检索日未核实到公开全文或官方代码。

## 检索边界

- SRAC 的录用状态和标题由 IGARSS 官方列表确认；75.65% mIoU、+8.08 pp、VMambaSSN 与外部标注库来自机构新闻，尚不能用公开全文或代码逐项复核。
- M²CD 当前检索到的是 arXiv 预印本和作者代码，不将其写成已发表顶会/顶刊论文。
- HeteCD 和多数 Optical-SAR CD 方法处理二值变化；其权重和完整任务头不能直接等价迁移到 BRIGHT 的 Intact/Damaged/Destroyed 三级损毁。

