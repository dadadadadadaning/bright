# 检索结果：BRIGHT 语义标注到伪实例监督

检索日期：2026-09-01  
检索范围：正式论文页、作者预印本、论文明确关联的官方代码仓库。  
主题：RGB-SAR 建筑损毁、语义到实例、弱监督实例、中心/偏移/边界表示、实例感知语义损失。

## P001 — BRIGHT: A Globally Distributed Multimodal Building Damage Assessment Dataset with Very-High-Resolution for All-Weather Disaster Response

- 年份 / 发表：2025 / Earth System Science Data
- 作者：Hongruixuan Chen 等
- 摘要：发布灾前光学与灾后 SAR 配对的全球建筑损毁数据，覆盖多事件和多灾害类型，为全天候跨模态损毁评估提供基础。
- 代码/数据：[BRIGHT 官方仓库](https://github.com/ChenHongruixuan/BRIGHT)
- 论文：[ESSD 正式论文](https://essd.copernicus.org/articles/17/6217/2025/)
- 与本项目的直接关系：任务、数据域、语义类别和事件划分的原始依据。

## P002 — BRIGHT Challenge Outcomes: Multimodal Instance-Level Building Damage Assessment

- 年份 / 发表：2026 / arXiv 预印本
- 作者：BRIGHT Challenge organizers 等
- 摘要：总结实例级 BRIGHT 挑战、参与结果以及域内验证和未见事件测试之间的显著泛化差距。
- 代码/数据：[CVPRW 2026 challenge page](https://github.com/ChenHongruixuan/BRIGHT/blob/master/cvprw26/README.md)
- 论文：[arXiv:2607.22746](https://arxiv.org/abs/2607.22746)
- 与本项目的直接关系：给出官方实例指标、基线和跨事件泛化难点。

## P003 — Beyond Semantic to Instance Segmentation: Weakly-Supervised Instance Segmentation via Semantic Knowledge Transfer and Self-Refinement

- 年份 / 发表：2022 / CVPR
- 作者：Beomyoung Kim 等
- 摘要：BESTIE 从语义知识产生伪中心与偏移监督，并通过实例感知指导和在线自精炼降低不完整伪实例带来的漂移。
- 代码：[clovaai/BESTIE](https://github.com/clovaai/BESTIE)
- 论文：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_Beyond_Semantic_to_Instance_Segmentation_Weakly-Supervised_Instance_Segmentation_via_Semantic_CVPR_2022_paper.html)
- 与本项目的直接关系：与“语义监督产生伪实例，再训练中心/偏移”最接近的先行工作。

## P004 — Weakly Supervised Learning of Instance Segmentation With Inter-Pixel Relations

- 年份 / 发表：2019 / CVPR
- 作者：Jiwoon Ahn、Sunghyun Cho、Suha Kwak
- 摘要：IRNet 学习像素间位移和类别边界关系，并据此从弱标注生成实例级伪标签。
- 代码：[jiwoon-ahn/irn](https://github.com/jiwoon-ahn/irn)
- 论文：[CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2019/html/Ahn_Weakly_Supervised_Learning_of_Instance_Segmentation_With_Inter-Pixel_Relations_CVPR_2019_paper.html)
- 与本项目的直接关系：说明偏移场和边界关系可用于拆分粘连前景。

## P005 — Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation

- 年份 / 发表：2020 / CVPR
- 作者：Bowen Cheng 等
- 摘要：通过语义分割、实例中心和像素到中心偏移三类输出完成自底向上的全景分割。
- 代码：[google-research/deeplab2](https://github.com/google-research/deeplab2)
- 论文：[CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2020/html/Cheng_Panoptic-DeepLab_A_Simple_Strong_and_Fast_Baseline_for_Bottom-Up_Panoptic_CVPR_2020_paper.html)
- 与本项目的直接关系：对应本地 center/offset 目标与解码器的标准技术来源。

## P006 — Deep Watershed Transform for Instance Segmentation

- 年份 / 发表：2017 / CVPR
- 作者：Min Bai、Raquel Urtasun
- 摘要：将实例分割转化为能量图预测，再通过 watershed 分离接触实例。
- 代码：未核验到与论文明确对应、仍维护的官方仓库
- 论文：[CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2017/html/Bai_Deep_Watershed_Transform_CVPR_2017_paper.html)
- 与本项目的直接关系：为 RGID 的 watershed 拆分提供直接方法背景。

## P007 — TernausNetV2: Fully Convolutional Network for Instance Segmentation

- 年份 / 发表：2018 / CVPR Workshops
- 作者：Vladimir Iglovikov 等
- 摘要：针对航空建筑实例输出建筑、接触边界等类别，并通过 watershed 得到独立建筑。
- 代码：[ternaus/TernausNetV2](https://github.com/ternaus/TernausNetV2)
- 论文：[CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w4/html/Iglovikov_TernausNetV2_Fully_Convolutional_CVPR_2018_paper.html)
- 与本项目的直接关系：直接说明建筑接触边界监督对拆分粘连屋顶的价值。

## P008 — BoxInst: High-Performance Instance Segmentation With Box Annotations

- 年份 / 发表：2021 / CVPR
- 作者：Zhi Tian 等
- 摘要：利用框投影约束和成对颜色相似性，在无实例掩膜条件下训练实例分割。
- 代码：[aim-uofa/AdelaiDet](https://github.com/aim-uofa/AdelaiDet)
- 论文：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2021/html/Tian_BoxInst_High-Performance_Instance_Segmentation_With_Box_Annotations_CVPR_2021_paper.html)
- 与本项目的直接关系：提供“廉价结构监督替代完整实例掩膜”的对照范式。

## P009 — Pointly-Supervised Instance Segmentation

- 年份 / 发表：2022 / CVPR
- 作者：Bowen Cheng 等
- 摘要：研究少量点标注对实例分割的监督价值，并建立点监督与全掩膜监督之间的成本效果比较。
- 代码：[PointSup](https://github.com/facebookresearch/detectron2/tree/main/projects/PointSup)
- 论文：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Pointly-Supervised_Instance_Segmentation_CVPR_2022_paper.html)
- 与本项目的直接关系：适合作为官方真实实例的低标注预算基线。

## P010 — PointWSSIS: Weakly-Supervised Semantic and Instance Segmentation Using Points

- 年份 / 发表：2023 / CVPR
- 作者：Beomyoung Kim 等
- 摘要：用点监督联合学习语义和实例分割，强调两项任务之间的相互促进与伪监督精炼。
- 代码：[clovaai/PointWSSIS](https://github.com/clovaai/PointWSSIS)
- 论文：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Kim_PointWSSIS_Weakly-Supervised_Semantic_and_Instance_Segmentation_Using_Points_CVPR_2023_paper.html)
- 与本项目的直接关系：提供联合语义—实例训练与低成本监督的直接对照。

## P011 — PointSAM: Pointly-Supervised Segment Anything Model for Remote Sensing Images

- 年份 / 发表：2025 / IEEE Transactions on Geoscience and Remote Sensing
- 作者：Lansheng Wang 等
- 摘要：在遥感影像中以点监督适配通用分割模型，研究弱标注下的实例掩膜生成。
- 代码：[Lans1ng/PointSAM](https://github.com/Lans1ng/PointSAM)
- 论文：[IEEE Xplore](https://ieeexplore.ieee.org/document/10819451)
- 与本项目的直接关系：可作为遥感弱监督实例生成的外部强基线，但不应成为主方法依赖。

## P012 — Instance Awareness of Multi-class Semantic Segmentation Loss Functions

- 年份 / 发表：2026 / CVPR Workshops
- 作者：Abhijeet Kundu 等
- 摘要：分析多类语义损失的实例意识，并研究按连通组件、尺寸和 one-vs-rest 方式重加权语义损失。
- 代码：未核验到论文官方实现
- 论文：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026W/CV4Clinic2026/html/Kundu_Instance_Awareness_of_Multi-class_Semantic_Segmentation_Loss_Functions_CVPRW_2026_paper.html)
- 与本项目的直接关系：最直接支持“伪组件均衡损失能否改善原始语义输出”的低风险实验。

## P013 — Instance-Specific and Model-Adaptive Supervision for Semi-Supervised Semantic Segmentation

- 年份 / 发表：2023 / CVPR
- 作者：Zicheng Zhao 等
- 摘要：根据实例与模型状态调整半监督语义学习的监督强度，以减轻伪标签置信度不均带来的误导。
- 代码：未核验到论文官方实现
- 论文：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_Instance-Specific_and_Model-Adaptive_Supervision_for_Semi-Supervised_Semantic_Segmentation_CVPR_2023_paper.html)
- 与本项目的直接关系：支持对不可靠伪实例采用置信度门控，而非等权训练。

## P014 — Segment-Fusion: Hierarchical Context Fusion for Robust 3D Semantic Segmentation

- 年份 / 发表：2022 / CVPR
- 作者：Siddharth Thyagharajan 等
- 摘要：在 3D 语义分割中用区域级上下文融合增强像素预测的稳定性。
- 代码：未核验到论文官方实现
- 论文：[CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/html/Thyagharajan_Segment-Fusion_Hierarchical_Context_Fusion_for_Robust_3D_Semantic_Segmentation_CVPR_2022_paper.html)
- 与本项目的直接关系：提供“区域表示反向改善语义一致性”的跨域证据，但不是直接方法模板。

## P015 — MSNet: A Multilevel Instance Segmentation Network for Natural Disaster Damage Assessment in Aerial Videos

- 年份 / 发表：2021 / WACV
- 作者：Yuhang Zhu 等
- 摘要：面向自然灾害影像的多层级实例分割，联合利用不同尺度的目标与场景信息。
- 代码：未核验到论文官方实现
- 论文：[CVF Open Access](https://openaccess.thecvf.com/content/WACV2021/html/Zhu_MSNet_A_Multilevel_Instance_Segmentation_Network_for_Natural_Disaster_Damage_WACV_2021_paper.html)
- 与本项目的直接关系：说明灾害场景中的实例建模与多尺度目标问题具有既有研究基础。

## P016 — Creating xBD: A Dataset for Assessing Building Damage from Satellite Imagery

- 年份 / 发表：2019 / CVPR Workshops
- 作者：Ritwik Gupta 等
- 摘要：建立大规模灾害前后卫星建筑损毁数据及建筑级标注协议。
- 代码/数据：[xView2 baseline](https://github.com/DIUx-xView/xView2_baseline)
- 论文：[CVF Open Access](https://openaccess.thecvf.com/content_CVPRW_2019/html/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.html)
- 与本项目的直接关系：提供建筑级损毁评估的传统数据与任务参照。

## P017 — Pseudo-label Alignment for Semi-supervised Instance Segmentation

- 年份 / 发表：2023 / ICCV
- 作者：Xiaoke Hu 等
- 摘要：对齐教师和学生生成的伪实例标签，降低实例边界与分类伪标签的不稳定。
- 代码：未核验到论文官方实现
- 论文：[CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Hu_Pseudo-label_Alignment_for_Semi-supervised_Instance_Segmentation_ICCV_2023_paper.html)
- 与本项目的直接关系：可为后期自训练中的伪实例稳定性提供参考。

## P018 — Simple Does It: Weakly Supervised Instance and Semantic Segmentation

- 年份 / 发表：2017 / CVPR
- 作者：Anna Khoreva 等
- 摘要：系统研究弱标注生成伪掩膜并训练实例与语义分割的简洁管线。
- 代码：未核验到仍维护的论文官方实现
- 论文：[CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2017/html/Khoreva_Simple_Does_It_CVPR_2017_paper.html)
- 与本项目的直接关系：提供弱监督伪标签管线与强监督上界比较的早期标准框架。

## 已核验的代码入口汇总

| 目的 | 仓库 | 与本地代码的关系 |
|---|---|---|
| BRIGHT 数据与官方挑战 | https://github.com/ChenHongruixuan/BRIGHT | 任务定义、官方实例指标与基线 |
| 语义到伪实例 | https://github.com/clovaai/BESTIE | 最值得借鉴自精炼和缺失实例忽略策略 |
| 边界与位移关系 | https://github.com/jiwoon-ahn/irn | 对照本地 boundary/offset 目标 |
| 中心—偏移自底向上实例 | https://github.com/google-research/deeplab2 | 对照本地 center/offset decoder |
| 建筑接触边界 | https://github.com/ternaus/TernausNetV2 | 对照建筑边界与 watershed 拆分 |
| 点监督/标注成本基线 | https://github.com/facebookresearch/detectron2/tree/main/projects/PointSup | 官方实例低预算对照 |
| 联合弱监督语义与实例 | https://github.com/clovaai/PointWSSIS | 联合训练和消融设计参考 |
| 遥感点监督基础模型 | https://github.com/Lans1ng/PointSAM | 外部强基线，避免成为主方法依赖 |

