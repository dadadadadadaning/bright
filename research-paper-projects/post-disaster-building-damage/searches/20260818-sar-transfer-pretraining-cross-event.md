# SAR 表征迁移、专属预训练与跨事件泛化检索

检索日期：2026-08-18  
研究对象：灾前高分辨率 RGB + 灾后 SAR 的建筑损毁评估  
目标：为“停止堆叠局部匹配、转向 SAR 表征与跨事件泛化”建立可复现的论文与代码入口。

## 可直接复制的检索式

### A. SAR 表征迁移

```text
("synthetic aperture radar" OR SAR)
("representation transfer" OR "domain transfer" OR "modality adaptation" OR "continual pretraining")
(segmentation OR "dense prediction")
(code OR github)
```

```text
(SAR OR "synthetic aperture radar")
("RGB pretraining" OR ImageNet OR "model migration")
("multi-stage transfer" OR adapter OR "parameter efficient")
```

### B. SAR 专属预训练

```text
(SAR OR "synthetic aperture radar")
("self-supervised" OR "masked autoencoder" OR JEPA OR "foundation model")
(speckle OR "noise-aware" OR "semantic anchor")
(segmentation OR change detection)
```

```text
("radar-optical" OR "SAR-optical")
(pretraining OR "contrastive masked autoencoder" OR "cross-modal pretext")
("modality-specific encoder" OR "dual encoder")
```

### C. 跨事件训练与模型选择

```text
("building damage" OR disaster OR "remote sensing segmentation")
("leave-one-event-out" OR "cross-event" OR "unseen event" OR "domain generalization")
("worst-group" OR "event-balanced" OR "model selection")
```

```text
("remote sensing" OR geospatial)
("domain generalization" OR "cross-domain generalization")
(segmentation OR damage)
(benchmark OR code)
```

### D. GitHub 定向检索

```text
site:github.com SAR masked autoencoder segmentation pretrained weights
site:github.com radar optical pretraining dual encoder PyTorch
site:github.com remote sensing domain generalization segmentation
site:github.com BRIGHT building damage SAR optical
```

## 论文与官方代码

1. **P001｜SARMAE: Masked Autoencoder for SAR Representation Learning**
   - 年份 / 收录于：2026 / CVPR 2026
   - 作者：Danxu Liu, Di Wang, Hebaixu Wang, Haoyang Chen, Wentao Jiang, Yilin Cheng, Haonan Guo, Wei Cui, Jing Zhang
   - 中文摘要：构建百万级 SAR 预训练数据，使用散斑感知增强学习抗噪结构表示，并用配对光学编码作为语义锚点约束 SAR 表示；下游迁移到分类、检测和分割。
   - 对应问题：最直接回答“怎样让 SAR 编码器先学会 SAR，而不是突然更换输入头后直接监督训练”。
   - [论文](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_SARMAE_Masked_Autoencoder_for_SAR_Representation_Learning_CVPR_2026_paper.html)｜[代码与模型](https://github.com/MiliLab/SARMAE)

2. **P002｜SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection**
   - 年份 / 收录于：2024 / NeurIPS 2024
   - 作者：Yuxuan Li, Xiang Li, Weijie Li, Qibin Hou, Li Liu, Ming-Ming Cheng, Jian Yang
   - 中文摘要：指出 RGB 预训练到 SAR 的差距同时来自输入域、模型结构和迁移流程，提出滤波增强与多阶段迁移，避免只替换第一层。
   - 对应问题：提供 `RGB 初始化 → 光学遥感中间迁移 → SAR 任务迁移` 的受控实验模板。
   - [论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e7eb8128eb26eafbe901348df1dbacdc-Abstract-Conference.html)｜[代码](https://github.com/zcablii/SARDet_100K)

3. **P003｜CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders**
   - 年份 / 收录于：2023 / NeurIPS 2023
   - 作者：Anthony Fuller, Koreen Millard, James R. Green
   - 中文摘要：分别编码雷达和光学影像，以跨模态对比目标学习地点级语义，以掩码重建保存局部信息，并额外产生联合多模态表示。
   - 对应问题：支持“模态独立编码 + 预训练阶段语义迁移”，但迁移到灾害任务时应只在未变化区域对齐。
   - [论文](https://papers.nips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html)｜[代码与权重](https://github.com/antofuller/croma)

4. **P004｜Parameter Efficient Self-Supervised Geospatial Domain Adaptation**
   - 年份 / 收录于：2024 / CVPR 2024
   - 作者：Linus Scheibenreif, Michael Mommert, Damian Borth
   - 中文摘要：用无标签目标模态进行自监督参数高效适配，只更新约 1%–2% 参数，面向不同传感器和小样本下游任务减少全量微调的过拟合。
   - 对应问题：适合把公开 SAR 预训练权重迁入 BRIGHT 后，先做小型适配器预热，再做损毁监督训练。
   - [论文](https://openaccess.thecvf.com/content/CVPR2024/html/Scheibenreif_Parameter_Efficient_Self-Supervised_Geospatial_Domain_Adaptation_CVPR_2024_paper.html)｜[代码](https://github.com/HSG-AIML/GDA)

5. **P005｜Bridging Remote Sensors with Multisensor Geospatial Foundation Models (msGFM)**
   - 年份 / 收录于：2024 / CVPR 2024
   - 作者：Boran Han, Shuai Zhang, Xingjian Shi, Markus Reichstein
   - 中文摘要：在包含 Sentinel-1、Sentinel-2、RGB 和 DSM 的大规模数据上进行跨传感器掩码建模，同时利用配对与非配对数据学习统一表示。
   - 对应问题：检验“只做 SAR-only 预训练”与“跨传感器预训练”哪个更适合 BRIGHT。
   - [论文](https://openaccess.thecvf.com/content/CVPR2024/html/Han_Bridging_Remote_Sensors_with_Multisensor_Geospatial_Foundation_Models_CVPR_2024_paper.html)｜[代码](https://github.com/boranhan/Geospatial_Foundation_Models)

6. **P006｜SSL4EO-S12: A Large-Scale Multimodal, Multitemporal Dataset for Self-Supervised Learning in Earth Observation**
   - 年份 / 收录于：2023 / IEEE Geoscience and Remote Sensing Magazine
   - 作者：Yi Wang, Nassim Ait Ali Braham, Zhitong Xiong, Chenying Liu, Conrad M. Albrecht, Xiao Xiang Zhu
   - 中文摘要：发布全球、多季节的 Sentinel-1/2 无标签数据与 MoCo、DINO、MAE、data2vec 预训练实现，并覆盖分类、分割和变化检测迁移。
   - 对应问题：提供现成的雷达—光学预训练数据、预处理与权重基线，适合先做离线初始化比较。
   - [论文记录](https://elib.dlr.de/199502/)｜[代码、数据与权重](https://github.com/zhu-xlab/SSL4EO-S12)

7. **P007｜MMEarth: Exploring Multi-Modal Pretext Tasks for Geospatial Representation Learning**
   - 年份 / 收录于：2024 / ECCV 2024
   - 作者：Vishal Nedungadi, Ankit Kariryaa, Stefan Oehmcke, Serge Belongie, Christian Igel, Nico Lang
   - 中文摘要：构建 120 万地点多模态数据，使用多个预文本任务训练 MAE；实验显示仅更换为遥感数据并不足够，输入尺度、结构和多模态目标也需共同适配。
   - 对应问题：提醒预训练数据域正确不等于迁移有效，必须把 patch 尺度和密集预测需求纳入消融。
   - [论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/8085_ECCV_2024_paper.php)｜[项目与训练代码](https://vishalned.github.io/mmearth/)

8. **P008｜AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities**
   - 年份 / 收录于：2025 / CVPR 2025
   - 作者：Guillaume Astruc, Nicolas Gonthier, Clément Mallet, Loic Landrieu
   - 中文摘要：以 JEPA 和尺度自适应空间编码器处理多数据集、多传感器和多分辨率输入，并在外部数据和变化检测等任务上验证迁移。
   - 对应问题：为 VHR BRIGHT 与中分辨率公开 SAR 权重之间的尺度不匹配提供可参考的输入投影设计。
   - [论文](https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html)｜[代码](https://github.com/gastruc/AnySat)

9. **P009｜CrossEarth: Geospatial Vision Foundation Model for Domain Generalizable Remote Sensing Semantic Segmentation**
   - 年份 / 收录于：2025 / IEEE TPAMI（DOI 在线年份；卷期为 2026）
   - 作者：Ziyang Gong 等
   - 中文摘要：面向未知地区、光谱、平台和气候差异，使用 Earth-Style Injection 与多任务训练提升遥感语义分割的跨域泛化，并给出系统化跨域基准。
   - 对应问题：最适合参考“事件作为域”的训练/验证协议和域风格增强，而不是直接复制其全部模型。
   - [论文](https://doi.org/10.1109/TPAMI.2025.3649001)｜[代码与基准](https://github.com/VisionXLab/CrossEarth)

10. **P010｜CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation**
    - 年份 / 收录于：2026 / 预印本：arXiv（未证实正式收录）
    - 作者：Ziqi Ye, Ziyang Gong, Ning Liao 等
    - 中文摘要：面向 SAR 跨传感器、地区与物理条件的域偏移，提出 SAR 中心的大规模基础模型、物理描述符驱动的稀疏专家结构与跨域分割基准。
    - 对应问题：研究方向高度相关，但代码与数据尚需等待正式公开，不宜作为第一轮必须依赖的实现。
    - [论文](https://arxiv.org/abs/2603.12008)

11. **P011｜Advancing All-Weather Building Damage Mapping to the Instance Level: Outcomes and Insights from the 2026 BRIGHT Challenge**
    - 年份 / 收录于：2026 / 预印本：arXiv（挑战在 CVPRW 2026 举办）
    - 作者：未获取
    - 中文摘要：最终测试由训练中未出现的两个 2025 灾害事件构成；领先方案共同采用模态专属编码、分阶段或后期融合、光学主导定位，跨事件泛化仍明显下降。
    - 对应问题：直接证明你的首要矛盾是跨事件泛化，模型选择不能使用测试事件。
    - [论文](https://arxiv.org/abs/2607.22746)｜[数据、基线与获胜代码](https://github.com/ChenHongruixuan/BRIGHT/tree/master/cvprw26)

12. **P012｜HeteCD: Feature Consistency Alignment and Difference Mining for Heterogeneous Remote Sensing Image Change Detection**
    - 年份 / 收录于：2025 / ISPRS Journal of Photogrammetry and Remote Sensing
    - 作者：Wei Jing, Haichen Bai, Binbin Song, Weiping Ni, Junzheng Wu, Qi Wang
    - 中文摘要：使用非共享 Transformer 编码器建模光学与 SAR 特征空间，以一致性对齐和时空差异挖掘完成异构变化检测。
    - 对应问题：可作为“独立编码器”的结构对照，但其变化检测对齐损失不能无条件搬到损毁等级分类。
    - [论文](https://doi.org/10.1016/j.isprsjprs.2025.03.008)｜[代码](https://github.com/weiAI1996/HeteCD)

13. **P013｜In Search of Lost Domain Generalization (DomainBed)**
    - 年份 / 收录于：2021 / ICLR 2021
    - 作者：Ishaan Gulrajani, David Lopez-Paz
    - 中文摘要：建立统一域泛化评测与模型选择框架，指出许多复杂方法在公平调参后并不稳定超过 ERM，且模型选择规则是完整方法的一部分。
    - 对应问题：用于设计 `leave-one-event-out`、非测试域选模、多种子和统一超参数预算。
    - [论文](https://openreview.net/pdf?id=lQdXeXDoWtI)｜[代码](https://github.com/facebookresearch/DomainBed)

14. **P014｜SWAD: Domain Generalization by Seeking Flat Minima**
    - 年份 / 收录于：2021 / NeurIPS 2021
    - 作者：Junbum Cha, Sanghyuk Chun, Kyungjae Lee, Han-Cheol Cho, Seunghyun Park, Yunsung Lee, Sungrae Park
    - 中文摘要：通过密集、过拟合感知的权重平均寻找更平坦的解，以减少未知域泛化差距，并可附加到现有训练流程。
    - 对应问题：比继续加融合层更适合作为低风险跨事件优化基线。
    - [论文](https://proceedings.neurips.cc/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html)｜[代码](https://github.com/khanrc/swad)

## 检索边界

- 本清单优先收录正式论文页、出版社/会议页和论文明确关联的官方代码。
- `CrossEarth-SAR` 与 `BRIGHT Challenge 2026 outcomes` 当前按预印本记录，不把 arXiv 自动视为正式会议或期刊收录。
- 本轮未下载 PDF；项目已有 SARMAE、CROMA、SARDet-100K、BRIGHT 与跨灾害迁移的精读笔记，可直接从项目索引进入。

## 实验映射（由检索结果导出）

1. **表征迁移对照**：锁定 H6-A2，比较随机/现有初始化、SARDet 式多阶段迁移、公开 SARMAE/CROMA-SAR 初始化；主任务结构与损失完全相同。
2. **预训练对照**：只在训练集或外部无标签 SAR 上预训练，监督阶段移除预训练损失；比较 SAR-only、光学语义锚定、跨模态联合预训练。
3. **跨事件协议**：把 14 个灾害事件视为 14 个域，采用 leave-one-event-out 或分组 holdout；报告事件宏平均、最差事件、类别 IoU 与跨事件方差。
4. **模型选择**：验证事件与测试事件隔离；任何 test 指标只做最终一次报告，不能选 checkpoint、阈值或预训练配置。

