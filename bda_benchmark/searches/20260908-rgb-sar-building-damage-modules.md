# RGB–SAR建筑损毁与高层特征比较论文检索结果

- 检索日期：2026-09-08
- 检索关键词：BRIGHT、optical-SAR change detection、building damage、multimodal foundation model、cross-event generalization
- 数据源：IEEE/Elsevier/CVF/ESSD/arXiv 论文页与论文明确关联的官方 GitHub

## P014 ChangeMamba: Remote Sensing Change Detection With Spatiotemporal State Space Model

- 年份：2024
- 收录于：IEEE TGRS
- 作者：Hongruixuan Chen, Jian Song, Chengxi Han, Junshi Xia, Naoto Yokoya
- 中文摘要：用状态空间模型进行长距离时空变化建模，覆盖二值变化、语义变化和建筑损毁任务。
- 有无代码：有：https://github.com/ChenHongruixuan/ChangeMamba
- 原文链接：https://doi.org/10.1109/TGRS.2024.3417253
- PDF：开放预印本
- 相关性说明：官方仓库已提供 BRIGHT 训练脚本和权重，是最直接的结构级对照。

## P015 SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation

- 年份：2024
- 收录于：CVPR 2024
- 作者：Xin Guo 等
- 中文摘要：使用光学、SAR及时序数据进行多粒度对比预训练与地理上下文原型学习。
- 有无代码：有：https://github.com/Jack-bo1220/SkySense
- 原文链接：https://openaccess.thecvf.com/content/CVPR2024/html/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.html
- PDF：开放
- 相关性说明：可作为冻结高层特征来源，但模型很重，且预训练传感器尺度与 BRIGHT VHR 不完全一致。

## P016 QuickQuakeBuildings: Post-Earthquake SAR-Optical Dataset for Quick Damaged-Building Detection

- 年份：2024
- 收录于：IEEE GRSL
- 作者：Yao Sun, Yi Wang, Michael Eineder
- 中文摘要：提供土耳其地震 VHR SAR–光学建筑级数据和二分类损毁基线。
- 有无代码：有：https://github.com/ya0-sun/PostEQ-SARopt-BuildingDamage
- 原文链接：https://doi.org/10.1109/LGRS.2024.3406966
- PDF：开放预印本
- 相关性说明：其模态和建筑级单位与本项目相近，可借鉴 SAR 专属预训练和建筑裁块协议。

## P017 BRIGHT: A Globally Distributed Multimodal Building Damage Assessment Dataset

- 年份：2025
- 收录于：Earth System Science Data
- 作者：Hongruixuan Chen 等
- 中文摘要：建立灾前 VHR 光学与灾后 VHR SAR 的全球多事件建筑损毁数据和标准基线。
- 有无代码：有：https://github.com/ChenHongruixuan/BRIGHT
- 原文链接：https://doi.org/10.5194/essd-17-6217-2025
- PDF：开放
- 相关性说明：是当前任务与事件隔离协议的第一依据。

## P018 Building-Guided Pseudo-Label Learning for Cross-Modal Building Damage Mapping

- 年份：2025
- 收录于：IGARSS 2025
- 作者：Jiepan Li, He Huang, Yu Sheng, Yujun Guo, Wei He
- 中文摘要：先从灾前光学提取建筑，再用低不确定性伪标签和建筑引导改进跨模态损毁图。
- 有无代码：有：https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping
- 原文链接：https://doi.org/10.1109/IGARSS55030.2025.11243835
- PDF：开放预印本
- 相关性说明：与“建筑定位优先、概率图、TTA、低不确定性筛选”高度一致。

## P019 M3ICNet: A Cross-Modal Resolution Preserving Building Damage Detection Method

- 年份：2025
- 收录于：ISPRS Journal of Photogrammetry and Remote Sensing
- 作者：未获取
- 中文摘要：面向光学–SAR损毁检测，在不简单重采样的情况下保持不同模态的原始分辨率并进行跨模态交互。
- 有无代码：未证实
- 原文链接：https://doi.org/10.1016/j.isprsjprs.2025.02.004
- PDF：未获取
- 相关性说明：可借鉴分辨率保持与跨尺度对齐思想，但当前未核实官方代码。

## P020 AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities

- 年份：2025
- 收录于：CVPR 2025
- 作者：Guillaume Astruc, Nicolas Gonthier, Clément Mallet, Loic Landrieu
- 中文摘要：用 JEPA 和尺度自适应编码器统一处理多分辨率、多尺度、多传感器数据。
- 有无代码：有：https://github.com/gastruc/AnySat
- 原文链接：https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html
- PDF：开放
- 相关性说明：适合做冻结特征探针，不适合未经验证就替换 P3。

## P021 A Semantic-Enhanced Multi-Modal Remote Sensing Foundation Model for Earth Observation

- 年份：2025
- 收录于：Nature Machine Intelligence
- 作者：Kang Wu 等
- 中文摘要：SkySense++增强多模态遥感基础表征，并在包括灾害管理在内的多类任务上评估。
- 有无代码：有：https://github.com/kang-wu/SkySensePlusPlus
- 原文链接：https://doi.org/10.1038/s42256-025-01078-8
- PDF：未获取
- 相关性说明：可提供更强语义特征，但环境、显存和预训练尺度迁移成本较高。

## P022 Advancing All-Weather Building Damage Mapping to the Instance Level

- 年份：2026
- 收录于：预印本：arXiv（未证实正式收录）
- 作者：Hongruixuan Chen 等
- 中文摘要：总结 2026 BRIGHT 实例级挑战；两个优胜方案共同采用模态专属编码、后期融合以及建筑定位与损毁识别解耦。
- 有无代码：部分可用：https://github.com/ChenHongruixuan/BRIGHT/tree/master/cvprw26
- 原文链接：https://arxiv.org/abs/2607.22746
- PDF：开放
- 相关性说明：这是与当前 RGB–SAR、三类建筑损毁和未见事件设置最接近的最新证据。

## P023 Disentangling Representation and Cross-Domain Adaptation for Unsupervised Multimodal Change Detection

- 年份：2026
- 收录于：ISPRS Journal of Photogrammetry and Remote Sensing
- 作者：Erlei Zhang, Xinyu Li, He Zong, Yuan Gao, Jinchang Ren
- 中文摘要：将异构影像分解为模态相关风格与模态共享内容，再在统一内容空间中比较变化。
- 有无代码：有：https://github.com/sighdn/DRUA
- 原文链接：https://doi.org/10.1016/j.isprsjprs.2026.04.011
- PDF：未获取
- 相关性说明：适合借鉴“内容/风格分离”来避免把 SAR 成像差异误认为损毁。

## P024 Joint Learning of Frequency-Spatio-Temporal Dependencies and Structural Context for Optical-SAR Change Detection

- 年份：2026
- 收录于：International Journal of Applied Earth Observation and Geoinformation
- 作者：Yujie Li, Xuanguang Liu, Chenguang Dai, Zhenchao Zhang, Lei Ding, Han Zhu, Ruihua Lu, Hanyun Wang
- 中文摘要：FSTS-Net结合频率–空间–时间编码、结构图建模和双路自适应融合，并在 BRIGHT 等数据上评估。
- 有无代码：作者声明将发布：https://github.com/liuxuanguang/FSTS-Net；当前可用性需复核
- 原文链接：https://doi.org/10.1016/j.jag.2026.105534
- PDF：未获取
- 相关性说明：任务最匹配，但模块较多，应拆成频率证据、结构上下文和融合三项分别验证。

## P025 UniverSat: Resolution- and Modality-Agnostic Transformers for Earth Observation

- 年份：2026
- 收录于：预印本：arXiv（未证实正式收录）
- 作者：未获取
- 中文摘要：以通用 patch encoder 将不同空间、光谱、时间分辨率与传感器映射到共享表示空间。
- 有无代码：有：https://github.com/gastruc/UniverSat
- 原文链接：https://arxiv.org/abs/2606.23503
- PDF：开放
- 相关性说明：适合未来做跨模态冻结探针，目前成熟度低于 AnySat 和 BRIGHT 官方代码。

## P026 CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders

- 年份：2023
- 收录于：NeurIPS 2023
- 作者：Anthony Fuller, Koreen Millard, James R. Green
- 中文摘要：分别编码雷达与光学影像，以跨模态对比学习和掩码重建获得共享及联合表征。
- 有无代码：有：https://github.com/antofuller/CROMA
- 原文链接：https://arxiv.org/abs/2311.00566
- PDF：开放
- 相关性说明：虽然早于检索年份，但它是“RGB/SAR高层表示比较”最直接的基础参照；输入是 Sentinel-1/2，不能直接等同 VHR RGB/SAR。

需要下载开放论文时，可使用编号，例如：`P014,P018,P022`。

