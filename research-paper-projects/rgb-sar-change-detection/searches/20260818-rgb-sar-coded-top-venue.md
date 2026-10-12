# RGB–SAR 变化检测与可迁移 SAR 表征论文检索结果

- 检索日期：2026-08-18
- 检索关键词：optical-SAR heterogeneous change detection；RGB-SAR building damage；modality-specific encoder；SAR pretraining；cross-modal fusion；few-shot disaster response
- 数据源：出版社/会议官方页面、作者官方 GitHub 仓库
- 收录规则：论文已正式发表或有明确会议录；仓库必须包含实际源代码，而不只是 README；严格区分“直接任务”和“可迁移模块”。

## P001 HeteCD: Feature Consistency Alignment and Difference Mining for Heterogeneous Remote Sensing Image Change Detection

- 年份：2025
- 收录于：ISPRS Journal of Photogrammetry and Remote Sensing
- 作者：Wei Jing 等
- 中文摘要：使用不共享权重的孪生 Transformer 分别编码光学与 SAR，以特征一致性对齐约束同语义区域，并用 3D 时空注意力差分挖掘变化。
- SAR 处理：SAR 拥有独立编码参数；不是复制成 RGB 后共享浅层；在高层特征空间进行对齐和差异挖掘。
- 有无代码：有，仓库包含数据、模型、训练和推理代码。
- 原文链接：https://www.sciencedirect.com/science/article/pii/S0924271625001066
- 代码：https://github.com/weiAI1996/HeteCD
- PDF：本轮仅检索，未下载。
- 相关性说明：最贴近“灾前光学—灾后 SAR 异构变化检测”的强期刊基线；可迁移非共享编码器、仅在语义一致区域对齐的原则。任务为二值建筑变化，不直接解决三级损毁。

---

## P002 Few-Shot Change Detection in Optical and SAR Remote Sensing Images for Disaster Response (DSTCD)

- 年份：2026
- 收录于：International Journal of Applied Earth Observation and Geoinformation
- 作者：Di Wang 等
- 中文摘要：面向少样本灾害响应，先在异构影像配准任务上学习可迁移对应关系，再把结构知识转移到光学–SAR变化检测。
- SAR 处理：不把 SAR 直接塞入 RGB 网络；先通过配准预任务学习跨模态结构，再进行任务引导迁移。
- 有无代码：有，仓库包含数据读取、模型、训练和推理代码。
- 原文链接：https://www.sciencedirect.com/science/article/pii/S1569843226000166
- 代码：https://github.com/Lucky-DW/DSTCD
- PDF：本轮仅检索，未下载。
- 相关性说明：特别适合 BRIGHT 的小样本与跨事件问题；可借鉴“配准/结构预训练→损毁任务迁移”，但原任务仍是二值变化。

---

## P003 CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders

- 年份：2023
- 收录于：NeurIPS 2023 Main Conference Track
- 作者：Anthony Fuller、Koreen Millard、James Green
- 中文摘要：分别编码雷达和多光谱影像，联合使用雷达–光学对比学习和掩码重建，并通过联合编码器获得多模态表示。
- SAR 处理：SAR/光学独立编码器；跨模态对比与联合 masked autoencoding；利用空间位置偏置而不是硬共享输入头。
- 有无代码：有，提供预训练模型和使用代码。
- 原文链接：https://papers.nips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html
- 代码：https://github.com/antofuller/croma
- PDF：本轮仅检索，未下载。
- 相关性说明：适合做 SAR 表征教师或预训练原则参照；其 Sentinel-1 双极化、10 m尺度与 BRIGHT 单通道 VHR SAR 差异大，不应直接替换 A2 主干。

---

## P004 SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection

- 年份：2024
- 收录于：NeurIPS 2024 Main Conference Track
- 作者：Yuxuan Li 等
- 中文摘要：指出 RGB 预训练到 SAR 的差距同时来自输入域、过渡域和模型结构，并提出多阶段滤波增强预训练与模型迁移框架。
- SAR 处理：先构造 SAR 风格/滤波过渡，再分阶段迁移兼容骨干，而不是只替换第一层卷积。
- 有无代码：有，官方代码和大规模 SAR 数据基准均公开。
- 原文链接：https://proceedings.neurips.cc/paper_files/paper/2024/hash/e7eb8128eb26eafbe901348df1dbacdc-Abstract-Conference.html
- 代码：https://github.com/zcablii/SARDet_100K
- PDF：本轮仅检索，未下载。
- 相关性说明：是当前提升 A2 SAR 主干的最高优先级来源；应迁移匹配的 SAR stage 权重并做分阶段解冻，而不是照搬检测头。

---

## P005 AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities

- 年份：2025
- 收录于：CVPR 2025 Highlight
- 作者：Guillaume Astruc 等
- 中文摘要：以尺度自适应 JEPA 在多数据集、11种传感器上预训练统一地球观测表征，可输出 tile、patch 或 dense 特征。
- SAR 处理：为 Sentinel-1/ALOS 等传感器定义各自输入接口与归一化，先进行尺度/传感器适配，再进入共享表示空间。
- 有无代码：有，可直接载入预训练权重并提取密集特征。
- 原文链接：https://openaccess.thecvf.com/content/CVPR2025/html/Astruc_AnySat_One_Earth_Observation_Model_for_Many_Resolutions_Scales_and_CVPR_2025_paper.html
- 代码：https://github.com/gastruc/AnySat
- PDF：本轮仅检索，未下载。
- 相关性说明：适合作为多尺度教师/线性探针候选；其 SAR 多为低分辨率多极化时序，需先做表征可迁移性审计。

---

## P006 OmniSat: Self-Supervised Modality Fusion for Earth Observation

- 年份：2024
- 收录于：ECCV 2024
- 作者：Guillaume Astruc、Nicolas Gonthier、Clément Mallet、Loïc Landrieu
- 中文摘要：利用空间对齐的多模态遥感数据自监督学习模态内编码与模态间融合，强调缺失模态和多分辨率兼容性。
- SAR 处理：传感器专属编码后再融合，不要求原始像素外观一致。
- 有无代码：有，官方训练与下游任务代码公开。
- 原文链接：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04127.pdf
- 代码：https://github.com/gastruc/OmniSat
- PDF：本轮仅检索，未下载。
- 相关性说明：适合借鉴“对齐数据上的自监督融合”，但传感器尺度与 BRIGHT 差异明显。

---

## P007 SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation

- 年份：2024
- 收录于：CVPR 2024
- 作者：Yue Guo 等
- 中文摘要：构建因子化多模态时空编码器，通过多粒度对比学习和地理上下文原型学习大规模光学–SAR表示。
- SAR 处理：光学和 SAR 保留专属编码路径，在时空层次进行组合；不是浅层通道复制。
- 有无代码：有，官方代码与权重公开。
- 原文链接：https://openaccess.thecvf.com/content/CVPR2024/html/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.html
- 代码：https://github.com/Jack-bo1220/SkySense
- PDF：本轮仅检索，未下载。
- 相关性说明：适合作为大模型教师或事件上下文编码思想来源；模型重、尺度不匹配，不适合直接成为5090上的主干对照。

---

## P008 Keep the Balance: A Parameter-Efficient Symmetrical Framework for RGB+X Semantic Segmentation

- 年份：2025
- 收录于：CVPR 2025 Oral
- 作者：Jiaxin Cai 等
- 中文摘要：对 RGB 与辅助模态采用对称的提示/适配器，并用动态稀疏跨模态融合减少全局注意力引入的噪声。
- SAR 处理：论文并非 SAR 专门方法；可迁移的是“模态专属轻量适配 + 稀疏深层融合”原则。
- 有无代码：有，训练、验证和模型权重公开。
- 原文链接：https://openaccess.thecvf.com/content/CVPR2025/html/Cai_Keep_the_Balance_A_Parameter-Efficient_Symmetrical_Framework_for_RGBX_Semantic_CVPR_2025_paper.html
- 代码：https://github.com/imcjx/KTB
- PDF：本轮仅检索，未下载。
- 相关性说明：只有 E0-C 证明某种可靠性信号可预测专家互补后，才值得借鉴其稀疏融合；不能作为当前第一步。

---

## P009 SARCLIP: A Multimodal Foundation Framework for SAR Imagery via Contrastive Language-Image Pre-Training

- 年份：2026
- 收录于：ISPRS Journal of Photogrammetry and Remote Sensing
- 作者：Chaowei Jiang 等
- 中文摘要：针对多源、多分辨率 SAR 建立大规模图文预训练，学习抗噪语义表示并支持零样本和少样本迁移。
- SAR 处理：使用 SAR 专属视觉编码器和噪声鲁棒编码，不依赖 RGB 输入统计。
- 有无代码：有，模型、推理和示例代码公开。
- 原文链接：https://www.sciencedirect.com/science/article/abs/pii/S0924271625004058
- 代码：https://github.com/CAESAR-Radi/SARCLIP
- PDF：本轮仅检索，未下载。
- 相关性说明：可做 SAR 语义教师/冻结特征线性探针；图文目标未必保留像素级建筑破坏几何，需先审计再迁移。

---

## P010 QuickQuakeBuildings: Post-Earthquake SAR-Optical Dataset for Quick Damaged-Building Detection

- 年份：2024
- 收录于：IEEE Geoscience and Remote Sensing Letters
- 作者：Yao Sun、Yi Wang、Michael Eineder
- 中文摘要：提供2023土耳其—叙利亚地震后VHR SAR、光学和建筑足迹数据，并给出建筑级二分类基线。
- SAR 处理：使用在 TerraSAR-X 上预训练的 SAR-HUB ResNet-18 权重；光学使用 ImageNet 权重；支持 SAR-only、optical-only 和双模态模式。
- 有无代码：有，数据读取、模型、训练和权重接口公开。
- 原文链接：https://doi.org/10.1109/LGRS.2024.3406966
- 代码：https://github.com/ya0-sun/PostEQ-SARopt-BuildingDamage
- PDF：本轮仅检索，未下载。
- 相关性说明：不是顶会主会论文，但与 VHR SAR 建筑损毁最贴近；最适合做低成本 SAR 迁移来源和建筑级特征审计。

---

## P011 M²CD: A Unified MultiModal Framework for Optical-SAR Change Detection with Mixture of Experts and Self-Distillation

- 年份：2025
- 收录于：IEEE Geoscience and Remote Sensing Letters
- 作者：Ziyuan Liu 等
- 中文摘要：用模态专属专家处理光学与 SAR，并以光学到 SAR 的训练路径和自蒸馏提高统一变化检测能力。
- SAR 处理：通过专家分工和迁移路径适配 SAR，而不是共享一个浅层输入头。
- 有无代码：有，基于 OpenCD 的官方实现。
- 原文链接：https://doi.org/10.1109/LGRS.2025.3590959
- 代码：https://github.com/circleLZY/M2CD
- PDF：本轮仅检索，未下载。
- 相关性说明：适合作为专家分工工程对照；你已有 H7 路由失败证据，当前不应先引入更复杂 MoE。

---

## P012 FlowMamba: Learning Optical Flow and Vision State Space Model for Building Change Detection

- 年份：2025
- 收录于：IEEE Transactions on Circuits and Systems for Video Technology
- 作者：Wei Weng 等
- 中文摘要：在建筑变化检测中联合学习光流式对齐与状态空间建模，以缓解双时相影像错位。
- SAR 处理：无专门 SAR 处理；输入主要是同质光学双时相。
- 有无代码：有，训练和推理代码公开。
- 原文链接：https://github.com/flying318/FlowMamba
- 代码：https://github.com/flying318/FlowMamba
- PDF：本轮仅检索，未下载。
- 相关性说明：可作为建筑区域对齐与长程建模参考，不能作为 SAR 表征方案。

## 明确排除

- IM4CD：论文相关，但未核实到作者公开的完整可运行代码。
- FreqHete：仓库当前主要为说明页，不能按“代码已公开”计入。
- CDPrompt：仓库说明代码仍在整理，不能按可复现模型计入。
- msGFM：官方仓库仍标注相关实现待发布，不能按可运行代码计入。

需要下载论文时，请输入编号，例如：`P001,P003` 或 `P002-P006`。

