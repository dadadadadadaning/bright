# RGB–SAR 异构变化检测论文与开源代码检索

- 检索日期：2026-08-17
- 研究目标：寻找可用于灾前 RGB / 灾后 SAR 建筑损毁评估的异构变化检测代码基础
- 检索范围：光学–SAR异构变化检测、模态独立编码、跨域对齐、差异挖掘、灾害应急响应
- 数据源：ISPRS JPRS、IEEE TGRS/GRSL/J-STARS、Pattern Recognition、arXiv、论文官方 GitHub

## P001 HeteCD: Feature Consistency Alignment and Difference Mining for Heterogeneous Remote Sensing Image Change Detection

- 年份：2025
- 收录于：ISPRS Journal of Photogrammetry and Remote Sensing, 223:317–327
- 作者：Wei Jing, Haichen Bai, Binbin Song, Weiping Ni, Junzheng Wu, Qi Wang
- 核心方法：采用不共享权重的 Siamese Transformer 分别编码光学和 SAR；使用 Feature Consistency Alignment 协调异构特征分布；使用 3D Spatio-temporal Attention Difference 模块挖掘变化。
- 数据集：论文新建的雄安异构城市建筑变化数据集。
- 有无代码：有，训练、预测、数据集与模型目录均已公开：https://github.com/weiAI1996/HeteCD
- 原文链接：https://www.sciencedirect.com/science/article/pii/S0924271625001066
- DOI：https://doi.org/10.1016/j.isprsjprs.2025.03.008
- 相关性说明：输入关系和当前任务最接近，是改造成“变化条件跨模态统一”网络的首选直接代码基础；需要把二类变化头改为定位与三级损毁头，并避免在真实损毁区域强制对齐。

## P002 GLCD-DA: Change Detection from Optical and SAR Imagery Using a Global-Local Network with Diversified Attention

- 年份：2025
- 收录于：ISPRS Journal of Photogrammetry and Remote Sensing, 226:396–414
- 作者：Jie Li, Meiru Wu, Liupeng Lin, Qiangqiang Yuan, Huanfeng Shen
- 核心方法：先以图像转换网络缩小光学/SAR域差异，再通过边缘注入融合模块、CNN–Transformer全局局部交互模块和多层差异融合完成检测。
- 数据集：Gloucester I、California、Gloucester II 光学–SAR异构变化数据集。
- 有无代码：有，训练、评估、预测及预训练模型入口完整：https://github.com/WHU-SGG-RS-Pro-Group/ChangeDetection-GLCD-DA
- 原文链接：https://www.sciencedirect.com/science/article/pii/S0924271625002163
- DOI：https://doi.org/10.1016/j.isprsjprs.2025.05.029
- 相关性说明：适合借用低层边缘注入、深层全局局部交互和多层差异融合；图像翻译可能抹去真实损毁证据，不宜原样移植。

## P003 Refined Change Detection in Heterogeneous Low-Resolution Remote Sensing Images for Disaster Emergency Response

- 年份：2025
- 收录于：ISPRS Journal of Photogrammetry and Remote Sensing, 220:139–155
- 作者：Di Wang, Guorui Ma, Haiming Zhang, Xiao Wang, Yongxian Zhang
- 核心方法：从图像结构关系与语义信息出发进行异构特征对齐，面向低分辨率、多源灾害场景强调快速响应和跨数据集泛化。
- 数据集：五组异构灾害变化数据集；具体名称需在全文精读阶段逐表核验。
- 有无代码：有，包含数据、HRSICD模型、训练与推理入口：https://github.com/Lucky-DW/HRSICD
- 原文链接：https://www.sciencedirect.com/science/article/pii/S0924271624004805
- DOI：https://doi.org/10.1016/j.isprsjprs.2024.12.010
- 相关性说明：其灾害泛化目标与BRIGHT跨事件问题一致，但低分辨率设定与0.3–1米建筑损毁存在明显尺度差异。

## P004 Adversarial Feature Equilibrium Network for Multimodal Change Detection in Heterogeneous Remote Sensing Images

- 年份：2024
- 收录于：IEEE Transactions on Geoscience and Remote Sensing, 62:1–17
- 作者：Yan Pu, Maoguo Gong, Tongfei Liu, Mingyang Zhang, Tianqi Gao, Fenlong Jiang, Xiaobo Hu
- 核心方法：通过生成器与判别器的对抗学习建立异构特征平衡，用于多模态变化检测。
- 数据集：MT-Wuhan、MT-HTCD。
- 有无代码：有，并提供预训练权重：https://github.com/pu7yan9/AFENet_MCD
- 项目页：https://pu7yan9.github.io/AFENet_MCD/
- 相关性说明：可研究跨域平衡，但生成器约2.98亿参数、训练链路复杂，不适合作为当前最小可行基线。

## P005 Sentinel-1 and Sentinel-2 Data Fusion for Urban Change Detection Using a Dual Stream U-Net

- 年份：2021
- 收录于：IEEE Geoscience and Remote Sensing Letters
- 作者：Sebastian Hafner, Andrea Nascetti, Hossein Azizpour, Yifang Ban
- 核心方法：SAR和光学各走一套独立U-Net，直到最终决策阶段才融合，用结构隔离显式处理传感器差异。
- 数据集：OSCD，并补充Sentinel-1数据。
- 有无代码：有，包含预处理、训练和评估：https://github.com/SebastianHafner/DS_UNet
- 原文链接：https://ieeexplore.ieee.org/document/9570476
- DOI：https://doi.org/10.1109/LGRS.2021.3119856
- 相关性说明：结构简单、因果关系清楚，适合建立“独立编码+纯后期融合”最小基线，但分辨率和任务复杂度低于BRIGHT。

## P006 A Graph Contrastive Learning Network for Change Detection with Heterogeneous Remote Sensing Images

- 年份：2026
- 收录于：Pattern Recognition, 172:112394
- 作者：Zhiyong Lv, Sizhe Cheng, Linfu Xie, Junhuai Li, Minghua Zhao
- 核心方法：以超像素构建近邻/远邻成对图，通过图特征学习与对比目标进行无监督异构变化检测。
- 数据集：五组异构遥感变化数据集。
- 有无代码：有，包含模型与训练入口：https://github.com/ImgSciGroup/2024-GCLN
- 原文链接：https://www.sciencedirect.com/science/article/pii/S0031320325010556
- DOI：https://doi.org/10.1016/j.patcog.2025.112394
- 相关性说明：适合后续研究建筑级结构一致性或无监督预训练，不适合作为第一版像素级三级损毁主干。

## P007 Change Detection in Heterogeneous Optical and SAR Remote Sensing Images via Deep Homogeneous Feature Fusion

- 年份：2020
- 收录于：IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing
- 作者：Xiao Jiang, Gang Li, Yu Liu, Xiao-Ping Zhang, You He
- 核心方法：将语义内容与成像风格分离，并通过迭代图像风格迁移提高异构特征同质性，避免低层转换直接损害变化区域语义。
- 数据集：多组真实光学–SAR异构遥感影像；具体名称需在全文精读阶段逐表核验。
- 有无代码：未证实存在作者官方实现。
- 原文链接：https://arxiv.org/abs/2004.03830
- DOI：https://doi.org/10.1109/JSTARS.2020.2983993
- 相关性说明：可作为“对齐不能破坏变化语义”的理论依据，但不宜作为当前代码基线。

---

## 可直接复制的检索式

### 1. Google Scholar：先找完整光学–SAR变化检测方法

```text
("optical-SAR" OR "SAR-optical" OR "RGB-SAR") ("heterogeneous change detection" OR "multimodal change detection") (transformer OR "dual stream" OR "non-shared encoder")
```

```text
("pre-event optical" OR "pre-disaster optical") ("post-event SAR" OR "post-disaster SAR") ("building damage" OR "damage assessment" OR "damage mapping")
```

```text
("optical SAR" AND "change detection") ("feature alignment" OR "difference mining" OR "common-specific" OR disentanglement) GitHub
```

### 2. 搜索“只对齐未变化区域、保护真实变化”的方法

```text
("heterogeneous change detection" OR "cross-modal change detection") ("unchanged-region alignment" OR "change-aware alignment" OR "change-conditioned alignment" OR "semantic-preserving alignment")
```

```text
("optical SAR") ("preserve change" OR "damage-preserving" OR "change-specific feature") (alignment OR fusion)
```

### 3. 搜索独立编码和后期融合

```text
("optical SAR" OR "RGB SAR") ("dual encoder" OR "dual stream" OR "modality-specific encoder" OR "non-shared weights") ("late fusion" OR "decision fusion")
```

### 4. 搜索高分辨率建筑与灾害场景

```text
("very high resolution" OR VHR) (optical SAR) (building OR urban) ("change detection" OR "damage assessment")
```

```text
(BRIGHT OR "Data Fusion Contest 2025") ("building damage" OR "damage mapping") (SAR optical) (code OR GitHub)
```

### 5. Google/GitHub代码定向检索

```text
site:github.com "optical SAR" "change detection" language:Python
```

```text
site:github.com heterogeneous change detection SAR optical PyTorch
```

```text
site:github.com ("HeteCD" OR "GLCD-DA" OR "HRSICD" OR "AFENet")
```

GitHub站内搜索可分别使用：

```text
"optical SAR" "change detection" language:Python
heterogeneous-change-detection SAR optical language:Python
topic:remote-sensing-change-detection SAR optical
```

### 6. Web of Science

```text
TS=(("optical-SAR" OR "SAR-optical" OR "RGB-SAR") AND ("heterogeneous change detection" OR "multimodal change detection") AND (alignment OR fusion OR disentanglement OR "difference mining"))
```

```text
TS=(("pre-disaster optical" OR "pre-event optical") AND ("post-disaster SAR" OR "post-event SAR") AND (building NEAR/3 damage))
```

### 7. Scopus

```text
TITLE-ABS-KEY(("optical-SAR" OR "SAR-optical" OR "RGB-SAR") AND ("heterogeneous change detection" OR "multimodal change detection") AND (fusion OR alignment OR "common-specific" OR "difference mining"))
```

### 8. IEEE Xplore

```text
(("All Metadata":"optical SAR") AND ("All Metadata":"change detection") AND (("All Metadata":"heterogeneous") OR ("All Metadata":"multimodal")))
```

```text
(("All Metadata":"building damage") AND ("All Metadata":"SAR") AND ("All Metadata":"optical"))
```

### 9. arXiv

```text
all:("optical SAR") AND all:("change detection") AND (all:heterogeneous OR all:multimodal)
```

```text
all:("building damage") AND all:(SAR) AND (all:optical OR all:RGB)
```

### 10. 减少无关结果的排除词

```text
-"optical-optical" -hyperspectral -medical -video -point-cloud -cloud-removal
```

建议先使用宽检索式，再加入以下限定词逐步收窄：

```text
"official implementation" / GitHub / PyTorch / "source code"
"very high resolution" / building / disaster / semantic
"unchanged region" / "change-specific" / "event generalization"
```

---

需要下载或精读时，可输入编号，例如：`P001,P002,P003`。
