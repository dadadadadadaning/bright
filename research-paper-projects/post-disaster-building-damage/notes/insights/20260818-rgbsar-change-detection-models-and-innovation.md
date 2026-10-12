# RGB–SAR 异构变化检测代码地图与可发表创新路线

> 检索与核验日期：2026-08-18  
> 研究对象：灾前高分辨率 RGB/Optical + 灾后单通道 SAR 的建筑定位与 Intact/Damaged/Destroyed 损毁评估  
> 当前项目：BRIGHT；已有 P3、H6-A2、H7/H8 等受控实验  
> 说明：不同论文使用的数据集、标签协议和指标不同，本文不把论文中的原始分数横向排名；“效果好”主要依据同一论文内的对照、公开代码完整度以及与当前任务的结构匹配度判断。

## 1. 结论摘要

最值得作为新主干起点的不是单独某一个“最强模型”，而是以下四个功能互补、问题边界清楚的代码源：

1. **HeteCD**：提供非共享双流 Transformer、异构特征对齐和差异挖掘，是最适合改造成 BRIGHT 四类损毁模型的干净主干。
2. **DSTCD**：提供“异构配准预训练 → 少样本变化检测迁移”的训练范式，直接对应 BRIGHT 跨事件、小样本和局部错位问题。
3. **IM4CD**：给出最重要的方法启发——不假设 RGB/SAR 每个像素能直接对齐，而用局部模板匹配的响应峰、空间偏移和残差表示变化；目前没有完整官方代码，适合自行实现核心相关体。
4. **BRIGHT/P3 的层级损毁头**：保留已经验证有效的建筑定位与损毁层级建模，但不再保留“单通道 SAR 复制三次 + ImageNet RGB 归一化”的输入合同。

建议将新主线命名为：

> **CCU-BDA：Change-Conditioned Cross-Modal Unification for Building Damage Assessment**

核心不是“再加一个融合模块”，而是验证一个可证伪假设：

> RGB 与 SAR 只有在可靠对应且未变化的区域才应该统一；在疑似变化区域，跨模态差异应被保留并转化为损毁严重程度证据。全局对齐会把真实损毁和传感器差异一起抹掉。

这条路线同时解释了你的既有结果：

- H8 的全图单原型 Oracle 只有 `+0.2659 pp`，说明一个全局传感器均值不足以描述局部建筑差异；
- P3+A2 Oracle 却有 `+5.8528 pp` Valid mIoU 和 `+7.2742 pp` Damaged IoU，说明原生 SAR 中存在大量局部互补证据；
- 像素 Router 几乎不开门，说明仅凭最终类别概率无法识别这些局部证据；
- 因此下一步应在**特征比较阶段产生“对应可靠性 + 变化证据”**，而不是在最终 logits 后面猜哪个专家可信。

## 2. 冻结的研究问题

### RQ1

2023—2026 年有哪些有官方或作者代码的 RGB–SAR/异构变化检测方法，分别解决输入域、编码、对齐、配准、融合、少样本或损毁分类中的哪一个问题？

### RQ2

现有证据是否支持把 RGB 与 SAR 全局拉入同一特征空间，还是更支持局部结构匹配、变化条件对齐和后期决策？

### RQ3

结合当前实验，什么改动能够形成一个单一、清晰、可消融的论文贡献，而不是 HeteCD、CROMA、M2CD、MICB 等模块的机械拼接？

## 3. 代码与论文清单

### 3.1 与建筑损毁任务直接相关

| 方法 | 年份/出处 | 主要输入与任务 | 核心做法 | 代码 | 对当前研究的价值 |
|---|---|---|---|---|---|
| BRIGHT | 2025, ESSD | 灾前 Optical + 灾后 SAR，建筑损毁 | 全球多灾害数据、像素/实例协议 | [官方代码](https://github.com/ChenHongruixuan/BRIGHT) | 数据、划分、官方指标和当前 DamageFormer 基线必须保留 |
| BRIGHT Challenge | 2026, arXiv | 实例级建筑损毁 | 总结竞赛方案与跨事件泛化 | [官方代码](https://github.com/ChenHongruixuan/BRIGHT) | 用于确定实例级评估和论文问题，不宜把竞赛结论直接当成模块 |
| M3ICNet | 2025, ISPRS JPRS | Optical/SAR、多分辨率建筑损毁 | 多模态、多分辨率、多层交互及分辨率保持 | 未核实公开代码 | 与任务最直接，但无法作为可复现代码起点；适合做设计对照 |
| Building-Guided Pseudo-Label | 2025, IGARSS | Optical→SAR 跨模态损毁映射 | 建筑先验约束低不确定伪标签 | [作者代码](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping) | 适合少标签/半监督扩展；不是当前第一主线 |
| QuickQuakeBuildings | 2024, IEEE GRSL | 灾后 SAR–Optical 地震损毁 | 数据集与快速检测基线 | [作者代码](https://github.com/ya0-sun/PostEQ-SARopt-BuildingDamage) | 可验证模型是否只在 BRIGHT 协议有效 |
| FlowMamba | 2025, IEEE TCSVT Early Access | RGB–RGB/xBD 建筑损毁 | 光流对齐 + 状态空间模型 | [官方代码](https://github.com/flying318/FlowMamba) | 是强损毁解码器/错位建模参考，但不是现成 RGB–SAR 模型，不能直接照搬光流 |
| Multiclass Post-Earthquake Assessment | 2025, Earthquake Spectra | Optical、SAR 损伤代理、地震动、土壤 | Transformer 建筑级多源融合 | 未核实代码 | 说明建筑级元数据有用，但会改变数据假设，不作为 BRIGHT 主基线 |
| VHR SAR + Geospatial BDA | 2025, arXiv | 单时相 SAR + OSM/DSM/暴露属性 | 建筑轮廓与地理属性融合 | 论文称后续公开 | 可作 SAR-only/地理先验支线，当前代码成熟度不足 |

### 3.2 有监督或可监督迁移的 Optical–SAR 变化检测

| 方法 | 年份/出处 | 核心机制 | 代码状态 | 迁移判断 |
|---|---|---|---|---|
| **HeteCD** | 2025, ISPRS JPRS 223:317–327 | 非共享双流 Transformer；FCA 特征一致性；3D 时空差异挖掘 | [完整训练/预测代码](https://github.com/weiAI1996/HeteCD) | **主干首选**。输入合同正确，但其全局一致性目标要改成变化条件一致性 |
| **M²CD** | 2025, IEEE GRSL | 主干内模态专家、Optical-to-SAR 引导路径、自蒸馏 | [OpenCD 代码](https://github.com/circleLZY/M2CD) | 强对照基线。MoE 有价值，但 O2S 生成/蒸馏可能把真实变化误当域差异 |
| **GLCD-DA** | 2025, ISPRS JPRS 226:396–414 | 图像转换、边缘注入、CNN–Transformer 全局/局部交互、多层差异融合 | [完整代码](https://github.com/WHU-SGG-RS-Pro-Group/ChangeDetection-GLCD-DA) | 可作为“翻译后检测”基线；不建议作为主线，灾害变化区存在翻译幻觉风险 |
| **AFENet** | 2024, IEEE TGRS | 多尺度对抗域适配、差异感知对比、多流融合 | [作者代码](https://github.com/PuYan79/AFENet_MCDpytorch) | 适合作为全局域对齐代表；重点验证是否损伤 Damaged/Destroyed 可分性 |
| **HFA-PANet** | 2025, Pattern Recognition 162:111355 | 非 Siamese；层级特征对齐；渐进式相加；MK-MMD | [完整 PyTorch 代码](https://github.com/TongfeiLiu/HFA-PANet-for-MCD) | 工程简单、适合快速基线；对齐仍是全局式，创新空间有限 |
| **DSTCD** | 2026, IJAEOG 146:105100 | 异构配准源任务预训练；任务引导特征迁移；少于 20 对样本 | [官方代码](https://github.com/Lucky-DW/DSTCD) | **训练策略首选**。最贴合少样本、跨灾害和局部配准问题 |
| STSF-Net | 2026, arXiv | 模态特有/共有特征分解；先验引导融合 | [仓库](https://github.com/liuxuanguang/STSF-Net) | 思想贴近当前问题，但检索时训练代码不完整，只宜参考方法 |
| CDPrompt | 2025, IEEE TGRS 63 | 域内 Prompt；缺失模态鲁棒性 | 论文称数据与代码，未核实稳定官方仓库 | 可做“灾后 SAR 缺失/质量差”扩展，不解决当前首要比较问题 |

### 3.3 结构、匹配、图关系与无监督异构变化检测

| 方法 | 年份/出处 | 如何绕开模态差异 | 代码 | 对当前任务的启发 |
|---|---|---|---|---|
| **IM4CD** | 2026, ISPRS JPRS 233:487–501 | 放弃逐像素不变特征；局部模板匹配；响应峰空间偏移；配准与变化联合 | 暂未发现完整官方代码 | **核心思想首选**：在建筑 ROI 内建局部相关体，用峰值/熵/偏移产生可靠性和变化证据 |
| **DRUA** | 2026, ISPRS JPRS 237:22–33 | 风格/内容解耦；非配对域适配；动态变化掩膜 | [官方代码](https://github.com/sighdn/DRUA) | 可做内容/风格对照，但完全生成式解耦较重，可能牺牲严重程度细节 |
| **Bi-DiffCD** | 2025, IJCAI Main Track | 条件扩散双向模态对齐；多级互补特征协作 | [官方代码](https://github.com/Jiahuiqu/Bi-DiffCD) | 顶会且代码公开；适合作为强对齐基线，但训练成本和生成误差较高 |
| **CGSL** | 2025, ISPRS JPRS 229:92–106 | 超像素异构图；概率图编码器；公共图结构 | [官方代码](https://github.com/TongfeiLiu/CGSL-for-MCD) | 建筑天然可视为图节点；适合做结构证据或实例支线，不适合直接替代高分辨率像素解码器 |
| **GCLN** | 2026, Pattern Recognition 172:112394 | 超像素近/远邻图；图对比学习 | [作者代码](https://github.com/ImgSciGroup/2024-GCLN) | 可借用“近邻稳定、远邻差异”的可靠性特征，不建议整套搬入 |
| **FD-MCD** | 2023, ISPRS JPRS 198:99–114 | Fourier/图 Fourier 局部与非局部结构关系；频率解耦融合 | [官方代码](https://github.com/ChenHongruixuan/FDMCD) | 强结构基线；可检验 SAR 高频是否真的对应毁伤而非散斑 |
| **SDIR** | 2024, ISPRS JPRS 208:70–88 | KNN/KFN 图分离相似/不相似关系；图像回归与 MRF | [官方代码](https://github.com/yulisun/SDIR) | 适合做无学习/弱学习结构上限，不宜作为四类端到端主干 |
| **LPEM** | 2024/2025, IEEE TNNLS 36(4):6955–6969 | 保持局部邻域关系而非直接比较像素值 | [官方代码](https://github.com/yulisun/LPEM) | 对局部建筑邻域建模很贴合，可借鉴其局部保持目标 |
| **HRSICD** | 2025, ISPRS JPRS 220:139–155 | 结构关系 + 语义细化，面向灾害与低分辨率泛化 | [官方代码](https://github.com/Lucky-DW/HRSICD) | 跨场景评估方式值得迁移，但分辨率与 BRIGHT 不完全相同 |
| **CSTN** | 2024, arXiv | 双分支内容/风格分离；自重建、跨域翻译、循环重建和 CD 联训 | [官方代码](https://github.com/OMEGA-RS/CSTN) | 可作为内容–风格解耦基线；尚未核实正式顶刊录用，不作为核心文献锚点 |
| **COMIC** | 2024, Information Fusion 106:102240 | CycleGAN 同域生成 + Copula 混合依赖建模 | 未核实代码 | 统计解释性较强，但二阶段误差传播和生成伪影风险高 |

### 3.4 SAR 表征迁移和通用 RGB+X 代码

| 方法 | 年份/出处 | 贡献 | 代码 | 正确使用方式 |
|---|---|---|---|---|
| **CROMA** | 2023, NeurIPS | Radar/Optical 独立编码；跨模态对比 + Masked Reconstruction | [官方代码](https://github.com/antofuller/CROMA) | 用作 SAR/Optical 初始化或预训练对照；不要直接把同时相对齐目标用于灾前/灾后变化区 |
| **SARDet-100K / MSFA** | 2024, NeurIPS | 证明 RGB→SAR 差距同时来自输入域、模型结构和迁移过程；多阶段过滤增强 | [官方代码](https://github.com/zcablii/SARDet_100K) | 用于设计 SAR 专属迁移流程，不是变化检测网络本身 |
| **AnySat** | 2025, CVPR | 模态专属投影、尺度自适应编码、JEPA、多传感器训练 | [官方代码](https://github.com/gastruc/AnySat) | 可提供预训练权重/投影范式；高分辨率 BRIGHT 与其训练尺度存在差异 |
| **Keep the Balance** | 2025, CVPR | 对称参数高效模态适配；动态稀疏跨模态融合 | [官方代码](https://github.com/imcjx/KTB) | 适合最后验证稀疏融合，不应先于“可靠对应关系”建模 |
| **MM-OVSeg** | 2026, CVPR | Optical/SAR 独立编码与跨模态统一；开放词汇分割 | [官方代码](https://github.com/Jimmyxichen/MM-OVSeg) | 证明独立编码 + 统一空间可行；任务和分辨率差异较大，适合作为编码器参考 |

## 4. 这些方法实际分成哪几条路线

### 路线 A：强制变得可比较

代表：HeteCD、AFENet、HFA-PANet、CSTN、GLCD-DA、DRUA、Bi-DiffCD。

共同思想是把两种模态投到可比较空间，或把一种图像转换为另一种风格。优点是最终差分简单；风险是灾害区本来就应该不同，过强对齐会把损毁信号一起压平。你的 H8 原型实验和 C1 联合预训练退化已经给出同方向证据：**对齐质量高不等于损毁分类高**。

### 路线 B：不比较绝对外观，只比较结构关系

代表：FD-MCD、SDIR、LPEM、CGSL、GCLN。

它们把“一个像素值是否像另一个像素值”改成“局部邻域、频率关系或图结构是否一致”。这对 RGB–SAR 更合理，但大多面向二值、无监督和低/中分辨率，无法直接输出建筑严重程度。

### 路线 C：把配准和变化检测合并

代表：IM4CD、DSTCD，部分相关的 FlowMamba。

它们不把几何错位当作简单预处理错误，而把匹配响应、偏移和残差作为可学习信息。对于 BRIGHT，这是目前最关键但尚未被你的模型充分利用的方向：SAR 散射中心不一定与 RGB 屋顶边缘同位，稠密光流接近零并不表示局部对应问题不存在。

### 路线 D：先建立真正的 SAR 专家，再融合

代表：CROMA、SARDet-100K、AnySat、M²CD。

这条路线解释了为什么只换 1ch 输入头失败：问题同时位于输入统计、浅层结构、深层分布和迁移过程。你的 P4-A1 已经证明仅改归一化就下降约 5 pp，远大于 A2/A3 输入头差异，因此论文不应再把“1ch Stem”当核心创新。

## 5. 为什么不能直接挑论文分数最高的模型

这些论文的数字不能直接横向比较，原因包括：

- 有的是二值变化，有的是三/四级建筑损毁；
- 有的是整幅单景无监督优化，有的是大规模监督训练；
- 空间分辨率、SAR 极化、配准质量和变化占比不同；
- 有的随机划分图块，有的按灾害事件隔离；
- 很多 MCD 数据集背景占比极大，OA 很高但 Damaged 类仍可能很差。

因此代码选型应按“是否解决你的瓶颈”排序，而不是按论文表格中的 OA/F1 排序。

## 6. 与现有实验逐项对照

| 已有证据 | 能排除什么 | 指向什么 |
|---|---|---|
| P4-A1 仅改 SAR 归一化即明显退化 | “只换 1ch 输入头就是主要问题” | 需要完整 SAR 域迁移，而非局部 Stem 修补 |
| H6-A2 原生 SAR 提高 Destroyed、降低 Damaged 与 Event | “原生 SAR 没有价值” | SAR 有强结构证据，但中度损毁和跨事件泛化不稳 |
| P3+A2 Oracle `+5.8528 pp`，14/14 事件可提升 | “两个专家没有互补性” | 互补证据很多，但可靠性不可由最终概率简单识别 |
| B1R 安全但仅 `+0.0011 pp` Test mIoU4 | “低开启率门控自然能提分” | 只有门控安全性，没有可辨识性 |
| H7-CER-V2 仅打开 84/435,907，精度 21.43% | “像素级 logits 足以路由” | 需要在特征匹配阶段产生更强的结构可靠性 |
| H8 真实完好原型 Oracle 最大 `+0.2659 pp`，8/14 事件下降 | “一个事件/图像级正常原型能解释域差异” | 传感器差异是局部、建筑尺度、方向相关的 |

由此得到的关键结论是：

> 当前最缺的不是更复杂的 SAR 编码器或更强的最终 Router，而是一个能在建筑局部区域回答“这两处是否可靠对应、差异是错位还是结构破坏”的中间表征。

## 7. 建议的新模型：CCU-BDA

### 7.1 一句话方法

CCU-BDA 使用模态独立编码器，在 RGB 建筑区域内构建局部 RGB–SAR 相关体；由相关峰形状估计对应可靠性和局部位移，仅统一可靠未变化内容，同时把无法匹配但具有结构一致性的差异保留为损毁证据，最后进行层级严重程度预测。

### 7.2 架构

```text
灾前 RGB ── RGB 专属编码器 ── 建筑定位头 ── 建筑空间先验
       │               └──── stage3/4 RGB tokens ─────┐
       │                                               │
灾后 SAR ── 1ch SAR 专属编码器 ─ stage3/4 SAR tokens ─┤
                                                       ↓
                           建筑区域局部相关体 / 模板匹配
                                                       ↓
                         峰值、峰熵、互检、偏移一致性
                           ┌───────────────────────────┴─────────────────────┐
                           ↓                                                 ↓
                 可靠且疑似未变化                                    可靠变化/结构破坏
                 共享内容轻量统一                                    保留差异与 SAR 证据
                           └───────────────────────────┬─────────────────────┘
                                                       ↓
                                    层级严重程度解码器
                                      Building → Affected → Destroyed
```

### 7.3 四个模块的功能边界

#### M1：Modality-Specific Encoder

- RGB 和 SAR 输入投影、归一化、浅层参数完全独立；
- SAR 不复制三通道，不使用 ImageNet RGB 归一化；
- SAR 先经过 DSTCD 式配准迁移或 SARDet/CROMA 式分阶段初始化；
- 解决“原生 SAR 专家不稳定”，不负责融合。

#### M2：Building-Local Correlation Matching

- 只在 RGB 建筑先验扩张后的 ROI 内，在 stage3/4 搜索小窗口 SAR 候选；
- 输出相关体，而不是直接输出位移后的 SAR 图；
- 用峰值、top-1/top-2 间隔、熵、正反向互检和邻域偏移一致性衡量可靠性；
- 解决“错位、散射中心偏移和真实变化混淆”。

#### M3：Change-Conditioned Unification

- 高可靠、低变化：允许共享内容校准；
- 高可靠、高变化：禁止对齐，保留结构差异；
- 低可靠：拒绝 SAR 注入，以 RGB 定位/语义为主；
- 这是核心创新。它不是一个事后类别 Router，而是在特征形成阶段决定“统一什么、保留什么”。

#### M4：Hierarchical Severity Decoder

- 复用已验证有效的层级逻辑：建筑、受影响、摧毁；
- MICB/DSMIL 可以作为现有工程实现，但在论文中应明确它是下游损毁头，不把它和 CCU 同时包装为两个新贡献；
- 解决 Damaged 被 Intact/Destroyed 挤压的问题。

## 8. 这算创新的条件

以下写法不够创新：

> 使用 HeteCD 双编码器、CROMA 预训练、KTB 稀疏融合和 MICB 分类头构成新网络。

因为这是模块清单，没有统一研究假设。

以下写法具有清晰创新性：

> 现有 RGB–SAR 变化检测通常在全图追求特征一致性，忽略真实变化区域不应被统一。本文提出变化条件跨模态统一：以建筑局部匹配几何同时估计对应可靠性与变化状态，仅校准可靠未变化内容，并将可靠的不匹配结构显式保留为损毁严重程度证据。

它具备可发表创新的四个条件：

1. **问题新**：从“如何让模态相似”转成“何时应该相似、何时必须保持差异”；
2. **机制统一**：相关体同时服务配准可靠性、变化判断和融合，不是三个独立模块；
3. **能够证伪**：若全局对齐与条件对齐无差别，核心假设即不成立；
4. **与应用绑定**：建筑 ROI 与层级严重程度使其不是普通二值 MCD 的轻微改名。

## 9. 最小可行实验，不要一次写完整大模型

### E0：协议锁定

- P3 原始基线；
- H6-A2 原生 SAR 基线；
- 保持现有训练/验证指标与 SHA256 记录；
- 测试集在最终候选锁定前不参与选择。

### E1：HeteCD-BRIGHT Clean Baseline

- 直接移植 HeteCD 非共享编码器；
- RGB 3ch、SAR 1ch，模态专属归一化；
- 先只做晚期 concat/difference，不做 FCA、不做图像翻译；
- 用 BRIGHT 定位头 + 四类/层级损毁头替代二值头。

目的：证明离开 DamageFormer/P3 安全锚后，一个干净异构主干的真实起点。

### E2：Registration-to-Damage Transfer

- 在训练集内构造同区域局部配准/匹配预训练，不访问验证/test 标签；
- 对照：随机 SAR、ImageNet 折叠 SAR、CROMA 初始化、DSTCD 式配准迁移；
- 只迁移 SAR stage1–4 或 stage3/4，分别消融。

目的：证明收益来自迁移过程，而非只是换主干。

### E3：Building-Local Correlation

- stage3 单层相关体；
- stage4 单层相关体；
- stage3+4；
- 对照稠密光流、直接 cross-attention 和普通差分。

通过门槛：Validation Damaged IoU、Event mean 同时提高，至少 10/14 事件非负；相关熵必须能区分 P3-correct/A2-correct 区域。

### E4：Change-Conditioned Unification

- U0：不统一；
- U1：全局 FCA；
- U2：仅预测未变化区域统一；
- U3：U2 + 不可靠区域拒绝；
- U4：U3 + 可靠变化证据注入。

这组消融直接回答论文核心问题。

### E5：层级严重程度头

- 普通 3 类 softmax；
- Building→Affected→Destroyed；
- 层级头 + 既有 MICB 局部差异；
- 不再同时增加额外蒸馏、原型、频率、记忆库损失。

## 10. 关键评估设计

### 10.1 主指标

- Final mIoU；
- Valid 3-class mIoU；
- Intact/Damaged/Destroyed IoU；
- Localization F1；
- Damaged→Intact、Destroyed→Intact；
- Event mean 与 Event worst。

### 10.2 必须新增

- 每个灾害事件的指标与相对 P3 增益；
- 按灾害类型的 macro 平均；
- 匹配可靠性在 P3-only-correct、SAR-only-correct、both-wrong 三组中的分布；
- 正反向匹配一致率、偏移大小、相关峰熵；
- 3 个种子均值/标准差；
- Leave-one-event-out 或至少事件隔离的泛化实验。

### 10.3 测试集使用纪律

不建议继续“每轮训练后看 Test 再改模型”。这会把测试集变成隐性验证集，使最终论文的测试数字失去独立性。正确流程是：

```text
训练集训练
→ 验证集与逐事件验证筛选
→ 3种子确认
→ 冻结模型、阈值和checkpoint规则
→ 只运行一次最终Test
```

开发阶段若必须监控 Test，应将它明确标成 `development-test`，并另留一组完全未访问的事件作 final holdout。

## 11. 推荐阅读和复现顺序

### 第一组：必须先读/先跑

1. HeteCD：理解非共享双流和 FCA；先在 BRIGHT 做干净基线。
2. DSTCD：理解配准到变化检测的特征迁移。
3. IM4CD：理解为什么局部匹配残差比逐像素不变特征更合理。
4. BRIGHT + Challenge：锁定任务协议和跨事件问题。
5. SARDet-100K：理解 SAR 迁移不是只改第一层。

### 第二组：强对照

6. M²CD：模态专家和自蒸馏。
7. AFENet / HFA-PANet：全局域对齐代表。
8. GLCD-DA / Bi-DiffCD：图像转换和生成式对齐代表。
9. FD-MCD / LPEM：结构关系代表。

### 第三组：后续扩展

10. CROMA / AnySat：预训练初始化。
11. CGSL / GCLN：建筑图或实例支线。
12. Keep the Balance：已有可靠性以后再做稀疏融合。

## 12. 代码迁移建议

### 主工程基座

优先从 HeteCD 创建独立实验目录，而不是继续把所有新模块叠在 `DamageFormer` 上。保留 BRIGHT 的数据加载、标签协议、统一评估器和层级损毁头。

### 迁移映射

| 来源 | 只迁移什么 | 不迁移什么 |
|---|---|---|
| HeteCD | 非共享编码器、stage 特征接口、差异模块骨架 | 原始二值头、无条件 FCA |
| DSTCD | 配准预训练任务、task-guided feature transfer | 其具体低分辨率数据假设 |
| IM4CD | 局部相关体、峰偏移和匹配可靠性思想 | CRF 全流程和手工后处理 |
| BRIGHT/P3 | 数据、评估、RGB 建筑定位、层级严重程度逻辑 | SAR 复制三通道和 P3 完整安全锚 |
| M²CD | 作为 MoE/self-distillation 对照 | 不在首轮主模型中加入生成路径 |
| KTB | 后期稀疏融合对照 | 在没有可靠性之前不启用动态稀疏交互 |

## 13. 风险与停止规则

### 风险 1：局部相关体只学到纹理/散斑

检测：相关熵不能区分互补正确与错误区域；对 SAR 随机 speckle 增强极不稳定。  
停止：E3 在三种子上不提高 Damaged 与 Event，停止 CCU，不继续堆融合。

### 风险 2：变化条件掩膜退化成全关或全开

检测：统一率 <1% 或 >95%；事件间方差很大。  
处理：用匹配几何而不是最终类别概率作为门控输入；加入覆盖率报告，不先加新损失。

### 风险 3：验证提高、测试下降

处理：事件隔离、leave-one-event-out、3 种子、单次最终 Test；不要以 Test 反复调阈值。

### 风险 4：贡献变成模块拼接

停止条件：如果删掉“change-conditioned”后模型仍同样有效，就不能把 CCU 写成核心贡献，需要回到问题定义。

## 14. 最终建议

现在不要继续 H8 全局原型、H7 像素 Router、1ch Stem 比较，也不要直接进入实例 Query。建议下一阶段只做下面四件事：

1. 用 HeteCD 代码建立全新的 BRIGHT 异构双流基线；
2. 加入 DSTCD 风格的配准到损毁迁移，建立可靠原生 SAR 专家；
3. 自行实现 IM4CD 启发的建筑局部相关体，验证它能否识别 P3/A2 互补区域；
4. 只有 E3 成功后，加入变化条件统一和层级严重程度解码。

论文主线应始终围绕一个问题：

> 如何在灾前 RGB 与灾后 SAR 不同成像机制、局部错位和真实损毁同时存在时，选择性统一可比较内容，并保留能够区分 Damaged 与 Destroyed 的异构结构差异？

这比“更换 SAR 输入头”“两个专家后期 Router”“增加全局原型”更集中，也比简单拼接顶会模块更容易通过消融证明。

## 参考资料（核验入口）

1. BRIGHT, *Earth System Science Data*, 2025.
2. BRIGHT Challenge, arXiv:2607.22746, 2026.
3. Jing et al., HeteCD, *ISPRS Journal of Photogrammetry and Remote Sensing*, 2025.
4. Liu et al., M²CD, *IEEE Geoscience and Remote Sensing Letters*, 2025.
5. Li et al., GLCD-DA, *ISPRS Journal of Photogrammetry and Remote Sensing*, 2025.
6. Wang et al., DSTCD, *International Journal of Applied Earth Observation and Geoinformation*, 2026.
7. Chen et al., IM4CD, *ISPRS Journal of Photogrammetry and Remote Sensing*, 2026.
8. Zhang et al., DRUA, *ISPRS Journal of Photogrammetry and Remote Sensing*, 2026.
9. Zhao et al., Bi-DiffCD, *IJCAI*, 2025.
10. Xu et al., CGSL, *ISPRS Journal of Photogrammetry and Remote Sensing*, 2025.
11. Chen et al., FD-MCD, *ISPRS Journal of Photogrammetry and Remote Sensing*, 2023.
12. Sun et al., SDIR, *ISPRS Journal of Photogrammetry and Remote Sensing*, 2024.
13. Sun et al., LPEM, *IEEE Transactions on Neural Networks and Learning Systems*, 2024/2025.
14. Fuller et al., CROMA, *NeurIPS*, 2023.
15. Li et al., SARDet-100K, *NeurIPS*, 2024.
16. Astruc et al., AnySat, *CVPR*, 2025.
17. Cai et al., Keep the Balance, *CVPR*, 2025.
18. Guo et al., FlowMamba, *IEEE TCSVT*, 2025.
