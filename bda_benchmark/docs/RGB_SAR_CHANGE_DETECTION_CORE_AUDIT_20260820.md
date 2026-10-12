# RGB–SAR 建筑损毁：核心瓶颈、可迁移模型与下一轮实验

## 1. 结论先行

当前最有证据支持的判断不是“你的 SAR 输入头太弱”，也不是“还缺一个更复杂的融合模块”，而是以下四个问题叠加：

1. **跨事件选择协议失真**：同一灾害内随机划分的验证集不能预测未见灾害测试集。BRIGHT Challenge 2026 中 46 个最终参赛队全部在未见事件上下降，验证/测试名次相关系数仅约 0.35。
2. **Damaged 类具有事件相关的决策边界**：单极化、8-bit SAR 对局部和中度破坏证据有限；成像源、入射方向、建筑材料和灾害机理变化后，Damaged 同时与 Intact、Destroyed 重叠。
3. **当前证据单位偏小**：现有多数改动在像素级或整图单原型上施加残差；而 BRIGHT 2026 两个冠军都在建筑候选、实例或 query 层面聚合 SAR 证据。
4. **冻结和安全残差既保护了 P3，也限制了学习**：本地 H6 训练冻结 RGB、定位、SAR stem、SAR stage 1/2，只允许深层少量适配；后续实验中大量 transition 接近零，说明模型能安全回退，但无法重塑浅层 SAR 表征或事件相关分类边界。

因此，应停止以 H15、H16 的方式继续串联小模块。下一轮首先要改**实验协议和证据单位**，然后再比较整模型。

## 2. 顶会/顶刊与官方竞赛方案实际上如何处理 SAR

| 方法 | 任务与证据 | SAR 的实际处理 | 可否整模型迁移到 BRIGHT | 对当前项目的判断 |
|---|---|---|---|---|
| BRIGHT Challenge 2026 第一名 | 完全相同输入，实例级，未见事件测试 | 双流编码；先定位建筑，再以 Cross-Layer Attention 将 SAR 注入损毁 Mask2Former；场景分类、动态阈值、伪标签适配 | **理论上最匹配**；论文给出的代码链接当前不可访问，暂不能直接复现 | 证明事件条件校准和建筑实例推理比继续换 SAR stem 更关键 |
| BRIGHT Challenge 2026 第二名 | 完全相同输入，实例级 | RGB/SAR 双 ResNet50-FPN；每层后期融合；1x1 融合初始化为 RGB passthrough；proposal/query 级损毁推理 | 架构可实现；公开仓库当前为空，暂无可直接加载实现 | 与 P3→A2 安全思想相近，但将 SAR 证据放到了建筑 query，而非像素残差 |
| DFC 2025 Track 2 第一名 | 完全相同输入，像素级三类 | RGB 建筑提取 + UACD 跨模态变化检测；建筑引导的低不确定性伪标签；目标场景适配 | **可以整模型迁移并重训**，官方代码完整 | 最适合先建立外部强基线；但其测试 Damaged 仍严重下降，不能当作已解决方案 |
| DFC 2025 Track 2 第二名 DF2025 | 完全相同数据/语义任务 | VMamba/STMambaBDA 体系，仓库包含 change detection、classification、checkpoints、BRIGHT 训练/测试脚本 | **可以作为第二个整模型基线**；代码仅 1 次提交、文档弱，需严格审计 | 适合验证“完整双流训练是否优于冻结 A2” |
| HeteCD, ISPRS JPRS 2025 | 光学–SAR 异构二值变化 | 非共享 Transformer 编码器、Feature Consistency Alignment、3D 时空差异挖掘 | 结构可迁移，原权重/分类头不能直接迁移 | 只迁移非共享编码思想；FCA 必须限制在可靠 Intact 区；不能再照搬深层 3D-STA |
| DSTCD, JAG 2026 | 少样本光学–SAR 二值变化 | 先做异构配准预任务，再迁移到变化检测 | 只能迁移预任务/对应模块，不能把整模型直接当三级损毁器 | 本项目稠密流近零、局部匹配无增益，当前不满足其优先使用条件 |
| CROMA, NeurIPS 2023 | Sentinel-1/2 表征学习 | 雷达、光学独立编码器 + 联合编码器；S1 为双通道后向散射 | 不能直接替换 BRIGHT 单极化 VHR SAR；只可做受控 backbone probe | 传感器、极化、分辨率差异太大，优先级低于 DFC/BRIGHT 同任务模型 |
| SARMAE, CVPR 2026 | 大规模 SAR 自监督预训练 | SAR 专属重建与光学语义锚定 | 本地 T1/T1E 已完成直接迁移反证 | 不能继续用相同残差接法调参；若再研究，只能做低层 frozen probe 或完整预训练消融 |
| SARDet-100K, NeurIPS 2024 | SAR 目标检测 | MSFA 分阶段解决输入域、模型迁移和检测适配 | 不能整模型迁移到变化分割 | 可借鉴“分阶段迁移”，不应把检测器直接嫁接到 A2 |
| FlowMamba, TCSVT 2025 | 建筑损毁/变化，光流对齐 | 差分光流 + Mamba + 类别距离损失 | 可移植模块，但本地已出现流近零、E3 匹配无收益 | 不应继续投入配准分支；可借鉴有序类别监督，不迁移光流主体 |

### 已核实的代码资源

- DFC 2025 冠军（完整训练/推理）：https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping
- DFC 2025 第二名： https://github.com/zxqyiyang/DF2025
- HeteCD： https://github.com/weiAI1996/HeteCD
- DSTCD： https://github.com/Lucky-DW/DSTCD
- CROMA： https://github.com/antofuller/CROMA
- SARMAE： https://github.com/MiliLab/SARMAE
- SARDet-100K： https://github.com/zcablii/SARDet_100K
- FlowMamba： https://github.com/flying318/FlowMamba

## 3. 你的 SAR 到底出了什么问题

### 3.1 不是单一代码错误

本地原生 SAR 路径不是简单复制三通道：`NativeSARPyramid` 使用 1 通道 Conv/GN/ConvNeXt 风格层次编码；数据端以训练集统计对 SAR 裁剪和标准化。这一实现可以稳定训练，H6-A2 也在 Destroyed 上提供过互补收益。因此不能归因于“网络完全不会读 SAR”。

### 3.2 物理信息对 Damaged 本来就不足

BRIGHT 使用来自 Capella/Umbra 的单极化 HH 或 VV、地形校正、8-bit SAR 幅度图。中度损毁常是局部屋顶破坏或结构轻微变化，而单极化强度还同时受到入射方向、材料、叠掩、阴影和斑点影响。于是：

```text
明显坍塌 → SAR 强变化 → Destroyed 较容易
局部损坏 → SAR 弱变化/与场景因素重叠 → Damaged 容易回到 Intact
```

这与本地 `Damaged gate-miss` 长期偏高、`Damaged→Intact` 为主的结果一致。

### 3.3 事件域偏移比输入归一化更关键

不同灾害的破坏机制、建筑材料、城市密度、SAR 传感器及成像几何不同。同一数值强度或同一特征距离，在不同事件中不对应同一损毁等级。BRIGHT 2026 的 46 个团队全部出现未见事件下降，说明这不是个别实现问题。

更强的直接证据是：DFC 2025 冠军在开发阶段 Damaged IoU 约 45.3%，未见事件测试约 9.75%。即便采用建筑引导、伪标签和冠军级模型，Damaged 的跨事件边界仍会崩溃。

但必须区分两件事：本仓库的标准 `train/val/test` 都包含相同 14 个事件，所以当前本地 `val→test` 下降并不是严格意义的“未见事件”下降。新增的无模型域审计得到：

| Split | 图像数 | 建筑像素占比 | 建筑内 Damaged | 建筑内 Destroyed | SAR mean/std |
|---|---:|---:|---:|---:|---:|
| train | 2918 | 14.046% | 10.808% | 6.227% | 56.875 / 45.848 |
| val | 450 | 14.683% | 10.454% | 9.368% | 56.295 / 46.027 |
| test | 878 | 14.191% | 10.365% | 6.541% | 58.469 / 46.221 |

总体 SAR 直方图差异很小（val/test JS divergence 约 0.001 bits），不支持“测试集整体 SAR 归一化完全不同”这一解释。更突出的差异发生在事件内部的空间/AOI 与损毁构成，例如：Hawaii 的建筑内 Destroyed 比例从 val 的 48.44% 变为 test 的 62.03%，La Palma 从 39.44% 变为 24.31%，Myanmar 的 Damaged 从 18.88% 变为 5.35%。这说明当前 `val→test` 问题更接近**同事件内空间区域和严重度构成变化 + 小验证集反复选择**，而真正未见事件泛化还没有被现有划分测量。

### 3.4 当前冻结策略限制了 SAR 自适应

本地 H6 冻结：

- RGB 编码与定位；
- SAR stem；
- SAR stage 1/2 及前两次下采样；
- BN。

这保证定位和基线安全，但也意味着后续模块只能在已经固定的浅层 SAR 表征上做 stage 3/4 修补。SARMAE、原型、局部匹配、超像素和 common/unique 审计反复接近零增益，部分原因是新增支路被设计成“可关闭”，而优化器确实选择了关闭。

### 3.5 像素级残差不是最合适的决策单位

P3+A2 Oracle 的 `+5.8528 pp` 证明两专家有互补信息，但不证明像素 Router 能找到它。单个 SAR 像素噪声大，同一建筑内部应共享一个损毁等级。冠军方法使用 proposal/query/instance 聚合后再判断严重度，能利用建筑内证据比例、分布和一致性。这比全图单原型、超像素平滑或逐像素 residual 更符合标签生成机制。

## 4. 为什么之前很多实验“看似合理却几乎不涨”

| 实验 | 它回答的问题 | 负结果真正说明什么 |
|---|---|---|
| A2/H6 原生 SAR | 单通道 SAR 能否稳定参与 | 能参与，但 Damaged/跨事件未解决 |
| B1/B1R Router | 能否安全修正 P3 像素 | 安全性解决，可靠选择没有解决 |
| H7 Oracle | P3/A2 是否互补 | 有很高上限，但 Oracle 不提供可部署选择规则 |
| H8 单原型 | 单图平均传感器偏差能否解释所有建筑 | 全图一个平均偏差表达力不足 |
| CCU E0/E3 | 局部相关是否与损毁相关、能否转成增益 | 相关性不等于能预测 A2 错误；匹配证据冗余或事件依赖 |
| H10 3D-STA | 深层跨模态变化能否改善 Affected gate | 更偏强破坏，Damaged gate-miss 未解决 |
| H11 T1/T1E | 外部 SARMAE 深层特征能否直接残差迁移 | 严格输入也失败，说明不是只修 PatchEmbed 就够 |
| H12/H13/H14 | 浅层旁路、超像素、unique 能量能否提供稳定弱损毁证据 | 当前统计/区域化并不能预测跨事件纠错 |

这些实验不是毫无价值；它们共同排除了“加局部对齐、加平均原型、加深层 SARMAE、加平滑区域就能解决”的假设。

## 5. 整模型迁移：可以，但必须作为新基线而不是继续嫁接

### 5.1 第一优先：DFC 2025 冠军整模型

它与当前语义任务最接近：同样是灾前 RGB、灾后 SAR、Intact/Damaged/Destroyed 像素图。建议完整迁入独立目录并建立 `External-B1`，不要将其模块拆碎后接到 H6。

必须拆成两个协议：

- `B1-clean`：只用训练事件和标签，不使用测试伪标签；用于公平比较泛化。
- `B1-transductive`：允许无标签目标事件伪标签；单独报告，不能与纯监督结果混写。

### 5.2 第二优先：DF2025/STMambaBDA 整模型

仓库自带 BRIGHT 训练/测试脚本和 checkpoint 目录，适合建立 `External-B2`。但仓库只有一次提交且缺少详细说明，迁移时必须记录：输入通道、SAR 归一化、标签映射、ignore index、裁剪、学习率、验证选择与官方评估脚本。

### 5.3 BRIGHT 2026 冠亚军

方法最匹配，但当前论文给出的第一名仓库不可访问，第二名仓库为空。因此现在只能复现论文结构，不能声称直接迁移官方代码或权重。若后续官方补齐代码，再建立实例级外部基线。

### 5.4 不能“全部迁移”到一张网络

把 HeteCD FCA、DSTCD 注册、CROMA、SARMAE、SARDet、FlowMamba、MoE 全部堆叠会带来：

- 不同传感器输入契约冲突；
- 二值变化与三级损毁目标冲突；
- 对齐目标可能抹掉 Damaged 弱差异；
- 参数和损失增加后无法做因果消融；
- 在小规模事件数据上更易过拟合。

正确的“全部分析”是让它们作为独立候选基线或单因素实验，而不是全部装进一个模型。

## 6. 下一轮应先做的三个检查

### E0：事件留一验证（最高优先级）

将 14 个训练事件按事件整体划分，至少做 4 折 group split；同一事件不得同时进入训练和验证。对 P3、H6-A2、DFC-B1、DFC-B2 统一评估：

- mIoU4、Valid mIoU；
- Damaged/Destroyed IoU；
- Damaged gate-miss；
- event mean/worst；
- 每个灾害类型的均值与方差。

如果一个模型只在旧随机验证集提高、事件留一不提高，立即停止，不再访问测试集。

这里的“事件分级”不是人为把灾害分成一级、二级、三级，而是三种由弱到强的评估层级：

1. `standard split`：train/val/test 均含 14 个相同事件，只测同事件内空间迁移；
2. `leave-one-event-out`：每次完整留出一个事件，测未知地点/事件；
3. `leave-one-disaster-type-out`：完整留出 earthquake、wildfire 等一种灾害类型，测更强的未知灾害机制迁移。

仓库脚本 `build_RGBSAR_event_holdout_folds.py` 已支持这三种事件互斥模式。必须注意：所有有监督源权重（P3、H4-A4、A2）、SAR statistics 和 class priors 都要在折内重建；加载见过留出事件的旧权重会造成监督泄漏。

### E1：建筑级互补与可靠性审计

使用官方实例标签；每栋建筑分别统计 P3 与 A2 的：

- 类别概率直方图、熵、margin；
- RGB/SAR 特征差和 signed/absolute difference；
- SAR 建筑内部均值、方差、分位数、纹理；
- 建筑大小、形状、边界质量；
- 场景级特征。

目标不是再算一个 Oracle，而是评估这些特征预测“谁正确”的 AUROC/AUPRC，并严格按留出事件测试。如果跨事件 AUPRC 没有提升，停止 Router；如果有效，再做 query selector。

### E2：冻结策略消融

以 H6-A2 为基线做三组、同样的事件留一协议：

```text
F0：当前冻结策略
F1：解冻 SAR stage2–4，stem/stage1 冻结
F2：SAR 全分支训练，RGB/定位冻结
```

使用分层学习率，SAR stem 最小、深层较大；同时冻结/固定 RGB 定位。此实验能回答“当前 SAR 上限是表征不足，还是冻结造成的”。此前只在安全残差上继续加模块，无法回答这一问题。

已实现的受控入口是 `run/run_H6_A2_freeze_ablation.sh`，三组共享 A2 初始化、训练步数、损失和验证选择，只改变 `H6_SAR_FREEZE_MODE`。F1 将 stage1→stage2 的下采样也设为可训练，避免 Stage2 在固定投影上被动适配；F2 仍保持 RGB、定位、P3 anchor 和 BN 冻结。

在冻结消融前，先运行 `run/run_H6_SAR_stage_probe.sh val`。该探针不训练神经网络，而是在训练列表上为每层建立 Intact/Damaged/Destroyed 和事件的 cosine centroid，再在验证列表上报告：Affected AUROC、条件严重度 AUROC、三级准确率和事件宏平均准确率。判读原则：若 Stage2 的 Affected AUROC 显著高于 Stage4，说明深层下采样丢失弱损毁；若事件准确率高而损毁 AUROC 低，说明 SAR 表征主要纠缠事件/传感器；若 SAR probe 良好而完整模型差，瓶颈在融合/决策头而非编码器。

完整诊断链已经整理为单一入口 `run/run_H6_A2_core_diagnosis.sh`：`preflight` 只检查代码与依赖，`smoke` 执行有限样本探针和 20 步 F0/F1/F2，`screen` 依次运行完整验证探针、Smoke、1500 步冻结消融及自动判读。自动报告使用 F0 作为基线，只有 Damaged IoU、Affected Recall、Damaged gate-miss、Event mean 和 Event worst 同时通过门槛，才允许 F1/F2 进入后续阶段；整个流程不访问 Test。

## 7. 建议的论文模型：不是另一个像素 Router

建议把方法收敛为 **Event-Calibrated Optical-Anchored Building Query Fusion**：

```text
灾前 RGB ── RGB-FPN ── 建筑定位/候选框（光学主导）
                         │
灾后 SAR ── SAR-FPN ─────┼─ 各尺度后期融合（RGB passthrough 初始化）
                         │
                    ROIAlign / building queries
                         │
       RGB、SAR、signed diff、abs diff、建筑内统计
                         │
        Occurrence: Intact / Affected
                         │
        Severity: Damaged / Destroyed
                         │
       场景条件温度/偏置校准（不使用测试标签）
```

与已有冠军的区别不能只是拼接模块。可形成论文创新的部分是：

1. **事件留一的 episodic 训练**：每个训练 episode 将一个事件视作未知域；
2. **建筑级有序校准**：场景 token 只预测严重度温度和 ordinal bias，不直接记灾害名称；
3. **光学 passthrough 后期融合**：保证 SAR 不破坏定位；
4. **显式报告跨事件 Damaged 边界稳定性**。

损失不必堆很多：一个层级交叉熵即可分解为 `building + affected + conditional severity`；可选的校准项最多保留一个。创新重点是训练/推理单位与跨事件校准，不是损失数量。

## 8. Go / No-Go 顺序

1. 先完成 P3、H6-A2 的 event-held-out 重评估。
2. 整体移植并重训 DFC 2025 冠军与 DF2025，建立两个真正外部强基线。
3. 做 H6-A2 解冻消融 F0/F1/F2。
4. 只有建筑级可靠性在未见事件上有效，才实现 query fusion。
5. 最终结构在验证事件上锁定后，只访问一次测试集。

建议晋级门槛（相对 P3 或最强外部基线）：

- leave-event-out Damaged IoU 至少 `+0.5 pp`；
- event-worst 不下降；
- 至少 10/14 事件非负；
- Damaged gate-miss 至少下降 `0.5 pp`；
- 独立种子方向一致。

## 9. 参考文献

[1] Chen et al. *Advancing All-Weather Building Damage Mapping to the Instance Level: Outcomes and Insights from the 2026 Bright Challenge*. arXiv, 2026.

[2] Chen et al. *Bright: a globally distributed multimodal building damage assessment dataset with very-high-resolution for all-weather disaster response*. Earth System Science Data, 2025.

[3] Li et al. *Building-Guided Pseudo-Label Learning for Cross-Modal Building Damage Mapping*. IGARSS, 2025.

[4] Wei et al. *Feature consistency alignment and difference mining for heterogeneous remote sensing image change detection*. ISPRS Journal of Photogrammetry and Remote Sensing, 2025.

[5] Wang et al. *Few-Shot Change Detection in Optical and SAR Remote Sensing Images for Disaster Response*. International Journal of Applied Earth Observation and Geoinformation, 2026.

[6] Fuller et al. *CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders*. NeurIPS, 2023.

[7] Li et al. *SARDet-100K: Towards Open-Source Benchmark and ToolKit for Large-Scale SAR Object Detection*. NeurIPS, 2024.

[8] FlowMamba authors. *FlowMamba: Learning Optical Flow and Mamba for Building Damage Assessment*. IEEE TCSVT, 2025.

## 10. 证据边界

- BRIGHT 2026 冠军结论来自官方 outcome paper；对应第一名代码链接目前返回不可访问，第二名仓库当前为空，因此“方法公开”不等于“当前可直接复现”。
- DFC 2025 冠军使用目标/test 伪标签的结果属于 transductive adaptation；必须与纯监督未知事件泛化分开报告。
- HeteCD、DSTCD 的成功主要在二值变化任务，不能直接外推到 Damaged/Destroyed。
- CROMA、SARMAE、SARDet 的预训练传感器或任务与 BRIGHT 不同；迁移失败不能证明预训练无价值，但也不能作为无需验证的强初始化。
