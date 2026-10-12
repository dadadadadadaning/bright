# RGB–SAR小样本跨灾害建筑损毁评估：文献—代码迁移分析

## 0. 执行结论

当前最值得写成论文的研究问题不是“再设计一个融合模块”，而是：

> 在灾前高分辨率RGB与灾后单通道显示域SAR之间，如何通过分阶段的SAR表征迁移与光学锚定的后期融合，使模型在少标注、未见灾害事件上保持稳定的中度/重度损毁判别？

推荐将后续工作压缩成一条主线：**Optical-Anchored Staged SAR Transfer（OASST，光学锚定的分阶段SAR迁移）**。它只包含两个可以分别证伪的贡献：

1. **SAR专家初始化与迁移课程**：解决“原生SAR编码器本身不够强”；
2. **深层保守融合**：在不破坏P3定位与基线输出的前提下，只注入可靠SAR增量。

建筑实例选择、记忆库和Learning-to-Defer保留为后续扩展，不再与第一轮SAR表征实验同时训练。

## 1. 当前实验局势：哪些结论已经成立

| 结论 | 本地证据 | 文献对应 |
|---|---|---|
| P3仍是测试集最可靠基线 | P3 test mIoU4约67.2725；原生SAR A2/H8约66.28 | BRIGHT强调光学主导定位与跨事件困难【A01，第13–15页；A09，第8–9页】 |
| 只改输入分布/输入头会破坏已学习特征 | P4 A1仅改归一化即下降约5.15 pp，A2/A3继续下降 | SARDet认为差距同时存在输入域、结构和迁移过程【C03，第5–8页，图3、表3–5】 |
| 原生SAR不是无用，而是“平均弱、局部互补” | P3+A2验证Oracle约+5.85 pp，14/14事件有互补 | Challenge第二名：SAR-only极弱，晚融合大幅优于早融合【A09，第8–9页，表5】 |
| 当前像素路由无法识别互补位置 | H7-CER-V2仅开放84/435907候选，净纠正18 | L2D只保证正确目标，不保证现有特征足以预测专家错误【F05/F07，方法与限制】 |
| 单一全图建筑原型上限很低 | H8真值完好原型Oracle仅约+0.266 pp，且8/14事件下降 | 事件/地域与SAR条件差异强，单原型难覆盖【A09，第6页，图6；H01，第11–18页】 |
| BRIGHT内自监督迁移没有自动变强 | A2-DCSM M1/M3显著退化，M0F也低于锁定A2 | CROMA/SARMAE依赖大规模、多样、配对预训练，不能等价为小数据同任务预训练【C01/C02】 |

因此，Oracle证明的是“两个专家的错误不完全重合”，不是“现有Router一定能学到5.85 pp”。下一步应先提高SAR专家的可迁移表征，再谈选择器。

## 2. 现有代码模块审计

### 2.1 已经成熟、应直接复用

| 模块 | 本地位置 | 作用 | 迁移难度 | 风险 |
|---|---|---|---:|---|
| P3完整基线与MICB-DSMIL | `bda_benchmark/model/DamageFormer_S8_MICB_DSMIL.py` | 稳定像素主干；建筑bag辅助头；层级Exist/Severity | 低 | MICB当前连通域不等于真实实例，主任务可复用，实例结论需谨慎 |
| P3 RGB空间锚点 | `bda_benchmark/model/DamageFormer_RGBSAR_H6_SDCR.py` 的 `P3RGBSpatialAnchor` | 严格继承RGB多尺度融合与定位头 | 低 | 必须严格加载并冻结，防止定位漂移 |
| 原生SAR金字塔 | `bda_benchmark/model/DamageFormer_RGBSAR_Heterogeneous.py` 的 `NativeSARPyramid` | 1通道、GN、ConvNeXt式层级编码 | 中 | 从随机/弱初始化训练是当前主要短板 |
| 局部稀疏对应 | 同文件 `LocalSparseCorrespondence` | 3×3局部窗口、top-k对应、熵可靠性 | 中 | 只适合深层；全层使用会放大错配 |
| 事件级评估 | `eval_*`、`summarize_H7_event_complementarity.py` 等 | 事件均值、最差事件、纠错/伤害 | 低 | 必须统一协议和checkpoint身份 |

### 2.2 可保留为对照，不应继续作为主线

| 模块 | 判定 | 原因 |
|---|---|---|
| H6-A2事件条件单原型 | 锁定为原生SAR基线 | 验证接近P3但测试更低；可用于对照，不再扩大全图原型 |
| H6-A3频率校准 | 负消融 | 验证小升、测试不升；FeaSpect同模态假设不完全适用RGB–SAR |
| H7像素级CER/CER-V2 | 暂停 | Router几乎全选P3；选择单位和特征不足 |
| H8建筑条件单原型 | 结束 | 学习系数接近0，真值原型Oracle也只有小上限 |
| A2-DCSM M1/M3 | 负结果保留 | 证明小规模BRIGHT内预训练/深层迁移不能替代大规模SAR初始化 |

### 2.3 有条件恢复的支线

`MICBDSMILHead`已将一栋建筑视作bag，聚合RGB、SAR、差分、乘积与融合特征，并分别做DamageExist与Severity；设计本身适合建筑级判别。但训练bag来自语义连通域，同类相邻建筑无法可靠分开，代码注释也明确承认这一点。因此只有在官方COCO实例标注接入后，才适合作为正式实例实验；否则只用于辅助损失或Oracle审计。

## 3. 论文模块的迁移价值、难度与风险

| 论文机制 | 建议迁移方式 | 难度 | 主要风险 | 是否进入MVP |
|---|---|---:|---|---|
| SARDet MSFA | 将光学遥感检测作为ImageNet与BRIGHT-SAR之间的中间迁移；只用一个结构视图 | 中高 | 外部权重/骨干不兼容；8位显示SAR不等于校准SAR | 是，优先级1 |
| CROMA雷达编码器 | 作为独立预训练对照，迁移深层token/adapter，不直接复制损失 | 高 | Sentinel-1双通道、10m与BRIGHT单通道VHR存在巨大尺度域差 | 是，优先级2 |
| SARMAE SARE/SARC | 若能获得兼容预训练权重，只迁移SAR编码器；监督阶段关闭预训练目标 | 高 | 2026预印本、数据域差、显示SAR无法声称物理散斑 | 作为可选对照 |
| Keep the Balance MAPA | 在冻结主干上加SAR专属低秩adapter，比较共享/专属等参数版本 | 中 | 论文为对称RGB-X；本任务应保持光学定位非对称 | 是，优先级1 |
| Keep the Balance DSCF | 只在Stage4做稀疏交互，残差零初始化 | 中 | 局部SAR散射中心不必与RGB边缘同位 | 表征有效后再做 |
| BRIGHT Challenge双阶段/Query | 光学直通初始化；建筑Proposal内聚合SAR证据 | 中高 | 需要实例标签；mAP与当前语义mIoU协议不同 | 第二阶段 |
| MS4D-Net双EMA | 无标签事件适配时，只在双专家一致区域做伪监督 | 中 | 确认偏差、自训练坍塌 | 小样本阶段 |
| STCA特征重组 | leave-one-event-out下用目标灾前RGB与源灾后损毁特征重组 | 高 | 原论文同模态；跨模态重组可能不物理 | 独立后续课题 |
| Learning-to-Defer | 冻结专家后，以建筑为单位训练keep/use/reject | 中 | 需要可靠实例与专家正确性训练标签 | 暂缓 |

## 4. 最小可行实验：先证明SAR专家迁移是否有效

### 4.1 核心假设

> 当前A2测试落后，不是原生单通道SAR没有信息，而是其SAR编码器没有经历与输入域、结构和下游任务一致的迁移课程。

### 4.2 第一阶段：只比较初始化/迁移，不改融合

统一使用P3 RGB锚点、相同A2损毁头、相同训练步数、相同采样与损失；只改变SAR路径初始化：

| ID | SAR初始化 | 目的 |
|---|---|---|
| M0 | 当前H6-A2 | 锁定原生SAR基线 |
| M1 | ImageNet权重按通道折叠/等价迁移 | 区分随机初始化与自然图像先验 |
| M2 | ImageNet → DOTA/DIOR完整检测预训练 → BRIGHT | 复现SARDet的“模型迁移+域过渡” |
| M3 | 公开SARDet或SARMAE兼容SAR权重 | 测试SAR专属预训练 |
| M4 | CROMA雷达深层表示/adapter迁移 | 测试雷达—光学联合预训练 |

控制规则：RGB编码器、定位头、BN全冻结；先只训练新SAR投影/adapter 500–1000步，再解冻SAR Stage3/4；不增加主任务损失。M2–M4若骨干不兼容，应先做adapter/特征蒸馏迁移，不应强行按名字复制权重。

### 4.3 第二阶段：只有强SAR专家通过后才比较融合

| ID | 融合 | 目的 |
|---|---|---|
| F0 | A2现有Stage3/4融合 | 控制 |
| F1 | 光学直通的Stage4零初始化残差 | Challenge第二名式安全起点 |
| F2 | Stage4低秩模态专属adapter | 验证MAPA思想 |
| F3 | Stage4局部top-k稀疏融合 | 验证DSCF/本地LocalSparseCorrespondence |

任何F变体都必须在第0步严格复现对应专家/基线，避免重演P4输入分布崩溃。

## 5. 小样本与跨事件协议

### 5.1 数据划分

先做4个代表事件的快速LOEO：野火、地震、洪水、人为爆炸各选1个事件；候选通过后再跑14事件完整LOEO。每折仅用其余事件训练，目标事件标签只能评估。

### 5.2 标注比例

每折报告100%、20%、10%、5%标注；无标签部分仅在扩展实验加入EMA一致性。必须固定同一随机种子抽样清单，避免不同方法看到不同建筑。

### 5.3 模型选择

严禁每轮查看并据此选择测试集。应使用训练事件内的事件宏验证集选checkpoint，候选锁定后对目标事件只评估一次。若持续查看测试集，论文不能再声称“未见事件泛化”。

### 5.4 指标

- Final mIoU、Valid 3类mIoU；
- Intact/Damaged/Destroyed IoU与F1；
- Localization F1；
- 事件宏平均、最差事件、标准差；
- Damaged→Intact、Destroyed→Intact；
- 若做路由：覆盖率、路由精度、净纠正数、ECE；
- 若做实例：COCO mAP/AP50/AP75和每栋建筑严重程度一致率。

## 6. 关键基线与消融矩阵

### 主表基线

1. 原始DamageFormer；
2. P3 MICB-DSMIL；
3. H6-A2原生SAR；
4. Early Fusion（4通道或等参数）；
5. Late Fusion双编码器；
6. OASST最佳变体；
7. P3+A2/新SAR专家Oracle，仅作为上限，不作为可部署方法。

### 必做消融

1. 共享编码器 vs 模态独立编码器；
2. 随机初始化 vs ImageNet vs 光学遥感中间迁移 vs SAR预训练；
3. 伪RGB接口 vs 原生1通道；
4. 无adapter vs 共享adapter vs 模态专属adapter（参数量对齐）；
5. Early、Stage3+4、仅Stage4、决策级晚融合；
6. 全局attention vs 局部稀疏top-k；
7. IID采样 vs 事件均衡采样；
8. 100/20/10/5%标签；
9. 逐事件结果和灾害类型宏平均；
10. 三种种子均值±标准差。

## 7. 建议的论文逻辑

### Research gap

现有RGB–SAR BDA通常将注意力放在融合层，但BRIGHT结果与SARDet共同表明，最大的断点可能发生在融合之前：SAR输入域、编码结构与迁移课程不一致；跨事件时，中度损毁又最容易被传感器差异吞没。

### Method claim

OASST不是组件拼盘，而是一条因果链：

1. 模态专属迁移建立可用的原生SAR专家；
2. P3光学锚点保持建筑几何与已验证基线；
3. 深层零初始化残差只注入SAR增量；
4. 事件均衡协议验证未见灾害和少标注条件。

### 可写的贡献

- 系统证明“输入适配、SAR表征迁移、融合位置”三类因素对VHR RGB–SAR损毁泛化的独立作用；
- 提出光学锚定、分阶段迁移的原生SAR专家和安全深层融合；
- 建立跨事件、少标注、逐灾害类型的严格评估，并公开失败路线与Oracle互补分析。

如果M2–M4不能让SAR专家在验证与独立测试均稳定提升，就不应包装成新模型论文；更合适的产出将是一个系统性benchmark/negative-results分析，阐明VHR显示域SAR迁移为何失败。

## 8. 近期执行顺序

1. 锁定P3与H6-A2权重、数据清单和评估脚本；
2. 新建OASST-M0，严格复现A2；
3. 实现参数量对齐的SAR专属低秩adapter，不改融合；
4. 接入一个外部初始化：优先SARDet式光学遥感中间迁移，其次CROMA/SARMAE；
5. 先跑500/1500/3000步pilot并只看验证；
6. 候选通过后跑三种子与一次独立测试；
7. 只有SAR专家稳定超过A2，才实现Stage4安全稀疏融合；
8. 官方实例标注准备完成后，再恢复建筑级选择/Query支线。

## 9. 主要论文与官方实现

- BRIGHT数据与基准：[论文主页](https://essd.copernicus.org/articles/17/6217/2025/)、[官方代码](https://github.com/ChenHongruixuan/BRIGHT)
- CROMA：[NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html)、[官方代码](https://github.com/antofuller/CROMA)
- SARDet-100K：[NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e7eb8128eb26eafbe901348df1dbacdc-Abstract-Conference.html)、[官方代码](https://github.com/zcablii/SARDet_100K)
- Keep the Balance：[CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Cai_Keep_the_Balance_A_Parameter-Efficient_Symmetrical_Framework_for_RGBX_Semantic_CVPR_2025_paper.html)
- FeaSpect：[CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zang_Feature_Spectrum_Learning_for_Remote_Sensing_Change_Detection_CVPR_2025_paper.html)
- Post-hoc Learning to Defer：[NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/bc8f76d9caadd48f77025b1c889d2e2d-Abstract-Conference.html)
- Two-Stage Learning to Defer：[NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0b17d256cf1fe1cc084922a8c6b565b7-Abstract-Conference.html)
- STCA：[Remote Sensing of Environment 2024](https://www.sciencedirect.com/science/article/pii/S0034425724004425)

