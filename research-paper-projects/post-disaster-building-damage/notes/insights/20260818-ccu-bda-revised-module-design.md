# CCU-BDA 模块重构与证据驱动实验设计

日期：2026-08-18

## 1. 研究方向与当前瓶颈

研究任务：灾前高分辨率 RGB + 灾后单通道 SAR 的建筑损毁评估，输出
Background、Intact、Damaged、Destroyed，并关注跨灾害事件泛化。

已知实验事实：

- P3 的伪 RGB-SAR 输入仍是强基线；
- 原生 SAR 路线能恢复到接近 P3，并在 Destroyed 上显示优势；
- P3 与 H6-A2 存在很高的 Oracle 互补上限，但像素级 Router 无法可靠选择；
- 全图或单原型校准难以稳定改善 Damaged 和跨事件指标；
- 因而当前缺少的不是另一个融合模块，而是“在特征比较时判断 RGB/SAR
  对应是否可信”的可解释证据。

## 2. 论文模块可迁移性审计

| 来源 | 原论文可支持的结论 | 对本项目可迁移模块 | 不应直接迁移或夸大的内容 | 难度/风险 |
|---|---|---|---|---|
| HeteCD | 非共享双流与异构差异建模适用于 optical/SAR CD | clean dual-stream、3D difference 对照 | 不把全图 FCA 当最终方法；当前 C0 不是严格 HeteCD 复现 | 中；二值变化到严重程度有任务差距 |
| IM4CD | 将多模态 CD 解释为局部 image matching，并使用响应峰空间偏移 | 小窗口相关体、峰值、偏移 | entropy、cycle、building ROI、severity reliability 是本项目新假设 | 高；需自行实现与验证 |
| DSTCD | registration source task 可向 few-shot optical/SAR CD 迁移结构/语义知识 | 只用于 SAR/匹配投影初始化 | 不在损毁训练中长期施加强 registration loss | 中高；可能抹平真实破坏 |
| CROMA | 雷达与光学独立编码及跨模态预训练有效 | SAR encoder 初始化候选 | 原始 Sentinel-1/2 通道与同步时相目标不可原样照搬 | 中高；输入与时间条件不同 |
| SARDet-100K | RGB→SAR 差距同时来自数据域、结构和迁移过程 | 完整 SAR 路径迁移对照 | 不迁移检测头，不把输入层替换当完整迁移 | 中高；检测到分割有任务差距 |
| Keep the Balance | 无条件全局 RGB-X 交互可能引入噪声，动态稀疏交互更稳健 | 后续可靠区域稀疏注入参考 | E0 前不加入新的稀疏门控 | 中；pre/post 与同步 RGB-X 不同 |
| BRIGHT | 固定灾前 optical、灾后 SAR 和损毁评估协议 | 数据、标签、事件指标、官方测试协议 | 不因模块迁移改变任务定义 | 低 |

原始证据：

- IM4CD 官方论文页：<https://www.sciencedirect.com/science/article/pii/S0924271626000559>
- CROMA（NeurIPS 2023）：<https://proceedings.neurips.cc/paper_files/paper/2023/hash/11822e84689e631615199db3b75cd0e4-Abstract-Conference.html>
- SARDet-100K（NeurIPS 2024）：<https://proceedings.neurips.cc/paper_files/paper/2024/hash/e7eb8128eb26eafbe901348df1dbacdc-Abstract-Conference.html>
- Keep the Balance（CVPR 2025）：<https://openaccess.thecvf.com/content/CVPR2025/html/Cai_Keep_the_Balance_A_Parameter-Efficient_Symmetrical_Framework_for_RGBX_Semantic_CVPR_2025_paper.html>
- BRIGHT 官方仓库：<https://github.com/ChenHongruixuan/BRIGHT>

## 3. 修订后的核心研究命题

不是“把 RGB 与 SAR 尽可能对齐”，而是：

> 在 RGB 建筑空间锚点内，先估计局部 RGB-SAR 对应的可靠性；只对可靠的
> Intact 内容施加统一约束，把可靠的不匹配保留为损毁证据，并拒绝不可靠 SAR。

逻辑链：

```text
Building ROI
    ↓
Local RGB-SAR matching geometry
    ↓
Correspondence reliability
    ↓
Reliable Intact: unify
Reliable mismatch: preserve as damage evidence
Unreliable match: reject/down-weight
    ↓
Intact / Damaged / Destroyed
```

创新边界：

- “局部匹配视角”由 IM4CD 启发；
- “peak/entropy/margin/offset/cycle 形成建筑对应可靠性”是待验证新假设；
- “只对 reliable Intact 统一”是 CCU 的任务特定新假设；
- “局部匹配几何提高跨事件泛化”必须由事件级实验证明，不能预先宣称。

## 4. 为什么先做 E0，而不是直接训练 C0-C4

当前 C2/C3 的合理性依赖一个前提：已有 RGB/SAR 深层特征的局部相关统计应
与损毁等级相关。如果三类的 peak、entropy、offset、cycle 几乎相同，那么：

1. 局部相关体没有 severity/reliability 信号；
2. 后续门控只能从任务损失中学习脆弱捷径；
3. 调窗口、top-k、学习率不会解决表征空间不对应；
4. 应先改 SAR 初始化或共同投影，而非继续加融合。

因此 E0 是最低成本的否证实验，不训练、不访问测试集。

## 5. E0 的精确定义

输入：锁定的 H6-A2 `best_val_miou.pth`。

特征：Stage3/4 `rgb_common` 与 `sar_common`，在 sensor-gap prototype 修正前
截取，避免用已经校准过的输出制造循环论证。

匹配：

- 固定恒等 Q/K/V/Output 投影；
- 5×5 局部窗口；
- top-9 soft matching；
- RGB→SAR 与 SAR→RGB 循环检查。

统计：

- correlation peak、entropy、margin；
- expected offset magnitude；
- cycle error / consistency；
- analytic reliability；
- Intact、Damaged、Destroyed 三类；
- 全局和逐事件 Cohen's d、方向一致事件数。

真值标签仅用于结果分层，不输入匹配器。测试集永不访问。

初始放行门槛：

```text
best |d(Intact→Destroyed)| >= 0.25
comparable events >= 10
direction-consistent events >= 10
```

这只是进入 E2 的最低标准，不代表模型可发表。

## 6. E1-E5 最小实验矩阵

### E1：Clean heterogeneous BDA baseline

目的：建立不含匹配、对齐、迁移预训练和层级头的异构双流基线。

两种可选实现应分开命名：

- E1-H：严格 HeteCD MiT/SegFormer 风格非共享编码器；
- E1-P：项目内 Swin RGB + ConvNeXt/GN Native-SAR 工程基线。

不能把 E1-P 写成 HeteCD 复现。

### E2：Difference operator

固定 E1 的骨干、采样、损失和训练步数，只替换：

- E2-A：absolute difference；
- E2-B：HeteCD-style 3D difference；
- E2-C：building-local correlation geometry。

E2-C 进入下一阶段的条件：Damaged IoU、Event mean 均优于 E2-A，至少
10/14 事件不下降，Destroyed 不回退。

### E3：Alignment scope

固定 E2 最佳差异算子，只比较：

- E3-A：no alignment；
- E3-B：global alignment；
- E3-C：GT Intact-only alignment；
- E3-D：reliable GT Intact-only alignment。

只增加一个对齐项：

```text
L_total = L_loc + L_damage + lambda_ccu * L_ccu

L_ccu = sum(I[y=Intact] * stopgrad(rho) * d(R,S))
        / (sum(I[y=Intact] * stopgrad(rho)) + eps)
```

关键防护：

- `rho` 在 L_CCU 中 detach，避免可靠性塌缩为 0；
- background 不默认视作 unchanged；
- damaged/destroyed 不强制推远；
- 推理时不使用 GT，GT 只决定训练正则的生效位置；
- 不再新增 contrastive、reconstruction、KD、router、prototype loss。

### E4：SAR transfer

只改变 SAR 初始化与迁移路径：

- E4-A：当前监督训练初始化；
- E4-B：SARDet-style full-path migration；
- E4-C：CROMA radar initialization；
- E4-D：DSTCD registration-pretraining initialization。

预训练目标不应在有损毁标签的监督阶段长期保留。

### E5：Severity parameterization

最后比较 flat 4-class 与 Building→Affected→Destroyed。只验证严重程度参数化，
不同时更改骨干、匹配和迁移。

## 7. 最小可行实验与风险

### MVP

1. E0 smoke（2 张）检查接口；
2. E0 full validation（450 张）；
3. 若通过，只实现 E2-A 与 E2-C；
4. 若 E2-C 通过，再做 E3-A/C/D；
5. 最后才扩展 HeteCD 3D difference、迁移初始化和层级头。

### 主要风险

| 风险 | 早期信号 | 处理 |
|---|---|---|
| RGB/SAR common feature 不能直接相关 | 三类 effect size 接近 0 | 转 E4 改初始化/投影 |
| padding 产生虚假偏移 | 图像边界 peak/margin 异常 | 审计时排除窗口边缘或加 valid mask |
| 类别像素量主导结论 | global 显著、事件方向混乱 | 以事件方向一致性作为硬门槛 |
| reliable gate 塌缩 | rho 接近 0 且 L_CCU 虚低 | 在 L_CCU 中 detach rho |
| 只提升 volcano/explosion | mean 上升、worst 下降、std 增大 | 不进入 Full，报告事件偏置 |
| test 被频繁使用 | 根据测试结果调窗口/权重 | 只在验证锁定后测试一次 |

## 8. 论文结构建议

论文类型：Technique paper。

逻辑：

1. optical-SAR BDA 的跨模态不一致混合了传感器差异、局部几何偏移和真实破坏；
2. 逐像素差分与全局一致性无法区分三者；
3. 建筑局部匹配几何提供对应置信度与结构变化证据；
4. 可靠稳定内容应统一，真实变化应保留，不可靠 SAR 应拒绝；
5. 以 Damaged、Event mean/worst/std 和灾害类型稳定性验证。

不能声称：首次共性/特有特征解耦、首次 optical-SAR damage、或论文来源已证明
本项目的 reliability/CCU 机制有效。

## 9. 可执行题目候选

1. **CCU-BDA: Reliability-Conditioned Local Cross-Modal Unification for Optical-SAR Building Damage Assessment**
2. **Match Before Difference: Building-Local Optical-SAR Correspondence for Post-Disaster Damage Severity Mapping**
3. **Preserve the Change: Reliable-Intact Cross-Modal Unification for Optical-SAR Building Damage Assessment**
4. **From Correspondence to Severity: Building-Aware Local Matching for Pre-Event Optical and Post-Event SAR Damage Mapping**

## 10. 当前结论

本轮不应立刻移植全部 HeteCD/DSTCD/IM4CD/层级头。已经实现的 C0-C4 保留为
模块原型，但正式实验入口前移到 E0。只有局部匹配几何在多个灾害事件中稳定
区分 Intact 与损毁后，才值得训练 E2/E3。否则下一步不是调 CCU，而是重做
SAR 迁移与匹配投影。
