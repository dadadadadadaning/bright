# RGB-SAR 建筑损毁模型演进、模块插入位置与论文路线审计

日期：2026-08-25  
项目：BRIGHT RGB-SAR Building Damage Assessment  
依据：本项目 P3/H6/H11/H12–H14/Q0–Q2/OA-HMD/OA-HSDNet/P3-SRAC 实验、代表性原论文和官方代码。

## 1. 执行结论

1. **代码主体并非普遍写错。** 多数安全守卫、冻结契约、零初始化、严格验证和测试隔离都工作正常；连续负结果主要来自“新增机制没有提供可泛化的 Damaged 证据”，而非训练程序没有运行。
2. **P3 的强点是任务分解，不只是 Transformer。** 它用 RGB 单独完成建筑定位，再在建筑先验约束下融合 RGB 与历史 SAR 表示进行四类像素损毁判断。稳定性来自“光学定位锚点 + 晚期融合 + 已训练分类头”。
3. **原生 SAR 中确实有信息。** Stage Probe 从 Stage1 到 Stage4 的 Affected/Severity 可分性持续提高，Stage4 的 Affected AUROC 约 84.44%、Severity AUROC 约 88.04%；Q0 去掉背景竞争后，Damaged Recall 从旧 sar_aux 的约 10.61% 提高到 35.41%。这证明问题不是 SAR 完全无效。
4. **信息没有稳定转化成最终增益。** 解冻 SAR、深层残差、像素 Router、关系矩阵、建筑统计残差、Cross-Attention、简单超像素池化都曾打开并影响预测，但大多增加 Damaged→Intact、降低 Event worst，说明“影响模型”不等于“提供跨事件可靠证据”。
5. **当前超像素分支已学到建筑几何，但尚未学到可用的损毁分区。** SR1PA 将 Building IoU 提升到约 66.56%、Boundary F1 提升到约 57.89%，但归一化熵仍约 0.908、每像素有效候选约 7.38、背景泄漏约 34.37%。它还是边界感知软平均，不是清晰区域决策单位。
6. **直接复刻“超像素 + 检索”不再具有论文新颖性。** IGARSS 2026 已录用 SRAC，且公开说明与 BRIGHT、VMambaSSN、标注库检索和稀有损毁类别高度重合。你的论文必须把创新点转向“原生 SAR 区域证据如何跨事件稳定检索和有序决策”，而不是再声称首次把超像素或 memory bank 用于 BRIGHT。

## 2. P3 最底层结构：为什么它一直是最稳基线

P3 的实际前向链不是“RGB 和 SAR 直接相减”，而是：

```text
灾前 RGB ─→ RGB 编码/多尺度融合 ─→ rgb_feature(256C)
灾后 P3 历史 SAR 表示 ─→ SAR 编码/多尺度融合 ─→ sar_feature(256C)

rgb_feature ─→ clf_1 ─→ P(building)
                           ↓
              roi = α + (1-α)·P(building)

[rgb_feature, sar_feature]
        ↓ fusion_layer_3
     fused_feature × roi
        ↓ clf_2
Background / Intact / Damaged / Destroyed
```

代码证据位于 `DamageFormer_S8_MICB_DSMIL.py`：RGB 定位由 `clf_1(rgb_feature)` 产生；`fusion_layer_3` 只在高层融合两模态；`roi_weight` 用 RGB 建筑概率约束损毁特征；`clf_2` 输出四类。默认 `mil_residual_scale=0`，因此 DSMIL 辅助头并不自动改写推理结果。

P3 强的根因有四个：

- **几何任务交给 RGB。** 灾前光学中的建筑轮廓清楚，不让 SAR 的叠掩、阴影和斑点参与定位。
- **模态先独立编码。** 避免第一层卷积把反射光纹理和微波后向散射混在一起。
- **先定位、再判损毁。** 背景竞争被建筑先验削弱。
- **最终头已联合适配。** 它不是一个新插入的小头，而是从训练中学会了 P3 特征的统计分布。

这与 BRIGHT 2026 两个优胜方案的共同结论一致：模态独立编码、晚期或分阶段融合、光学主导定位、损毁后置。挑战报告中，第一名早期融合测试 mAP 0.0542、双阶段 0.1095；第二名早期融合 0.120、晚期双骨干 0.264，说明“放在哪一层融合”比“换一个更大骨干”更关键。[BRIGHT Challenge 2026](https://arxiv.org/abs/2607.22746)

## 3. 从开始到现在：每条模型线真正验证了什么

| 实验线 | 插入位置 | 唯一核心假设 | 结果性质 | 已得到的结论 |
|---|---|---|---|---|
| P3 | RGB 定位 + 高层晚期融合 + 四类头 | 光学定位锚点能稳定承载 SAR 损毁证据 | 当前最稳主基线 | 保留 |
| H6-A2 | 原生 SAR 独立编码器，Stage3/4 参与融合 | 原生 SAR 比 P3 历史表示更有用 | 与 P3 有互补但总体略弱 | 原生 SAR 有价值，但融合/头未充分利用 |
| H11 T1/T1E | 冻结 SARMAE 深层残差 | 外部 SAR 预训练能作为高层补丁迁移 | 验证/测试无稳定增益 | 否定该迁移方式，不否定 SARMAE 本身 |
| H12 DGB | Stage2/3 → Affected gate 旁路 | 浅层弱损毁能修复 Damaged gate-miss | 接近零增益或恶化 | 冻结浅层差异旁路不是主解 |
| H13 SPER | 网格/RGB 超像素区域平滑 | 区域一致性可提升弱损毁 | Damaged 极小增益，Recall/Worst 下降 | 简单平滑会抹掉弱证据 |
| H14 unique evidence | common/unique 能量 | 模态独有成分携带 Damaged 证据 | AUROC 接近随机 | 该统计量不可用 |
| SAR Stage Probe | 各 Stage 冻结线性/原型读出 | 哪层 SAR 最可分 | Stage4 最强 | 深层有可读语义，但不等于最终互补 |
| F0/F1/F2 | 改 SAR 解冻深度 | 冻结策略限制 SAR | 只有极小变化，Worst 下降 | 冻结不是主要瓶颈 |
| Q0 | Stage4 + 建筑条件三类读出 | 背景竞争压垮 sar_aux | Damaged Recall 明显恢复 | 建筑条件化是必要的 |
| Q1-R0 | 建筑内 mean/max/top-k | SAR 是局部稀疏强响应 | 最佳仅约 +0.22 pp DmgRec | 不是聚合函数太简单的问题 |
| Q2-B1 | 建筑区域统计 → P3 残差 | 区域单位能学会何时纠正 P3 | Damaged、Event worst 明显下降 | 统计残差仍学不会可靠选择 |
| BIOS E0 | RGB/SAR 内部关系矩阵差 | 结构关系差能发现 P3 漏检 | miss-AUC Oracle≈0.51/P3≈0.48 | 当前关系信号接近随机 |
| External X1/X2 | 完整双流/HeteCD 整体迁移 | 外部异构 CD 主模型可直接替代 | X1≈54.5、X2≈39.8 Valid，低于 P3 | 二值 CD 架构/头不直接适配三级 BDA |
| OA-HSDNet N0/N1 | 完整原生双流层级主模型 | 非共享编码 + 任务分尺度 + Cross-Attention | N1 优于 N0，但仍低于 P3且 gate-miss 增加 | Attention 能读强损毁，不能自动救弱 Damaged |
| SR1/SR1F | 超像素区域主分类头 | 区域决策可替代 P3 像素头 | 显著失败 | 从零替换成熟头代价过大 |
| SR1P | 超像素池化/还原 → 原 P3 clf_2 | 安全结构投影可保留 P3 并增益 | 投影打开但约零增益，Worst略降 | 软分区只提供平滑 |
| SR1PA | P3 输出冻结，仅预训练边界分区 | 先学好几何再开放分类 | 几何提升、分配仍高熵 | 需要迭代种子或亲和度学习 |
| SR1PAI/SR1PIB | 三轮分配-更新 + 两阶段训练 | 高熵来自一次分配，迭代可形成明确区域 | 当前待完成严格验证 | 这是当前唯一应继续的超像素受控实验 |

这些结果组成一条一致的信息漏斗：

```text
SAR 有信息
→ 建筑条件下能读出来
→ 但像素/统计/关系残差不能稳定决定何时使用
→ 简单区域平滑也不能解决
→ 下一步必须改变“区域表示与监督”，而非继续增大残差权限
```

## 4. 你当前超像素模块到底在做什么

### 4.1 分区输入

当前边界感知版本把三类光学几何信号拼成 258 通道：

```text
冻结 P3 RGB feature：256C
冻结 P3 building probability：1C
灾前 RGB Sobel edge：1C
                 ↓ concat
               258C
```

这一步的设计是对的：超像素负责组织建筑结构，因此应主要看灾前 RGB 几何，而不是让灾后 SAR 的散射边缘决定建筑边界。

### 4.2 嵌入网络

```text
Conv 1×1：258→32
GroupNorm(4) + GELU
Depthwise Conv 3×3：32→32
GroupNorm(4) + GELU
Conv 1×1：32→8
L2 normalize
```

- 1×1 卷积完成跨通道压缩；
- Depthwise 3×3 只建模局部空间邻域，参数少且不重新混合通道；
- 8 维单位向量作为像素外观/结构嵌入。

### 4.3 局部九候选分配

特征网格先按固定 cell 初始化种子。每个像素只考虑所在网格及周围 3×3 共九个种子：

\[
q_{p,k}=\operatorname{Softmax}_k\left(-\frac{\|e_p-c_k\|^2}{T}-\lambda d_{p,k}^{spatial}\right).
\]

它同时考虑：

- 嵌入相似度：像素和种子是否属于相似结构；
- 空间紧致度：避免一个超像素跨越过远位置；
- 温度：控制分配软硬。

### 4.4 SR1PAI 的三轮迭代

新版本已经从一次分配改成：

```text
规则种子 C0
→ q0 分配
→ 按 q0 加权重算 C1
→ q1 分配
→ 重算 C2
→ q2 最终分配
```

种子更新为：

\[
c_k^{t+1}=\frac{\sum_p q_{p,k}^{t}e_p}{\sum_p q_{p,k}^{t}}.
\]

这与 SSN 的核心思想一致：超像素不是固定网格，也不是一次 Softmax，而是可微的局部软聚类。[SSN, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.html)

### 4.5 为什么必须先分区、后分类

SR1PAI 阶段把 P3 和结构投影全部冻结，只用建筑重建、边界与分区监督训练 q。只有分区达到门槛，SR1PIB 才开放：

```text
P3 fused feature
→ 按超像素池化 node feature
→ 按 q 还原 projected feature
→ F' = F + α(Fsp-F)
→ 原 P3 clf_2
```

并严格保留 P3 的 Background/Building 概率，只修改条件 Intact/Damaged/Destroyed。这解决了早期 SR1/SR1F 同时破坏定位和分类的问题。

### 4.6 仍存在的核心缺口

当前 `fused feature` 来自 P3 自己的 RGB/历史 SAR 路径。**SR1PAI/SR1PIB 主要验证“结构重组是否帮助 P3”，还不是完整的原生 RGB-SAR 超像素检索模型。** 若要形成真正论文主模型，后续区域节点必须显式包含：

```text
RGB geometry token
+ native-SAR Stage3/4 severity token
+ region shape/boundary statistics
```

而不能只把 P3 已有特征做平均后重新送回原头。

## 5. 别人改模型，一般加在哪一层

| 改动层级 | 典型做法 | 代表工作 | 解决什么 | 对你是否适合 |
|---|---|---|---|---|
| 输入头 | 1C SAR 专属 stem、独立归一化、保留输入契约 | SARMAE、BRIGHT 双流 | 防止 RGB 滤波器误读 SAR | 必须做，但你已验证仅改输入头不够 |
| 编码器内部 | 非共享权重、模态 adapter、MoE、SAR 预训练 | HeteCD、M²CD、SARMAE | 学习各自统计分布 | 非共享值得保留；MoE/大预训练后置 |
| 多尺度特征层 | 每层差异、对齐、状态空间解码 | HeteCD、FlowMamba | 建模尺度与空间关系 | 你的 flow/关系审计不支持直接搬整套 |
| 融合层 | FPN 晚期融合、光学 passthrough、小残差交互 | BRIGHT 2026 亚军 | 安全引入 SAR，不破坏定位 | 是 P3 已验证的正确位置 |
| 区域/实例层 | proposal query、ROIAlign、超像素 token | BRIGHT 2026 亚军、SSN、SRAC | 同一建筑统一决策、聚合稀疏证据 | 是你尚未完成的关键层 |
| 分类头 | 定位/损毁解耦、Occurrence/Severity 分解、有序损失 | BRIGHT winners、FlowMamba | 处理 Damaged 的两条边界 | 必须与区域 token 联合，而非像素后处理 |
| 训练/记忆层 | 类平衡 memory、伪标签、场景校准、event-holdout | SRAC、DFC 2025 冠军、BRIGHT 2026 冠军 | 稀有类与跨事件漂移 | 论文差异化最有潜力的位置 |

代表性方法的“下刀位置”如下：

- **HeteCD** 在编码后特征层加入 FCA 和 3D 差异挖掘；它解决二值异构变化，不负责建筑定位与三级严重度。你的整体迁移 X2 很差，说明其任务头和假设不可直接搬。[HeteCD 官方代码](https://github.com/weiAI1996/HeteCD)
- **M²CD** 把 MoE 放进每个骨干阶段，并把 Optical-to-SAR 自蒸馏放在训练路径；它改变的是编码容量和训练监督，而不是最终输出后处理。[M²CD](https://arxiv.org/abs/2503.19406)
- **FlowMamba** 把对齐和 Mamba 放在多尺度特征/解码层，再加类别距离损失；但你的流近零和关系审计已经否定了显式配准作为当前主因，因此只可借鉴分尺度和有序监督。[FlowMamba 官方代码](https://github.com/flying318/FlowMamba)
- **DFC 2025 冠军** 把建筑提取放在前一级，再用建筑引导的低不确定性伪标签训练损毁模型；它主要改变任务流程与监督来源。[论文与代码](https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping)
- **BRIGHT 2026 亚军** 使用双 ResNet50-FPN，在每个 FPN 层用 1×1 卷积晚期融合，初始化为 optical passthrough；随后在 proposal query 层做跨模态推理。即“编码器、融合层、实例层”三处改动，而不是一个末端模块。[Challenge 报告](https://arxiv.org/abs/2607.22746)
- **SSN/FCN superpixel** 把超像素放在密集特征与下游任务之间，作为可微降采样/上采样或 token 化层；不是把传统 SLIC 当作后处理。[FCN Superpixel, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Superpixel_Segmentation_With_Fully_Convolutional_Networks_CVPR_2020_paper.html)
- **BRDG** 将“区域生成”和“边界像素精修”拆成两条路径，避免区域平均抹掉边界；并用邻接区域对比损失增强相邻区域区分。[BRDG, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Ahmed_Boundary-Responsive_Differentiable_Gating_for_Superpixel-Based_Segmentation_CVPR_2026_paper.html)

## 6. 为什么 SRAC 公告中的提升可能很大

IGARSS 官方列表确认论文题目和录用；机构介绍报告 75.65% mIoU、相对基线 +8.08 pp，并说明使用 VMambaSSN 和标注库检索。但截至本次检索，公开全文与官方代码未找到，因此只能作机制推断，不能把内部细节当成已核实事实。[官方录用列表](https://2026.ieeeigarss.org/papers/accepted_papers.php) · [机构介绍](https://jianghai.ucas.ac.cn/index.php/zh/xshde/xsdte/461-igarss-2026)

若该提升可信，最可能不是“超像素平滑”单独贡献，而是三项共同作用：

1. **决策单位改变。** 像素预测变成结构一致的区域分类，减少建筑内部标签碎片。
2. **监督单位改变。** memory bank 让稀有 Damaged/Destroyed 节点直接看到类平衡的标注邻居，而不是被海量 Intact 像素梯度淹没。
3. **分类机制改变。** 检索提供外部同类证据，分类器不必只靠当前样本的抽象 SAR 特征推断非直观损毁。

这也解释你的 H13/SR1P 为什么不能复现其收益：你做过区域聚合和结构投影，但尚未真正完成“高质量区域 + 类平衡检索 + 区域主分类”这一整条链。

## 7. Idea Evaluator：当前想法是否值得作为论文核心

### 7.1 当前想法的一句话判定

> **“P3 + 可学习超像素 + 普通类平衡检索”作为论文核心应 Reject and Pivot。**

### 7.2 致命问题

- **CRITICAL — F1 新颖性被支配。** SRAC 已在同一 BRIGHT 任务公开宣称 VMambaSSN + 标注库检索。若你的主张仍是泛化的“超像素检索增强 BDA”，在问题、模块和数据集三个维度高度重合。
- **CRITICAL — F2 机制链在当前版本被数据反驳。** H13、SR1F、SR1P 已分别否定简单区域聚合、替换区域头和软结构投影；SR1PA 又显示几何可学但分配高熵。不能把被反驳的“区域平滑会提升 Damaged”继续作为核心机制。

### 7.3 允许继续的 Pivot

把主张改成：

> **跨事件约束的光学锚定原生-SAR区域检索：用灾前光学学习建筑结构区域，用独立原生 SAR 编码形成严重度 token，并在训练时禁止同事件近邻捷径，使检索只学习跨灾害可迁移的有序损毁证据。**

这与 SRAC 的公开描述有三个可检验差异：

- 明确分离 RGB 几何 token 与 native-SAR severity token；
- 训练时采用 leave-one-event-memory-out，禁止从当前事件检索；
- 分类头显式分解 Intact↔Affected 与 Damaged↔Destroyed，并报告校准和 event-worst。

若没有这三点，建议把超像素线只作为消融，不作为论文创新。

## 8. 修订后的模型建议：OASR-ER

暂定名：**Optical-Anchored Structural Retrieval with Event-Robust Ordinal Reasoning**。

### 模块 A：光学锚定迭代结构分区

```text
P3 RGB feature + P(building) + RGB edge
→ 3-iteration local SSN
→ building-consistent assignment q
```

只负责“区域在哪里”，不看 SAR 决定边界。SR1PAI 是该模块的受控验证。

### 模块 B：模态因子化区域表示

```text
RGB Stage3/4 → ROI/superpixel pool → q_geo
Native SAR Stage3/4 → ROI/superpixel pool → q_sar
shape/boundary/raw-SAR statistics → q_meta

[q_geo, q_sar, |P_rgb-P_sar|, q_meta]
→ region token
```

关键不是再做逐像素 `|R-S|`，而是让同一建筑结构区域内的两模态证据分别保留，再在区域层组合。

### 模块 C：跨事件类平衡检索

memory 按 Intact/Damaged/Destroyed 等容量存储训练区域 token。训练一个事件时，memory 排除同事件样本：

```text
query region
→ 每类 top-k 跨事件邻居
→ similarity + label + geometry metadata
→ cross-attention/weighted evidence
```

这样 memory 不能靠传感器、地理或灾害纹理找到“同事件近邻”，必须寻找跨事件重复出现的损毁结构。验证和测试只读 memory，不写入。

### 模块 D：层级有序区域头 + 边界精修

```text
P3 RGB：Background / Building
region token + retrieved evidence：Intact / Affected
affected region token：Damaged / Destroyed
region prediction + pixel boundary path → final mask
```

区域内部用统一严重度，边界附近保留高分辨率像素精修路径，避免超像素把弱边界和小损毁全部平均掉。

## 9. 严格实验链

| 实验 | 唯一变化 | 必须回答的问题 | 通过才进入 |
|---|---|---|---|
| E0 | P3 exact guard | 评估协议能否精确复现 | E1 |
| E1 | SR1PAI 三轮分区，只训 partition | 能否形成清晰建筑结构区域 | E2 |
| E2 | 固定分区，加入 native-SAR Stage3/4 region token；无 retrieval | 原生 SAR 区域表示是否比 P3 region token 增加互补性 | E3 |
| E3 | 加普通类平衡 train-only memory | 检索本身是否提升稀有 Damaged | E4 |
| E4 | 训练时排除同事件 memory 邻居 | 提升是否来自跨事件损毁证据而非事件捷径 | E5 |
| E5 | 层级有序头 + 边界精修 | 是否同时改善 gate-miss、severity 与边界 | 多种子/测试 |

E1 继续沿用分区门槛。E2 后的模型门槛应至少包括：

```text
Valid mIoU       +0.30 pp
Damaged IoU      +0.50 pp
Affected Recall  +0.50 pp
Damaged gate-miss -0.50 pp
Event mean/worst 不下降
非负事件         ≥10/14
```

如果 E1 分区仍高熵，不应进入 memory；如果 E2 native-SAR region token 没有净互补，不应进入检索；如果 E3 有效但 E4 失败，则增益主要依赖同事件近邻，不能声称跨事件稳健。

## 10. Tech Paper Template：修订论文骨架

### 10.1 论文定位

**类型：Technique，带明显跨领域迁移。** 迁移可微超像素和检索增强分类到 RGB-SAR BDA，但新增任务特定机制是模态因子化区域 token、跨事件排除式 memory 和有序层级决策。

### 10.2 思考表

| 字段 | 内容 |
|---|---|
| Research Background | 灾前光学提供建筑几何，灾后 SAR 提供全天候结构散射；BRIGHT 要求跨事件三级建筑损毁判断。 |
| Limitations of Prior Studies | 像素预测碎片化；早期融合混淆模态；普通 memory 可能检索同事件/同传感器捷径；Damaged 边界跨事件漂移。 |
| Your Key Idea or Goal | 将“区域在哪里”“SAR 表达什么”“何时相信检索证据”拆成三个受约束模块。 |
| Challenge 1 | 学出边界清晰且不过度平滑的建筑结构区域。 |
| Challenge 2 | 在不对齐像素语义的前提下组织 RGB 几何与 SAR 严重度证据。 |
| Challenge 3 | 防止检索记住事件/传感器风格而非损毁机理。 |
| Method Module 1 | Optical-Anchored Iterative Structural Partition。 |
| Method Module 2 | Factorized Native-SAR Region Token and Class-Balanced Retrieval。 |
| Method Module 3 | Event-Exclusive Memory Training and Hierarchical Ordinal Head。 |
| Contributions | 新任务级区域检索范式；跨事件排除式 memory；完整的结构质量—互补性—跨事件校准证据链。 |

### 10.3 Introduction 六段逻辑

1. 全气象建筑损毁的现实需求与 BRIGHT 的灾前 RGB/灾后 SAR 设置。
2. 现有像素早期融合、普通晚期融合和区域平滑在弱 Damaged 与跨事件上的局限。
3. 问题本质：几何、模态证据和事件校准三种不确定性被混在一个像素分类头中。
4. 三个挑战：结构区域、异构区域表示、跨事件检索捷径。
5. 方法概览：光学结构分区、原生 SAR 区域 token、事件排除 memory、有序头和边界精修。
6. 贡献：新机制、严格验证链、跨事件/类别/边界诊断。

### 10.4 主文章节

1. Introduction
2. Related Work
   - All-weather Building Damage Assessment
   - Heterogeneous Optical-SAR Change Detection
   - Differentiable Superpixels and Region Tokens
   - Retrieval-Augmented Recognition and Domain Generalization
3. Method
   - Optical-Anchored Localization Backbone
   - Iterative Structural Partition
   - Factorized RGB/SAR Region Encoding
   - Event-Exclusive Class-Balanced Retrieval
   - Hierarchical Ordinal Damage Decoder
4. Experimental Protocol
   - Official BRIGHT Split
   - Event-Held-Out Protocol
   - Metrics: mIoU, Damaged IoU, gate-miss, calibration, event-worst, boundary F1
5. Results
   - Main Comparison
   - Structure Quality Audit
   - Retrieval Complementarity
   - Cross-Event Generalization
6. Ablations and Failure Analysis
7. Conclusion

### 10.5 一致性检查

- 背景中的核心缺陷必须对应“像素碎片 + 模态混淆 + 事件捷径”，不能泛写“特征提取不足”。
- 三个挑战必须一一对应三个模块；不要再加入 Flow、FCA、MoE、Mamba 形成第四条无关主线。
- 每个贡献必须有独立实验：分区质量、native-SAR 区域互补、event-exclusive retrieval。
- 如果 E1/E2 不通过，不能提前写“检索提高跨事件泛化”的贡献。

## 11. 最终行动建议

1. **先完成 SR1PAI 的三轮迭代分区验证。** 这是当前唯一未被结果否定、且成本可控的结构实验。
2. **不要直接运行旧 SR2。** 旧 SR2 的 memory 基于尚未通过的区域和 P3 内部特征，无法回答原生 SAR 区域检索问题。
3. **E1 通过后，新建 E2：固定分区 + native-SAR Stage3/4 region token。** 不加检索、不加 ordinal，先看互补矩阵。
4. **E2 通过后才实现 event-exclusive memory。** 这既是最可能的指标增益点，也是相对公开 SRAC 描述真正能形成差异化的模型创新。
5. **P3 始终保留为正式主基线。** 新模型必须在 Event worst、Damaged gate-miss 和 10/14 事件上同时通过，才允许访问测试集。

