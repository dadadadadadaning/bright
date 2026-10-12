# P3–S2选择性结构证据主线：全部已运行实验结果汇总

- 汇总日期：2026-09-05
- 范围：锁定P3、LocCC/S2候选修改、结构蒸馏、Utility Gate、SAR-FSA/RCM，以及Evidence V2无参数证据审计
- 不包含：只有设计或代码但尚无结果的Memory verifier、Selective RGID、可微超像素和多模块联合
- 数据纪律：Val450结果与Train2918内部291张校准集结果分开报告；不同口径的绝对mIoU不得直接比较

## 1. 一句话结论

当前唯一稳定保留的改进是固定LocCC-S2。它在Val450上相对P3提高`+1.0696pp`，在内部291张校准集上提高`+0.7050pp`。后续蒸馏、基础Utility Gate、融合后SAR-FSA/RCM、局部Relation、IM4CD式Matching以及SLIC区域统计均未通过各自预注册门槛。因此当前模型不应继续堆叠模块，部署/论文主结果暂时应保持`P3 + fixed S2`，后续只允许针对明确失效原因开展新的隔离假设。

## 2. 评估协议与基线

### 2.1 Val450旧协议

锁定P3的已记录结果：

| 指标 | P3 |
|---|---:|
| Valid 3-class mIoU | 60.5019–60.5020 |
| mIoU4 | 69.5772 |
| Intact IoU | 75.58 |
| Damaged IoU | 43.03 |
| Destroyed IoU | 62.90 |
| Localization F1 | 90.57 |
| Event mean | 52.73 |
| Event worst | 40.73 |

### 2.2 Train2918内部协议

```text
Train2918
├── gate_train: 2627
└── internal_calibration: 291
```

最近的蒸馏、Utility Gate、SAR适配器和Evidence V2均使用这一内部协议，不访问新的Val450、Test或官方实例。内部291张上的锁定P3为：

```text
mIoU=65.5845
Damaged F1=75.8573
```

旧291张已参与多轮开发判断，因此它现在是development calibration，而不是全新的独立确认集。任何新候选在访问Val450前，应再冻结一个未查看的内部确认划分。

## 3. Val450上的区域上下文实验

| 变体 | Valid mIoU | 相对P3 | 结论 |
|---|---:|---:|---|
| P3 | 60.5020 | 0 | 锁定语义基线 |
| S1 Hard A6 | 61.5399 | +1.0379pp | 有效；建筑区域长距离意见共享有价值 |
| S2 Soft Context | 61.5716 | +1.0696pp | 当前最佳；仅比Hard高0.0317pp |
| Semantic Local Control | 60.5080 | +0.0060pp | 近似无效；普通局部平滑不能解释S2收益 |

Val450上的fixed S2修改审计：

| Changed | Corrected | Harmed | Neutral | Net |
|---:|---:|---:|---:|---:|
| 1,535,249 | 799,584 | 552,463 | 183,202 | +247,121 |

由此得到约：

- correction rate：`52.08%`；
- harm rate：`35.99%`；
- decisive precision：`59.14%`。

解释边界：S2整体有效，但仍产生大量有害修改。主要有效机制是“预测建筑区域内的长距离类别证据共享”，而不是Soft相对Hard的微小差异。

## 4. 内部291张上的fixed S2与蒸馏实验

### 4.1 fixed S2复现

| 模型 | mIoU | 相对P3 | Damaged F1 |
|---|---:|---:|---:|
| K0 frozen P3 | 65.5845 | 0 | 75.8573 |
| K1 fixed S2 | 66.2895 | +0.7050pp | 76.6319 |

修改审计：

| Changed | Corrected | Harmed | Neutral | Net | Harm rate | Decisive precision |
|---:|---:|---:|---:|---:|---:|---:|
| 699,348 | 349,822 | 280,522 | 69,004 | +69,300 | 40.11% | 55.50% |

### 4.2 S2结构蒸馏500步

| 变体 | mIoU | 相对P3 | Damaged F1 | 相对fixed S2 | 判定 |
|---|---:|---:|---:|---:|---|
| K0 frozen P3 | 65.5845 | 0 | 75.8573 | -0.7050pp | 基线 |
| K1 fixed S2 | 66.2895 | +0.7050pp | 76.6319 | 0 | 最佳 |
| K2 semantic local control | 65.4833 | -0.1012pp | 75.7736 | -0.8062pp | 失败 |
| K3 S2 distilled local | 65.6240 | +0.0395pp | 75.8541 | -0.6655pp | 最佳蒸馏但实质无效 |
| K4 S2 distilled GBC | 65.5635 | -0.0210pp | 75.8255 | -0.7260pp | 失败 |
| K5 GBC + variance | 65.5800 | -0.0045pp | 75.8205 | -0.7095pp | 失败 |

正式决定：`STOP_S2_DISTILLATION_AFTER_PILOT`。

原因：最好的K3只提高`0.0395pp`，没有接近fixed S2，Damaged F1没有改善，也未满足最终增益门槛。当前学生仍无法学习到fixed S2的关键结构作用。

## 5. 方向规则与基础Utility Gate

### 5.1 方向白名单

| 方案 | Changed | Corrected | Harmed | Neutral | Net | Decisive precision |
|---|---:|---:|---:|---:|---:|---:|
| Fixed S2 | 699,348 | 349,822 | 280,522 | 69,004 | +69,300 | 55.50% |
| Direction policy | 162,944 | 91,928 | 48,414 | 22,602 | +43,514 | 65.50% |

方向规则相对P3的mIoU增益为`+0.2770pp`，但只在`2/14`事件上不差于fixed S2。它显著降低伤害并提高决定性精度，同时拒绝了大部分有效修正，因此只适合作为低容量安全控制组。

### 5.2 基础Utility Gate 500步

```text
threshold=0.5
mean_on_changed=0.31694
mean_on_S2_only_correct=0.31403
mean_on_P3_only_correct=0.32438
accepted_fraction=0
selector AUROC=0.40899
AP=0.49383
```

Gate最终没有接受任何S2修改，输出退化为P3：

```text
Changed=0, Corrected=0, Harmed=0, Net=0
```

分数取反后的理论AUROC约为`0.59101`，仍低于0.60，因此不能通过事后翻转标签或降低阈值挽救。正式决定是拒绝基础Utility Gate，不延长到3000步。

## 6. SAR特征适配器500步pilot

| 变体 | mIoU变化 | Damaged F1变化 | Bootstrap 95% CI | 事件条件 | 判定 |
|---|---:|---:|---|---|---|
| SAR-FSA | +0.0001168pp | +0.0000806pp | [-0.0000168,+0.0002663] | 事件非负条件通过但预测几乎未变 | 无效/近似恒等 |
| SAR-RCM | -0.23051pp | -0.32010pp | [-0.40272,-0.06882] | 事件稳定性失败 | 稳定负效应 |

RCM的分类别IoU变化：

```text
Intact    -0.09434pp
Damaged   -0.41434pp
Destroyed -0.18285pp
```

正式决定：`STOP_SAR_ADAPTER_AFTER_PILOT`，两者均不进入完整训练，也不组合。

解释边界：这里只否定“冻结P3、融合后256通道SAR特征”这一插入位置。它不能证明FSA/RCM在所有Stage位置上普遍无效；若重启Stage3版本，必须先用冻结特征探针证明频率或轴向统计对目标错误子集有独立信号。

## 7. Evidence V2无参数审计

### 7.1 缓存完整性

Train 2627张：

```text
Changed=6,714,698
Corrected=3,085,050
Harmed=2,962,889
Neutral=666,759
Stored=3,209,949
```

Internal calibration 291张：

```text
Changed=699,348
Corrected=349,822
Harmed=280,522
Neutral=69,004
Stored=699,348
```

两个划分均满足`Corrected + Harmed + Neutral = Changed`，且Val450、Test、官方实例访问标志全部为false。

### 7.2 参数自由结果

| 证据组 | Train选出的最佳通道 | 校准AUROC | AP | Event-macro AUROC | 正方向事件 | 判定 |
|---|---|---:|---:|---:|---:|---|
| Direct | absolute_difference_std | 0.5678 | 0.6054 | 0.5248 | 9/14 | 弱信号，不准入 |
| Relation | relation_mean，反向 | 0.5142 | 0.5679 | 0.5304 | 10/14 | 接近随机 |
| Matching | colocated_cosine_distance | 0.5777 | 0.6098 | 0.5133 | 5/14 | 不稳定；最佳通道不是匹配特有证据 |
| Grid | relation_reconstruction_error | 0.5157 | 0.5752 | 0.5008 | 8/14 | 随机水平 |
| SLIC | region_relation_std | 0.5316 | 0.5939 | 0.5087 | 10/14 | 比Grid高0.0159，但绝对信号不足 |

准入结果：

```text
Admitted=[]
Decision=STOP_EVIDENCE_VERIFIER_TRAINING
```

自动控制器随后正确输出：

```text
Ranked admitted=[]
Selected=[]
Decision=STOP_NO_EVIDENCE_ADMITTED
```

因此没有训练任何Relation/Matching/SLIC Verifier，也没有进入联合阶段。

## 8. 当前总体排名

### 8.1 内部291张同协议排名

1. `P3 + fixed S2`：`66.2895`，当前最佳；
2. Direction policy：相对P3`+0.2770pp`，安全但覆盖不足；
3. K3 distilled local：`65.6240`，仅`+0.0395pp`；
4. P3 / Utility Gate / FSA：基本等于`65.5845`；
5. K5、K4、K2：略低于P3；
6. RCM：约`-0.2305pp`，且置信区间完全为负。

Evidence V2是诊断分数，不产生新的分割结果，不能放入mIoU排名。

### 8.2 假设状态矩阵

| 假设 | 当前状态 | 是否可继续 |
|---|---|---|
| 建筑区域内长距离语义共享 | 支持 | 保留fixed S2 |
| Soft显著优于Hard | 不支持 | 两者仅差0.0317pp，不能作为主创新 |
| 普通局部平滑可解释S2 | 否定 | 停止 |
| 当前LocCC条件蒸馏可替代S2 | 否定 | 停止当前学生 |
| 35维Utility可识别有害修改 | 否定 | 停止基础Gate |
| 修改方向具有一定安全规律 | 部分支持 | 保留为诊断/控制，不作最终模型 |
| 融合后SAR-FSA有效 | 否定当前接口 | 不延长训练 |
| 融合后SAR-RCM有效 | 否定且有害 | 停止 |
| 融合特征8邻域Relation可验证S2 | 否定 | 停止当前实现 |
| IM4CD局部匹配可验证S2 | 否定当前实现 | 停止；匹配特有通道无优势 |
| SLIC区域统计优于Grid | 有很弱相对迹象 | 不足以训练；只允许重新定义建筑内图假设 |

## 9. 尚未完成，不能写成实验结果

以下模块只有设计、旧旁支诊断或代码计划，没有在当前锁定协议下形成通过结果：

- S4跨事件Memory verifier；
- Selective RGID；
- 真正的建筑ROI内固定SLIC图关系差；
- 可微软超像素；
- Stage3 FSA频率探针及Stage3 FSA；
- 两个证据的S8联合Gate；
- 新的未查看内部确认集；
- 新候选的Val450和Test评估。

因此论文或汇报中不能写“Memory/RGID/可微超像素已经失败或已经有效”。当前只能说它们没有在这条严格父结果链上获得运行资格。

## 10. 失败模式归纳

### 10.1 候选生成有效，候选选择困难

fixed S2在两个评估协议上均有正增益，说明候选修改总体有价值；但其corrected与harmed在当前35维Utility、直接差、局部关系差和匹配统计中高度重叠。核心瓶颈不是“没有候选”，而是“现有证据不能稳定区分候选效用”。

### 10.2 当前跨模态统计主要学习到事件/模态差，而非修改效用

Direct最高仅0.5678，Matching的最佳通道退回同位余弦距离，Relation接近随机。这说明融合后RGB/SAR差异可以反映某种变化强度，但不能跨事件稳定判断某次S2改动是否比P3更正确。

### 10.3 分区略有作用，但读出定义太弱

SLIC比等预算Grid高0.0159 AUROC并有10/14事件方向正确，但绝对AUROC和事件宏均接近随机。当前实验只验证“SLIC聚合局部relation map”，并没有实现原始SP-E0所要求的“建筑ROI与SLIC相交、共享节点、模态内图边关系差、建筑级读出”。因此它否定当前区域统计，不完全否定建筑内图关系假设。

### 10.4 增加训练步数不能修复无信息输入

Utility Gate AUROC低于0.5，Evidence组均低于0.60，FSA几乎不改变预测，RCM稳定负向。这些结果不支持直接从500步延长到3000步，也不支持通过降低门槛制造候选。

## 11. 下一步优先级

### P0：冻结当前结果

保留`P3 + fixed S2`作为当前最佳主结果；保存所有JSON、checkpoint SHA256、split和日志。不要访问Val450重新选模块。

### P1：只用现有缓存做条件化诊断

无需重跑P3，按以下预定义因素重新报告所有通道AUROC：

1. P3→S2修改方向，重点`Intact→Damaged`；
2. 建筑面积和有效节点数；
3. P3 entropy/margin；
4. 14个事件；
5. 建筑核心与边界；
6. IM4CD真正的offset、residual、confidence通道，而非同位控制通道。

只有某个预定义子集同时满足AUROC≥0.60、event macro≥0.58、至少10个事件方向正确且样本量充分，才允许训练方向限定的单证据Verifier。

### P2：最后一次无参数SP-E0建筑图审计

若继续SLIC路线，应实现真正的建筑内图，而不是训练当前SLIC统计：

```text
detached P3 building ROI
  ∩ fixed pre-RGB SLIC
  → shared RGB/SAR nodes
  → intra-modal edge relations
  → relation difference
  → building-level score
```

它必须继续以Grid、直接节点差和空间打乱为控制。无参数审计失败就终止图路线，不训练GNN。

### P3：其他模块保持暂停

Memory、Selective RGID、可微超像素和多模块联合不应自动启动。Stage3 FSA只有在冻结频率探针通过后才重新获得训练资格。RCM当前接口永久停止。

## 12. 最终研究判断

现有结果支持的论文事实是：

> 光学定位约束下的建筑区域长距离语义共享能够稳定改善RGB–SAR建筑损毁分割，但固定区域传播同时产生大量有害修改；常见的像素效用特征、融合后频率/轴向适配以及简单跨模态关系统计尚不能跨灾害事件可靠识别这些有害修改。

当前创新问题应该收敛为：

> 如何在不改变P3定位和不直接对齐异构特征值的前提下，为fixed S2候选修改构造具有建筑实例语义、跨事件稳定且可校准的可靠性证据？

在回答这个问题之前，不应把Memory、RGID、超像素、FSA和关系模块堆入同一个最终网络。

## 13. 结果来源

- Val450 P3/S1/S2/Local Control：`notes/insights/20260904-next-innovation-correction-roadmap.md`及用户保存的`SPI_SELECTIVE_CONTEXT_VAL450`审计结果；
- Distillation：`SPI_S2_DISTILLATION_internal_calibration_step500_seed3407.json`；
- Utility Gate：`SPI_S2_UTILITY_GATE_V1/internal_calibration_step500_seed3407.json`；
- SAR adapters：`SPI_SAR_ADAPTER_V1/fsa_internal_step500_seed3407.json`与`rcm_internal_step500_seed3407.json`；
- Evidence V2：`SPI_S2_EVIDENCE_V2/parameter_free_admission_seed3407.json`；
- 本文只汇总用户已提供的日志与本地已保存记录，没有把仅有设计的实验推断为结果。
