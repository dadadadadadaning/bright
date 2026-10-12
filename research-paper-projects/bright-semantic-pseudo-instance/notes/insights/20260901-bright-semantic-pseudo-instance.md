# 从语义损毁标注到伪建筑实例：研究评估、代码迁移与实验协议

日期：2026-09-01  
结论等级：**Accept with revisions（有条件继续）**

## 一、先把研究问题说成一句可以被实验否定的话

核心假设不是“连通域能不能把建筑切开”，也不是“实例分割是否比语义分割先进”，而是：

> 在不使用官方实例掩膜或实例 ID 训练主方法的条件下，从语义损毁标签构造带置信度的伪建筑结构，并通过实例均衡、中心/偏移/边界辅助监督约束共享 RGB–SAR 表征，能否同时提高未经实例后处理的原始语义损毁预测，以及建筑实例几何质量？

这个表述有三个重要约束：

1. **原始语义头是主指标。** 如果只在输出后把语义区域切碎，像素类别本身没有改变，就不能声称语义分割提高。
2. **真实实例不是禁用数据。** 它不进入主分支训练，但必须用于评价、全监督上界和标注预算对照。
3. **实例收益与语义收益分开报告。** RGID 可以先证明“剪刀有效”，但只有共享特征经过实例监督后原始语义结果提高，才能证明“伪实例反哺语义”。

## 二、现有实验到底已经证明了什么

### 2.1 已证明：建筑实例几何有可利用价值

RGID 的结果是强证据：

| 结果 | F1 | 解释 |
|---|---:|---|
| G0 | 0.566187 | 没有充分实例拆分的起点 |
| Learned G3 | 0.599920 | 学习到的 Center/Boundary/Watershed 已有效 |
| Oracle G3 | 0.656250 | 正确中心仍有明显条件潜力 |
| Oracle G2 | 0.663134 | Center 准确后，现有 Rollback 反而过于保守 |

Learned G3 相对 G0 获得 0.033733，而 Oracle G3 的条件增益是 0.090063，因此 learned center 只实现了约 37.5% 的可见条件收益。Oracle 同时增加 TP、降低 FP/FN、减少 Merge/Split，说明问题不是单纯“中心数量不够”，还包括漏检、重复、偏移和低分辨率消失。

### 2.2 已证明：如果建筑级归属理想，语义结果有明显上界

H7 的 P3/A2 building oracle 将验证 mIoU 从 0.605019 提高到 0.663547，绝对提高 5.8528 个百分点；Damaged IoU 从 0.430309 提高到 0.503052，提高 7.2742 个百分点。

这说明“建筑级一致性”值得研究，但它仍是不可部署的 oracle。它没有证明伪实例训练一定能取得同样提升，也没有证明增益来自实例拆分而不是 oracle 标签传播。

### 2.3 已否定：当前 Memory 与局部 D-I 热区不是安全修正器

- CMR-V1 Top-20 能覆盖 486/522 个 D→I 漏检，说明候选检索覆盖率高。
- 但 FRN、Chamfer、DN4、EMD 的 combined AUC 都只在约 0.787–0.792，且没有锁定的正净收益操作点。
- CMR-C0 的 D-I Top20+NMS 未提高 set-only AUC，反而把 Intact Harm Exposure 从 40.45% 提高到 46.49%。
- 可视化显示热点多为普通 RGB 屋顶纹理、SAR 强散射和几何边缘；NMS 只是让错误位置更分散。
- EC2 raw 的跨地区专家 AUC 为 0.7969，但事件去中心化降至约 0.733。

因此 Memory 应暂停。它不能被包装成当前论文贡献，也不应与伪实例主实验同时推进。它留下的唯一可复用经验是：**任何语义修正必须有锁定阈值、Harm Exposure 和事件宏平均审计。**

## 三、现有代码与文献的准确对应

### 3.1 当前代码已经有的模块

1. `DamageFormer_S8_MICB_DSMIL.py` 中的 `build_training_bag_specs()`：按 Intact、Damaged、Destroyed 分别做 4 邻域连通组件。
2. `model/instance_geometry/targets.py`：唯一 EDT 中心、Gaussian center、boundary、可选 pixel-to-center offset；边界截断实例可忽略。
3. `model/instance_geometry/losses.py`：CenterNet focal、boundary BCE/Dice、按实例等权的 Smooth-L1 offset。
4. `model/instance_geometry/decode.py`：center+watershed 与 center+offset 两类解码器，以及 gate/rollback。
5. `train_S9_instance_geometry.py`：在严格冻结 S8/P3 的前提下训练几何头。

这些模块与 Panoptic-DeepLab、IRNet、Deep Watershed、TernausNetV2 和 BESTIE 的技术路线是一致的，所以无需另起炉灶重写 center/offset/watershed。

### 3.2 当前代码最关键的限制

当前 `build_training_bag_specs()` 只把不同损毁类别分别做连通域，能够避免 Intact 和 Damaged 被合成同一个伪区域，但**无法拆开同一损毁类别下互相接触的两栋建筑**。因此当前产物应称为“损毁类别连通伪区域”或 pseudo blob，而不是可靠的 pseudo instance。

更重要的是，`train_S9_instance_geometry.py` 明确冻结 S8：backbone 参数 `requires_grad=False`，训练时只更新几何头。这样可以检验实例几何是否可学，却不可能让几何损失反向改善 P3 原始语义头。

所以目前的实验链条是：

    语义模型固定 → 学一个实例剪刀 → 实例 F1 改善

而论文核心假设要求的是：

    语义标签 → 伪结构 → 几何辅助损失 → 更新共享表征 → 原始语义头改善

这是下一阶段必须补上的因果环节。

## 四、应该借鉴什么，不应该照搬什么

### 4.1 BESTIE：借鉴伪实例的不完整性处理

最值得迁移的不是整套网络，而是两个原则：

- 伪实例缺失时，不把未覆盖区域强行当负样本；使用 ignore/guidance mask。
- 伪实例需要在线自精炼，不能让第一轮错误连通域永久成为真值。

本地落地方式：先按伪区域质量分为 reliable、ambiguous、ignored。只有内部紧致、中心稳定、不过度接触图像边界的区域用于中心/偏移强监督；疑似同类粘连区域只用于弱边界或一致性，不生成单一“真中心”。

### 4.2 Panoptic-DeepLab：保留 center+offset，减少对复杂后处理依赖

本地已经实现该表示。优化重点不是增加更多 decoder，而是验证 offset 是否解决 center+watershed 的偏移、重复与小目标问题。

### 4.3 IRNet / TernausNetV2：边界只负责拆分，不负责损毁分类

边界监督应该表达“这里可能是两个接触建筑之间的分界”，而不是表达“这里更像 Damaged”。这与 CMR-C0 失败的局部 D-I 热区完全不同。

### 4.4 实例感知语义损失：从最低风险版本开始

第一项训练实验不应直接上完整 center+offset+boundary。先做 one-vs-rest 的组件均衡语义损失：每个损毁类别独立形成组件，每个组件总权重接近相等，避免大建筑或大连通块支配像素损失。

它的优点是：不需要相信组件就是准确实例，只要求小区域和大区域在损失中拥有更公平的权重。

### 4.5 不建议直接照搬

- 不把 SAM/PointSAM 作为主方法依赖；它可做外部基线，否则论文会变成基础模型微调。
- 不直接迁移 BESTIE 全部训练管线；BRIGHT 是灾前 RGB、灾后 SAR、损毁多分类和跨事件任务，数据假设不同。
- 不继续使用 P3 D-I 热点构造关键区域；已有 C0 证据已经否定其安全性。
- 不用灾害事件作为损毁标签，也不再做事件中心相减。

## 五、Idea Evaluator：审稿人式评估

### 5.1 五维评分

| 维度 | 分数 | 证据级别 | 评价 |
|---|---:|---|---|
| Higher | 7/10 | 机制支持，尚未实证 | H7 和 RGID oracle 显示上界，但伪实例到原始语义收益尚未闭环 |
| Faster | 4/10 | 设计推断 | 多头训练和实例解码不会天然更快；速度不是主卖点 |
| Stronger | 7/10 | 机制支持 | 实例均衡可能改善小建筑与跨事件稳定性，必须用 event-held-out 验证 |
| Cheaper | 8/10 | 任务设定支持 | 主分支只需语义标签；必须用标注预算曲线量化，而非口头声称 |
| Broader | 7/10 | 文献迁移支持 | 可推广到其他密集目标语义任务，但需至少一个外部或跨事件验证 |

综合不是“已经可以写成论文”，而是“值得做一个严格的决定性实验”。

### 5.2 致命风险审计

#### MAJOR-1：最接近工作已经存在

BESTIE 已经研究 semantic-to-instance，Panoptic-DeepLab 已经研究 center+offset，TernausNetV2 已经研究建筑接触边界。单纯“语义标签生成伪实例并训练中心头”新颖性不足。

修复方式：论文贡献必须聚焦于 RGB–SAR 建筑损毁和跨事件条件下的**双向收益**：伪实例不仅产生实例，还通过实例均衡/一致性监督改善 raw semantic damage；并量化相对真实实例监督的收益—成本比例。

#### MAJOR-2：后处理提升不能算语义学习提升

如果 E2/E3 只是池化或重赋值后提高 mIoU，而 raw semantic logits 不变，审稿人会认为只是后处理。

修复方式：把 raw semantic head 设为第一主表；reconstructed semantic 仅作第二表。

#### MAJOR-3：伪组件不等于建筑实例

同类粘连、碎片化和错误类别都会破坏伪实例监督。

修复方式：真实实例只用于离线测量 pseudo quality；训练采用 quality gating、ignore mask 和自精炼；单独报告 same-class-touching 子集。

#### MAJOR-4：研究范围过大

RGID、Memory、多原型、事件因子、关键区域和语义提升同时存在会让论文失去主线。

修复方式：主论文只保留“pseudo structure → instance-aware semantic learning”。RGID 是输出解码与几何评价工具；Memory 放入失败分析或完全移除。

#### MINOR：官方实例训练标签已经存在

完全不使用它们不是天然优势，必须证明为什么语义-only 值得研究。

修复方式：做 0%、1%、5%、10%、25%、100% 真实实例标注预算曲线；比较 pseudo-only、少量真实+其余 pseudo、full-supervised。

### 5.3 生命周期与能力匹配

- 项目类型：数据密集型应用研究 + 训练算法改进。
- 当前阶段：问题形成末期 / 决定性验证前期。
- 算力风险：中；现有 5090 足以完成分阶段训练，但全量多 seed 和 leave-one-event-out 成本较高。
- 数据风险：低到中；语义和实例数据存在，但两个验证协议可能不重叠，必须统一 split。
- 工程风险：中；center/offset/decoder 已实现，主要新增联合训练、置信度门控和统一评价。
- 最大管理风险：同时推进太多支线，而不是代码能力不足。

### 5.4 Paradigm-shift probe

- First-principles：部分成立。建筑损毁标签属于建筑实例，但像素 CE 不保证实例公平。
- Elephant-in-the-room：成立。真实实例标注昂贵，且官方挑战显示跨事件泛化弱。
- Technology-cycle：不明显。不是由新硬件或新平台直接催生。
- Hamming：部分成立。问题重要，但已有弱监督实例与全景分割工作，需靠 BRIGHT 特殊性和实证区分。

结论：这是一个**有潜力的增量方法，不是范式革命**。如果严谨展示更低实例标注成本和跨事件语义收益，论文意义足够；若只展示 connected components + watershed，则意义不足。

## 六、分阶段、可停止的实验设计

### Stage 0：统一 split 的无训练因果审计

目的：先回答“理想建筑实例归属是否真的能改善相同样本上的语义结果”。

所有行必须在同一个拥有语义标签与官方实例标签的 split 上运行，不能把 450 张语义验证集和 697 张实例验证集横向比较。

| 行 | 建筑区域来源 | 类别来源 | 回答的问题 |
|---|---|---|---|
| E0 | 无实例处理 | P3 pixel argmax | 原始语义基线 |
| E1 | binary CC | P3 logits pooling | 只做二值连通域是否有效 |
| E2 | damage-class CC | P3 logits pooling | 按损毁类别分组件是否有效 |
| E3 | learned RGID | P3 logits pooling | 可部署伪实例是否优于简单 CC |
| E4 | official GT instances | P3 logits pooling | 完美几何归属的可达上界 |
| E5 | official GT instances | P3/A2 oracle class | 不可部署的总上界，仅解释潜力 |

Stage 0 预注册门槛：

- E4 相对 E0 的 valid mIoU 至少 +0.5 个百分点；
- building-macro F1 至少 +1 个百分点；
- Damaged F1 至少 +1 个百分点；
- 至少 10/14 个事件不下降；
- Intact IoU 不下降超过 0.5 个百分点。

如果 E4 都不能通过，停止“实例结构改善语义”的主叙事，只保留实例分割研究。E5 不能替代 E4，因为 E5 混入了 oracle 类别。

### Stage 1：伪实例质量画像

目的：把 pseudo label 错误拆成可修复类型，而不是直接训练。

必须统计：

- pseudo region 对 true instance 的 Precision、Recall、F1；
- AR@IoU 0.25/0.50/0.75；
- Merge、Split、matched IoU；
- center recall within 1/2/3/5 head pixels；
- duplicate-center ratio；
- threshold 前、NMS 后、marker cap 后的 center recall；
- 256 head 分辨率下 parent 消失比例；
- small/medium/large、same-class-touching、per-event 分层结果。

由此生成训练置信度：

- reliable：单峰、紧致、不过度触边、与语义边界一致；
- ambiguous：疑似同类粘连、过长、洞多、多峰；
- ignored：过小、头尺度消失、严重截断、标签冲突。

### Stage 2：最小训练阶梯

每一步只增加一个变量，并以前一步为基线。

| 实验 | 新增内容 | 为什么先/后 |
|---|---|---|
| T0 | 原始 P3 训练 | 同 seed、同 schedule 基线 |
| T1 | one-vs-rest 组件均衡语义损失 | 最不依赖伪实例精确边界，风险最低 |
| T2 | + reliable center auxiliary | 直接针对 RGID 最大瓶颈 |
| T3 | + pixel-to-center offset | 处理中心偏移、粘连和多实例分配 |
| T4 | + contact boundary | 只在 same-class touching 子集证明必要后加入 |
| T5 | + confidence-aware online refinement | 只有静态伪标签已产生稳定增益后再做 |
| TF | true-instance full supervision | 上界，不属于主方法 |

关键实现对照：

- Frozen：继续现有冻结 backbone，只测实例几何可学性；不能用于语义收益主结论。
- Joint：语义损失和伪几何损失共同更新共享 encoder，这是核心实验。
- Detached warm-up：先只训练语义，再逐步打开几何梯度，防止早期伪标签污染表征。
- Head-only control：增加相同参数量但无结构监督的头，排除“多参数自然变好”。

建议的总损失逻辑为：

    L = L_semantic + lambda_comp L_component-balanced
      + lambda_center q L_center
      + lambda_offset q L_offset
      + lambda_boundary q_boundary L_boundary

其中 q 是伪区域质量，不可靠区域置零或只参与弱一致性。所有 lambda 只在 Train/cross-fit 上选择并锁定到 Validation。

### Stage 3：必要的负对照

没有这些对照，无法证明收益来自实例结构：

1. Random regions：数量和面积分布与伪区域相同，但位置随机。
2. Grid regions：固定网格分区。
3. Shuffled IDs：保留像素类别，打乱区域身份。
4. Class-balanced CE only：只做类别重加权。
5. Same-parameter auxiliary head：相同计算量但预测无关目标。
6. Post-process only：训练不变，只在输出后用 CC/RGID 重赋值。
7. True-instance upper bound：真实中心/偏移监督。

只有伪实例结构优于 random/grid/shuffled/class-balance，才能把提升归因于“实例结构”。

### Stage 4：标注成本实验

训练真实实例比例：0%、1%、5%、10%、25%、100%。

每个比例比较：

- 仅真实实例；
- 仅伪实例；
- 少量真实实例 + 其余伪实例；
- 完全语义基线。

横轴是人工实例标注量，纵轴同时画 raw semantic valid mIoU、Damaged F1 和 instance segm_AP。建议预注册：伪实例方法至少恢复 full-supervised 相对语义基线增益的 30% 才值得继续，达到 50% 可视为强结果。该比例是实验门槛，不是已有结论。

### Stage 5：跨事件泛化

官方挑战已经显示域内与未见事件存在大差距，因此随机切分不够。至少进行：

- leave-one-event-out；或
- 按灾害类型保留一个地区作为未见域；
- event-macro 和 worst-event 指标；
- 按灾害类型、建筑尺寸、损毁类别分层。

事件只用于划分和汇总，不参与损毁标签预测。

## 七、指标与统计规范

### 主指标

1. raw semantic valid mIoU；
2. Damaged F1/IoU；
3. Intact IoU 与 Harm Exposure；
4. event-macro mIoU 和 worst-event mIoU。

### 实例指标

- 官方 segm_AP、AP50、AP75、per-class AP；
- PQ、SQ、RQ（若协议可实现）；
- Precision、Recall、F1、Merge、Split、matched IoU。

### 统计

- 至少 3 seeds；
- 对同一验证图像做 paired bootstrap 置信区间；
- 同时对 event 做宏平均，避免大事件支配；
- 所有阈值和 loss weight 只在 Train/cross-fit 中选择；
- Test 只在最终方法冻结后访问一次。

官方 Mask R-CNN validation mAP 0.1854 可作为外部参照，但只有同 split、同标注版本、同官方评测脚本时才能直接比较。

## 八、对现有代码的具体优化顺序

### 优先级 1：不要先改 decoder，先增加 Joint 训练入口

复用 `instance_geometry` 的 targets/losses/head，新增隔离的联合训练脚本与配置，让几何损失能够更新共享特征；保留 frozen 脚本作为诊断对照。绝不能直接修改旧 P3/S8 实验入口。

### 优先级 2：先实现组件均衡语义损失

它不需要改 decoder，不会把模糊伪组件当作精确中心，最适合作为第一项决定性训练实验。

### 优先级 3：增加 pseudo-quality audit 与 confidence mask

先用真实实例离线标定哪些伪区域特征与 Merge/Split 相关，再冻结质量规则。真实实例不能被用来逐样本修正主训练伪标签，否则会泄漏监督。

### 优先级 4：Center 后接 Offset，再决定 Boundary

Oracle 证据显示 Center 是第一瓶颈；offset 已有实现。Boundary 在 oracle center 下增益很小，因此不应排在 center/offset 前面。

### 优先级 5：Center 改善后重标定 Rollback

当前 rollback 对 learned center 略有帮助，却使 oracle G2 到 G3 的 F1 从 0.663134 降到 0.656250。说明它是针对 noisy center 的补丁，不应固定为最终规则。

### 优先级 6：最后处理高分辨率小建筑

约 3.96% oracle parent 在 256 head 尺度消失，约 2.34% marker 需吸附/位移。只有 T2/T3 已证明有效后，再比较 256、512 或双尺度 center head。

## 九、技术论文逻辑骨架

论文类型：**Technique Paper，带跨领域迁移属性。**

### 9.1 Thinking template

| 项目 | 内容 |
|---|---|
| Research background | RGB–SAR 建筑损毁要求像素级损毁分类，也天然以建筑实例为语义单位；真实实例标注昂贵且跨事件泛化困难 |
| Existing limitation | 像素 CE 被大建筑与大连通区支配；语义输出粘连相邻建筑；现有弱监督实例方法未处理跨模态损毁等级和事件域偏移 |
| Key idea | 从语义损毁标签构造置信度感知的伪建筑结构，以组件均衡、中心/偏移和一致性监督反向约束共享 RGB–SAR 表征 |
| Challenge 1 | 同类相邻建筑在语义标签中粘连，伪组件不等于实例 |
| Module 1 | quality-aware pseudo structure：reliable/ambiguous/ignored + self-refinement |
| Challenge 2 | 伪标签噪声可能反向伤害语义分类 |
| Module 2 | instance-balanced semantic loss + warm-up/gradient gating + raw-head evaluation |
| Challenge 3 | RGB/SAR 风格和事件差异会制造伪边界与伪中心 |
| Module 3 | 共享结构监督但保留模态特征，使用 event-held-out 和 event-macro 校准 |
| Contribution 1 | 提出针对 RGB–SAR 损毁的语义到伪建筑结构学习问题与统一语义/实例评价协议 |
| Contribution 2 | 提出置信度感知、实例均衡的联合训练方法，在不使用真实实例训练主分支时反哺 raw semantic head |
| Contribution 3 | 用真实实例上界和标注预算曲线量化伪实例监督能恢复多少全监督收益 |

后三项贡献目前是**待验证贡献**，不能在实验通过前写成既成事实。

### 9.2 Introduction 六段逻辑

1. 建筑损毁响应需要全天候 RGB–SAR 和建筑级损毁决策。
2. 现有语义模型按像素优化，忽略建筑实例公平性并产生粘连；真实实例标注又昂贵。
3. 关键观察：语义损毁标签虽然没有实例 ID，但包含类别连通、中心、边界和区域一致性等不完全结构。
4. 三个挑战：同类粘连、伪标签噪声、跨事件模态漂移。
5. 方法：quality-aware pseudo structure、instance-balanced joint learning、event-held-out validation。
6. 贡献：问题设定、方法、成本—收益与跨事件实证。

### 9.3 方法章节

1. Problem formulation and supervision budget
2. Semantic pseudo-region construction
3. Quality-aware pseudo center/offset/boundary targets
4. Instance-balanced joint semantic–geometry learning
5. Instance reconstruction and building-level damage pooling
6. Training schedule and leakage-safe protocol

### 9.4 实验章节

1. Dataset, unified split, official metrics
2. Stage-0 oracle and pseudo-quality audit
3. Main semantic and instance results
4. Component/center/offset/boundary ablation
5. Negative controls and attribution
6. Annotation-budget curve
7. Cross-event generalization
8. Failure analysis: same-class touching, small buildings, SAR scatter

## 十、最终决策树

1. **先跑 Stage 0。** E4 不过门槛：停止语义反哺主线，保留实例论文。
2. E4 通过：跑 T1 组件均衡损失。
3. T1 raw semantic 无稳定增益：不进入复杂中心/偏移，结论是伪组件不适合作为语义训练信号。
4. T1 通过：加入 T2 center；用 RGID center recall 和 raw semantic 双指标决定。
5. T2 通过：加入 T3 offset；只在 same-class touching 和 merge 子集预期明显收益。
6. T3 仍通过：再考虑 confidence self-refinement、多尺度和少量真实实例混合。
7. 任一步只改善 reconstructed output、不改善 raw head：只能声称实例后处理收益，不能声称语义学习收益。

## 最简结论

你的方向有科学依据，也不是“脱裤子放屁”，前提是把目标从“用伪实例代替已有真实实例”改成：

> 测量语义标签中可免费提取的建筑结构，能以多低的实例标注成本，恢复多少全监督实例结构带来的语义与实例收益。

最重要的下一步不是继续优化 Memory 或 watershed 参数，而是完成同 split 的 Stage 0，并新增严格隔离的 Joint 训练入口，从组件均衡语义损失 T1 开始。只有 T1 的 raw semantic 指标稳定提高，才进入 center、offset 和自精炼。

