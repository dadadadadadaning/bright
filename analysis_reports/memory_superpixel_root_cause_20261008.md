# DamageFormer 区域 Memory 失效的根因审计与后续路线

日期：2026-10-08  
范围：Train/Val；不使用 Test 选择方案。  
性质：代码审计、已有日志综合、文献对照。未修改模型，未启动训练。

## 1. 一句话结论

当前结果不能解释为“Memory 没读到历史信息”，更准确的结论是：

> Train Memory 中存在一定标签相关信号，但当前 MHA 读出器没有把它稳定转成损毁分类收益；同时，现有区域 Token 与余弦检索仍未证明能为 Damaged/Destroyed 提供足够的跨事件近邻，硬建筑分组和区域汇总还会继承定位与类别混合误差。

因此，问题不是单一模块损坏，而是一条串联系统的接口错配：

`区域支持域 Q → 区域 Token → 检索 Key → Top-K 支持 → Reader → 区域回投`。

其中任何一环失效，Memory 都可能不如 query-only/noMemory。

## 2. 当前实际流程，不是图示的严格复现

当前代码实际执行：

```text
冻结的区域联合训练前端（不是原始 A1）
        ↓
原图融合特征 + 已有 Soft ROI
        ↓
冻结 Q + 预测建筑概率 B>=0.5 的建筑/背景硬拆分
        ↓
空间注意力 Pooling + mean/std 旁路
        ↓
区域 Token z 与 source_hard 区域 logits
        ↓
Train Bank 中按余弦 Top-K 检索区域 Token 与软标签
        ↓
MHA 或显式标签后验产生区域 logit 修正
        ↓
同一路由回投到像素
```

图示系统还画出了灾害类型、区域几何等元信息注入；当前 reader 没有使用这些信息。图示也不能告诉我们原方法是否使用了对比损失、Key 投影、联合训练、动态 Bank 或具体筛选规则，因此不能把目前实现称为严格复现。

## 3. 已有结果说明了什么

### 3.1 新 Selective Memory 仍是中途结果

当前已知到 1500/5000 步：

| 模式 | 前景 mIoU | 相对 noMemory |
|---|---:|---:|
| noMemory | 61.6080% | 0 |
| global_hard | 61.6018% | -0.0062 pp |
| fixed_balanced | 61.6022% | -0.0058 pp |
| adaptive_cap | 61.5963% | -0.0117 pp |

差异非常小，且尚未完成，不能宣布最终失败；但目前没有 Memory 独立增益。

### 3.2 上一轮完整 5000 步提供了更强机制证据

| 模式 | 前景 mIoU |
|---|---:|
| noMemory | 61.5685% |
| global_hard | 61.5571% |
| balanced_hard | 61.5656% |
| random support | 61.5843% |
| shuffled labels | 61.5642% |
| support off | 61.5428% |

这些差值很小，不能把单次 random 略高解释为“随机检索更好”。但它们说明：

1. 支持是否存在会影响输出，因为 off 更低；
2. 正常语义近邻并没有显示出高于 random 的特异收益；
3. 打乱支持标签后几乎不变，说明旧 MHA reader 很可能没有充分利用标签语义。

### 3.3 Memory 其实包含一定标签信号，但没被 Reader 转化

已有 label-only 审计中：

- balanced semantic kNN accuracy：92.3238%；
- random kNN accuracy：89.2458%；
- shuffle-label kNN accuracy：39.6084%；
- semantic soft-CE：0.319054，优于 random 的 0.334020。

这说明“Bank 完全无信息”不成立。但这个总体统计很可能受 Background/Intact 主导，尚不能证明 Damaged 与 Intact 的近邻可分性足够。

与此同时，adaptive 检索在约 1500 步时，Damaged 支持覆盖约 19.5%，Destroyed 约 22.3%；global 对应约 14.3% 与 8.2%。所以“每个查询都有邻居”不等于“每个查询都检到了需要的稀有类证据”。

## 4. 根因排序

### 4.1 最高优先：旧 MHA Reader 没有把标签证据转成收益

Memory 路径把支持值写成：

```text
support_token + soft_label_embedding
```

然后将查询自身也追加为一个永远有效的 fallback，再用 MHA 输出残差。Reader 没有显式使用余弦分数、纯度、熵或类别覆盖。它可以通过查询自身完成大部分校准，从而忽略支持标签。

noMemory 则使用四个稳定的可学习类别槽位和同一个查询 fallback。它面对的是稳定、低噪声输入，优化难度比八个跨事件近邻更低。因此 noMemory 略优并不奇怪。

最小判别：完成显式 posterior 实验。只有当 `posterior_quality_gate > query_only`，并且 random/shuffle/off 明显下降时，才能说 Memory 标签被有效利用。

方案二还有一个必须监控、但尚不能称为已发生故障的优化风险：`strength` 从 0 开始并被限制在 `[0,1]`。在 `strength=0` 时，主损失对 gate 参数的梯度被该乘子截断；需要先由 `strength` 离开 0，gate 才能从 Memory 分支得到有效学习信号。正式日志必须记录 strength、gate 均值及二者梯度；若 strength 长期钉在 0，不能把结果解释为“显式 posterior 无效”，因为它实际上没有被启用。

### 4.2 上游核心：Key 空间没有按跨事件损毁相似性训练

当前直接对冻结区域 Token 做 L2 归一化余弦检索，没有单独的 query/key 投影，也没有 metric、triplet 或 supervised-contrastive 目标。高 cosine 只能证明向量接近，不能证明它们具有相同损毁等级。

尤其在 RGB–SAR BDA 中，Token 可能更容易编码建筑外观、事件风格、背景和区域尺度，而不是细微的 Intact/Damaged 区别。

最小判别：按 GT 类别离线报告 foreground-only Recall@K、top-k 同类率、D↔I soft-CE 和每事件结果；再比较 raw Token、source logits/probability 与一个仅在 Train 上训练的小型 Key projector。

### 4.3 一个具体的路由接口缺口：检索没有保持预测建筑组一致

硬路由使用偶数槽表示预测背景组、奇数槽表示预测建筑组；Bank 也保存原槽位编号。但当前检索只根据 Token cosine、事件/图像排除和 dominant damage class 筛选，没有把 query 槽位的奇偶组传入检索约束。因此预测建筑 query 仍可能取到预测背景 support，反之亦然。

这不是 GT 泄漏问题，因为组别来自模型预测 B；它是当前区域设计已经拥有、但 Memory 接口没有继续使用的信息。

最小判别：不训练，统计 Top-K 的 predicted same-group 比例，并分别报告建筑 query 的 D/I kNN accuracy、soft-CE；再做 same-group-only 检索干预，缺少候选时只 mask，不用 GT 回填。若显著改善，再正式加入 predicted-group compatibility；若无改善，不应继续加强硬 ROI。

### 4.4 区域单元与 Memory 语义单位不一致

当前槽位是“软 Q 的一个 seed，再按预测建筑/背景二分”，不是一栋真实建筑：

- 不保证一个槽位只覆盖一栋建筑；
- 不保证连通；
- 可能把同一建筑切成多个碎片；
- 可能包含不同损毁等级或多个建筑；
- 软标签用 dominant class 入桶，会隐藏少数类别成分。

因此 Memory 中保存的不是稳定的“建筑案例”，而可能是大小、纯度和组成不同的区域片段。

但 Q 几何并非当前唯一或最大问题：此前把已有像素类别概率按 Q 汇总再回投几乎不降，而高维区域分类路径下降更多；学习 Q+建筑分组也优于固定网格+建筑分组。这说明更应区分“Q 的空间支持域”和“Token 的语义表示/分类方式”。

### 4.5 Bank 筛选存在“干净但稀有类缺席”的矛盾

adaptive_cap 只设置每类上限，不设置最低配额。它会得到高纯度、高 cosine 的邻居，但候选仍可能被 Background/Intact 占据。fixed_balanced 强制每类配额，又会把不相似或低质量的稀有类支持塞进来。

这不是单纯调一个 top-k 能解决的；根因仍是 Key 空间没有把正确稀有类排到前面。

质量过滤还同时使用 purity、entropy 与 hard_count，可能偏向大、简单区域，排除小且困难的 Damaged。需要做数量匹配的 purity-only / purity+entropy / 三条件诊断。

还有一个顺序问题：现有流程先在每个 event×dominant-class 桶中用随机 reservoir 截断到最多 64，再在这份小库上做质量分位数过滤。这不是“从全部 Train 候选中选择最好支持”，而是“先随机丢弃，再继续删减”；被提前丢掉的高质量或多样化支持无法恢复。应比较数量匹配的：

- A：当前 random-cap → quality-filter；
- B：all candidates → quality-filter → event×class cap/diversity selection。

先看前景 D/I 检索证据，不要直接扩大库并把容量增益解释成方法增益。

### 4.6 Bank、训练 Query 与 Val Query 的尺度/上下文不一致

Bank 在完整 1024 图像上提取，训练 Query 来自随机 512 crop，Val Query 又是 1024。区域 Token 依赖区域质量、邻域和 Pooling，因而比普通像素头更容易受这种上下文差异影响。

最小判别：比较同一图像 512 crop 与 1024 full-image Token 的分布与 Top-K 标签一致性，或建立一个数量匹配的 512 诊断 Bank。

### 4.7 默认排除同事件，跨事件域差异未被建模

排除同事件能防止过度依赖事件风格，是严格的泛化设置；但 BRIGHT 跨事件、灾害类型、SAR采集条件和建筑风格差异明显。当前事件 ID 只用于排除，没有作为可用元信息进入 Reader。

最小判别：始终排除同图像，比较“允许同事件”与“排除同事件”的离线检索和固定权重推理。如果只在同事件检索时有效，只能宣称域内案例检索，不能宣称跨事件 Memory。

### 4.8 像素损失会淹没少数困难区域的 Memory 信号

当前仍以区域回投后的全像素 NLL+Lovasz 训练，普通随机 crop 也不是 damage-aware 采样。大背景和大区域产生的梯度远多于小 Damaged 区域。这对 Memory/noMemory 都相同，不能单独解释两者差值；但它会让“只帮助少量困难 Damaged 查询”的信号在总体优化中被稀释。

第一步不要立刻换损失。先按 GT 主类与区域 mass 分桶比较 Memory 相对 query-only 的 rescue/harm。只有发现小 Damaged 区域存在稳定正信号但被总体指标淹没后，才单独验证 region-balanced query sampling 或 loss。

### 4.9 输出拓扑限制了上限

当前 Memory 修正 source_hard 区域 logits，再按硬路由回投；它没有以原始 A1 像素 logits 为安全锚点。于是 Memory 继承区域平均、边界损失、建筑定位错误和 D→I 混淆。

这主要解释“为什么难以可靠超过 A1”，不完全解释“为什么 Memory 略低于 noMemory”。后者更直接是 reader 与检索信号的问题。

### 4.10 冻结和步数不是当前第一嫌疑

冻结确实限制最终上限，但它是合理的机制诊断：noMemory 在同样冻结条件下仍能学习并改善，说明训练并没有整体失效。旧实验跑满 5000 步后仍未出现语义 Memory 优势，因此不能把问题简单归因于步数不足。

## 5. 文献为什么能用 Memory/超像素，而当前条件不完全满足

| 工作 | 成功机制 | 对当前方案的启示 |
|---|---|---|
| [SSN, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.html) | 可微软归属，允许下游任务塑造 task-specific superpixels | Q 不应只作为冻结的结构压缩器；若后续调整 Q，必须用任务目标并保持 Pool/Unpool 同一路由 |
| [SEAL, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Tu_Learning_Superpixels_With_CVPR_2018_paper.html) | segmentation-error-aware affinity；论文明确指出简单换成预训练深特征并不会自然得到更好超像素 | “能分类的特征”不自动等于“能分区/能检索的特征” |
| [SpixelFCN, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Superpixel_Segmentation_With_Fully_Convolutional_Networks_CVPR_2020_paper.html) | 直接预测局部 grid association，并与下游任务联合使用 | 当前 9 邻域接口合理，但固定 Q 是否有益必须用 fixed-vs-learn 对照验证 |
| [AINet, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_AINet_Association_Implantation_for_Superpixel_Segmentation_ICCV_2021_paper.html) | 显式 pixel-grid 关系与 boundary-perceiving loss | 若证据指向边界/连通，再改 association；不是先换主干 |
| [Superpixel Transformer, IROS 2023](https://arxiv.org/abs/2309.16889) | local cross-attention 联合生成 Token，Token 间 global self-attention，再按同一 association 回投；全流程由分割目标训练 | Transformer 不是不适合超像素；关键是 Tokenization 与任务联合、粒度和全局 Token 交互 |
| [BRDG, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Ahmed_Boundary-Responsive_Differentiable_Gating_for_Superpixel-Based_Segmentation_CVPR_2026_paper.pdf) | 用边界响应门控融合区域描述与原像素特征/坐标，避免完全区域化造成细节丢失和过度平滑 | 支持保留 A1 像素锚点，让区域/Memory 只作可靠性受控的残差，而非全量接管 |
| [MCIBI, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Jin_Mining_Contextual_Information_Beyond_Image_for_Semantic_Segmentation_ICCV_2021_paper.pdf) | 动态类别记忆、GT 监督的类别分布、representation consistency | Memory 不是静态 Top-K 插件；表示空间与 Memory 使用目标共同训练 |
| [Class-wise Memory Contrast, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Alonso_Semi-Supervised_Semantic_Segmentation_With_Pixel-Level_Contrastive_Learning_From_a_Class-Wise_ICCV_2021_paper.html) | 持续更新的高质量类别特征 + 端到端对比学习 | 高质量筛选只是第一步，还必须让 query 与同类 support 在表示空间靠近 |
| [RCA, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Regional_Semantic_Contrast_and_Aggregation_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2022_paper.pdf) / [ReCo, ICLR 2022](https://github.com/lorenmt/reco) | 区域 Memory 同时承担语义对比/聚合或困难负样本学习 | 有效 Memory 通常伴随显式表示学习，而不是只在推理端做邻居注意力 |
| [XBM, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html) | Memory 服务于 metric-learning objective，并讨论旧特征漂移 | 一旦解冻 Token 编码器，静态 Bank 会过期，需重建或使用 EMA/momentum key encoder |

BRIGHT 本身存在明显的跨事件与 SAR 成像差异；官方论文也把跨事件泛化列为核心挑战，并显示灾前光学信息不只用于定位，光学与 SAR 融合对损毁等级判别都有贡献。因此，纯 RGB 结构 Q 可以用于边界，但 Memory Key 应保留融合后的损毁证据，不能只按建筑外观找邻居：[BRIGHT 数据集论文](https://essd.copernicus.org/articles/17/6217/2025/)。

## 6. 最小排除实验：先定位，再重训

### E1：Reader 是否忽略 Memory

固定 checkpoint、Bank、Top-K 和 Q，比较：

1. normal semantic support；
2. random support；
3. shuffle labels；
4. support off；
5. label-only posterior；
6. oracle same-class support（仅离线诊断，GT 不进入推理）。

同时记录 support attention mass、自查询 attention mass、Memory 参数梯度、区域类别改变率和修正幅度。

当前约 1500 步时，修正绝对幅度约为 base logits 的 4%，但区域 argmax 改变率只有约 0.11%–0.15%。因此还应专门统计“真正被改变的区域”中的 rescue/harm、D↔I 转移，以及 Memory 与 noMemory 是否改变同一批区域；只看修正幅度会高估实际作用。

判定：

- semantic、shuffle、off 几乎一致：reader 忽略 support；
- label-only 有效、MHA 无效：reader 是近端瓶颈；
- oracle 有效、semantic 无效：Key/检索是瓶颈；
- oracle 也无效：区域单元或输出拓扑是瓶颈。

### E2：Key 是否能区分 Damaged 与 Intact

按查询 GT 主类与事件报告：

- Recall@1/4/8；
- 至少一个同类支持的概率；
- support posterior soft-CE；
- Damaged 查询检到 Intact 的比例；
- Destroyed 查询检到 Damaged/Intact 的比例；
- semantic 相对 random 的差值。

必须只看前景条件统计，不能让 Background 的高准确率掩盖失败。

同时增加 predicted-group 审计：query 为预测建筑组时，Top-K 中预测建筑 support 的比例；query 为预测背景组时同理。该组别只由 B 推理得到，不使用 GT。

### E3：区域单元是否破坏可检索语义

比较相同内容特征下：

1. 像素或小 patch 表示；
2. 当前 Q Token；
3. 当前 Q + 预测建筑软二分；
4. GT 建筑连通域代理 Token（仅上限诊断）；
5. 高纯度区域子集。

报告 region purity、D/I entropy、跨 GT 连通域代理比例、碎片数、区域面积、Memory rescue/harm 的关联。

### E4：域与尺度错配

- 允许同事件但排除同图像 vs 排除同事件；
- 512-query/512-bank vs 512-query/1024-bank vs 1024-query/1024-bank；
- 每事件报告检索质量与分类增益。

## 7. 推荐的 DamageFormer 改进顺序

### 阶段 A：完成当前实验，不再并行加模块

1. 跑完 Selective retrieval 5000 步；
2. 跑显式 posterior 四个匹配对照；
3. 做 E1–E4 只读审计；
4. 若 semantic 不优于 query-only 且 shuffle/off 无明显下降，停止当前 Memory reader。

### 阶段 B：把 A1 像素输出作为安全锚点

推荐拓扑：

```text
L_final(x) = L_A1(x) + beta * sum_k Q(x,k) * g_k * DeltaL_k
```

- `L_A1`：冻结并保留的原 A1 像素 logits；
- `DeltaL_k`：区域/Memory 只预测修正量；
- `g_k`：由检索可靠性决定的门控；
- `beta`：零初始化或受限小幅度。

公平对照：

- P0：A1；
- P1：A1 + 同容量 query-only 区域残差；
- P2：A1 + semantic Memory 区域残差。

只有 `P2 > P1` 才能归因于 Memory；`P2 > P0` 才有实际价值。

### 阶段 C：只训练小型检索投影，不解冻整个 DamageFormer

建立 `z_key = normalize(P_key(z_region))`，使用 Train-only 的监督对比或 triplet 目标：

- 正样本：同损毁类、优先不同事件；
- 困难负样本：Damaged vs Intact，尽量同事件、相近区域尺度；
- 只使用高质量前景区域；
- Bank 与 query 使用一致分辨率和处理流程。

先证明 foreground Recall@K、D/I soft-CE 和 semantic-vs-random 明显改善，再训练 P2。

### 阶段 D：最后才局部适配 Q

如果 E3 证明区域支持域是瓶颈：

1. 把硬 `B>=0.5` 改为软建筑/背景二分：`Q_bg=Q*(1-B)`、`Q_fg=Q*B`；
2. 只解冻 Q decoder，不先解冻双 Swin；
3. 加入损毁标签重建、紧致度、边界/连通性中的一个最小目标；
4. 扫描三种区域密度；
5. Q 或 Token 变化后必须重建 Bank，或使用 EMA key encoder。

不要同时加入新主干、Memory reader、新类别损失和残差融合，否则无法归因。

## 8. 停止条件

- 显式 posterior 不能优于 query-only，且 semantic/random/shuffle 无特异差异：停止当前 Memory reader。
- foreground oracle kNN 尤其 D/I 不优于 random：停止调 gate/top-k，先训练 Key。
- oracle same-class Memory 仍无法改善区域错误：停止 Memory，回到区域 Token/像素锚点。
- P2 不优于 P1：Memory 没有增量价值，保留 A1 或 query-only。
- 只允许同事件才有效：将结论限制为域内检索，不宣称跨事件泛化。
- 三个种子后差值小于波动：不把 Memory 作为核心贡献。

## 9. 最终责任判断

当前没有证据表明代码加载、梯度或 Bank 为空等基础故障；step-0 复现、支持覆盖和梯度检查均正常。最可能的责任结构是：

1. **近端第一责任：Reader/融合接口没有有效使用支持标签。**
2. **上游核心风险：区域 Key 未针对前景损毁与跨事件检索训练。**
3. **具体接口缺口：建筑/背景硬分组信息没有被检索继续约束。**
4. **结构性上限：区域槽位不是建筑实例，硬建筑路由会固化定位错误。**
5. **放大因素：先随机截断再质量过滤、稀有类覆盖不足、事件与尺度分布错配，以及像素目标的背景主导。**
6. **次要因素：冻结和训练时长；它们限制上限，但不能解释已有干预结果。**

总体研究逻辑“用区域压缩结构，再从历史案例补充稀有类证据”是合理的；当前缺口在于把三个未经证明的假设串在了一起：区域 Token 可比、余弦近邻等于损毁近邻、MHA 会主动采用标签证据。下一步应逐一证伪，而不是继续堆叠 Memory 容量或更复杂主干。
