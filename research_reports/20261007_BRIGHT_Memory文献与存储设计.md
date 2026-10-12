# BRIGHT 区域 Memory 的作用 长尾机制与存储设计

## 摘要

当前 BRIGHT 区域检索实验尚未体现相对同容量无 Memory 对照的稳定增益。这个结果不能由“库不够大”或“超像素无效”直接解释。相关研究包含三种不同机制：推理时检索训练实例、借助类别知识改善长尾分类、训练时用历史特征构造对比监督。它们保存的信息、更新对象和推理依赖并不相同。针对固定 DamageFormer 前端、固定 Q 和建筑分组的现有设置，应先评价检索的等级语义，再决定是否训练独立检索键；不应同时解冻分区、扩大库、修改损失和增加融合模块。存储设计应保留区域特征、Train 四类标签分布及可追踪来源，并补充质量与覆盖度字段。存储完整不代表这些字段全部作为分类输入。

## 1 研究问题与实验边界

这份说明回答三个问题：

1. Memory 在已有研究中保存什么，在哪个阶段产生作用？
2. 改善长尾识别依靠哪些具体机制，哪些适合当前 RGB–SAR 任务？
3. 当前区域库应该保存哪些信息，下一项实验怎样验证其增量价值？

任务是灾前 RGB 与灾后 SAR 的四类建筑损毁分割，不是同模态变化检测。建筑损毁 GT 是建筑等级标注；区域标签混合不能解释为楼内局部损坏比例。GT 连通域只能作为对象代理。所有建库和监督限定 Train，开发与选择限定 Val，不用 Test 挑方案。

当前固定前端是 A1 初始化后区域联合训练 3000 步的快照，不是原始 A1。Memory 实验沿用固定区域表示、Q、B 和建筑／背景 hard 分组，组内归属仍是软 Q。历史 A1 的 61.6866% 不能视为这轮同预算基线。

用户已提供的最后完整 Memory 验证为 4500 步：无 Memory 61.5699%，Memory 61.5549%，原 hard 约 61.5216%。5000 步训练完成记录不等于最终验证和检索审计完成。这些差异支持“当前未见稳定检索增量”，不支持“Memory 导致大幅崩溃”。

## 2 检索范围与证据来源

检索日期为 2026 年 10 月 7 日。按推理检索、长尾识别、语义分割训练记忆三个视角检索原始论文及作者仓库，另核查无需 Memory 的反例。检索词覆盖 key-value cache、nearest-neighbor retrieval、long-tailed memory、balanced contrastive learning、pixel/region memory、semantic segmentation；同时检查 SRAC 公开材料与较新的相关条目。

纳入 10 篇机制可核对的代表性原始论文，发表年份为 2018—2022 年；这不是截至检索日全部工作的穷尽清单，也不以年份较新作为迁移依据。论文机制与公开实现有差异时分别注明。SRAC 的官方新闻可以确认框架概述，但不足以恢复建库比例、检索监督、损失权重和消融协议，因此不作为可严格复现的配置来源。[SRAC 实验室新闻](https://jianghai.ucas.ac.cn/index.php/zh/xshde/xsdte/461-igarss-2026)

工程证据来自本地镜像中的建库、检索、修正头和诊断代码；服务器最终结果仍以服务器产物为准。本次没有修改训练代码，没有重建库，也没有启动训练。

## 3 Memory 按用途分类

| 用途 | 保存内容 | 直接解决的问题 | 是否在推理查库 |
| --- | --- | --- | --- |
| 训练实例检索 | 特征键及训练标签、区域素材等值 | 给当前查询提供训练样本参照 | 是 |
| 长尾知识共享 | 类中心、参数原型或训练特征队列 | 类别覆盖、表示与梯度失衡 | 因方法而异 |
| 密集预测对比监督 | 像素或区域特征、类别关系、负例队列 | 训练跨图同类聚合与混淆类分离 | 通常否 |

这是按主要作用划分，不是按是否出现 `memory_bank` 变量划分。OLTR 跨越知识共享与推理增强；PaCo 同时使用队列与类别中心。ReCo、BCL 和 cRT 是重要的无持久实例库对照，不应为了统一名称把它们也说成推理检索方法。

## 4 推理检索怎样利用历史实例

kNN-LM 与 Tip-Adapter 都将冻结特征关联到训练标签，但前者对邻居标签形成概率并与原模型概率插值，后者对缓存相似度形成分类分数并与 CLIP 分数相加。它们并不需要当前这套 Cross-Attention。SIMS 则检索真实视觉区域用于图像合成，说明外部区域库可以有结构信息，但不构成损毁分类有效性的证据。

| 研究及原任务 | 保存与更新 | 对当前任务的启发与限制 |
| --- | --- | --- |
| Khandelwal 等，Generalization through Memorization，ICLR 2020，语言建模 | key 是上下文向量，value 是下一词；冻结模型，用 Train 建库，按距离聚合标签概率 | 可借鉴固定表示、训练标签参照与原预测保留；不是 RGB–SAR 分类，也没有超像素 |
| Zhang 等，Tip-Adapter，ECCV 2022，CLIP 少样本图像分类 | 归一化图像特征为 key，Train one-hot 为 value；原版固定，Tip-Adapter-F 训练缓存 keys | 最接近当前冻结表示路线；原版是全缓存核加权，不是 Top-8 attention，少样本均衡设置不等于长尾 |
| Qi 等，SIMS，CVPR 2018，语义布局到图像合成 | 区域 RGB、mask、周围语义 context；同语义类内找形状和上下文接近的素材 | 可借鉴来源和结构字段；其查询语义类已知，不能迁成按查询 GT 损毁类筛选 |

机制与实现来源：[kNN-LM 原文](https://arxiv.org/pdf/1911.00172)及[作者代码](https://github.com/urvashik/knnlm)；[Tip-Adapter 正式论文](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950487.pdf)及[作者代码](https://github.com/gaopengcuhk/Tip-Adapter)；[SIMS 原文](https://www.cse.cuhk.edu.hk/~leojia/papers/imagesyn_cvpr18.pdf)及[作者代码](https://github.com/xjqicuhk/SIMS)。

两个实现边界值得保留。Tip-Adapter-F 的所核公开版本在每轮使用 test 指标保存最佳模型，BRIGHT 不能照搬这个选模流程，须统一使用 Val；这项代码观察不能反推论文实验必然采用了同样流程。[固定版本选模代码](https://github.com/gaopengcuhk/Tip-Adapter/blob/e6bdba44e5d8a06f6363a64ae7ec4950b9631512/main.py#L111-L124) SIMS 论文描述包含 context-IoU，但公开检索脚本主要实现 mask-IoU，README 说明 context 部分尚待加入。[SIMS 公开实现说明](https://github.com/xjqicuhk/SIMS/blob/master/README.md)

迁移判断：可先用相同邻居直接做软标签投票，判断“历史标签中有没有可用信号”，再评价复杂读出是否利用了它。投票不必单独超过完整分类器才算有价值，但应在原模型易错的前景区域体现互补，而不是仅在大量背景上正确。

## 5 长尾收益来自什么

OLTR 用类别中心共享知识，PaCo 用参数中心配合样本队列改变对比优化，BCL 则不依赖历史实例库而平衡类别对对比目标的贡献。三者共同提示：让少数类在存储中出现，不等于它们已在训练梯度和决策空间中获得足够作用。

| 研究 | 实际机制 | 对 BRIGHT 的可迁移部分 |
| --- | --- | --- |
| OLTR，CVPR 2019 | 学习每类判别中心，用输入相关权重组合中心，再门控融合自身特征 | 知识共享、按输入读取中心；不是原始特征余弦 Top-K 的直接依据 |
| PaCo，ICCV 2021 | 动量特征及标签队列加参数类别中心，改变样本与中心的对比权重 | 训练判别表示；需要相应表示更新，不能只加冻结库就复现 |
| BCL，CVPR 2022 | 当前批次特征与全类参数原型，类补全和按类平均对比贡献 | 控制类别供给和梯度失衡，不是推理查库 |
| cRT，ICLR 2020 | 学表示与训练分类器分离，冻结表示后用类别平衡采样训练分类器 | 必须保留的无 Memory 重平衡对照 |

来源：[OLTR 原文](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Large-Scale_Long-Tailed_Recognition_in_an_Open_World_CVPR_2019_paper.pdf)及[作者代码](https://github.com/zhmiao/OpenLongTailRecognition-OLTR)；[PaCo 原文](https://arxiv.org/abs/2107.12028)及[作者代码](https://github.com/JIA-Lab-research/Parametric-Contrastive-Learning)；[BCL 原文](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhu_Balanced_Contrastive_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2022_paper.pdf)及[作者代码](https://github.com/FlamieZhu/Balanced-Contrastive-Learning)；[cRT 原文](https://arxiv.org/pdf/1910.09217)及[作者代码](https://github.com/facebookresearch/classifier-balancing)。

cRT 论文附录 B.4 对比过 cRT 与 cRT 加 OLTR memory，在该 ImageNet-LT 设置中没有观察到额外提升。这一反例要求把“重平衡的收益”与“检索的收益”分开；它不证明 BRIGHT 上 Memory 必然无效。[原文附录 B.4](https://arxiv.org/pdf/1910.09217)

BRIGHT 首先需要的不是给 Damaged 自动加大权重，而是四种 Train 统计：每类有效像素数、每类对象代理数、事件×类别的覆盖、实际区域标签质量和数量。它们衡量不同长尾：大建筑可能贡献很多像素却只有少数对象；同一建筑的许多区域也不是许多独立建筑案例。Damaged 低分可能与类别相似、模态错位、事件偏差有关，不能直接等同于“样本最少”。

均衡也有四个不同位置：建库配额、查询采样、检索候选、损失贡献。当前只明确在建库时做事件×多数类别限额。不能把这四种平衡一次同时加入，再将所有变化归因于 Memory。特别是每类强制检索相同数量可能提升尾类供给，也可能引入很不相似的支持，必须单独验证。

## 6 分割文献中的训练记忆

ContrastiveSeg 与 U²PL 都提供跨批次特征供给，但前者利用 GT 类别关系，后者重点处理半监督情境的不可靠伪标签。ReCo 直接在当前批次构造类别原型和困难对比样本，因此能作为不维护队列的机制对照。

| 研究 | 存什么及如何用 | 不能照搬的部分 |
| --- | --- | --- |
| ContrastiveSeg，ICCV 2021 | 类别像素队列和区域记忆；同类正例、异类负例，训练跨图特征 | 其中 region 是图内某类像素的平均，不是建筑实例；推理不检索队列 |
| U²PL，CVPR 2022 | 按类维护负例队列；可靠表示用于 anchor，不可靠无标签像素经筛选提供负例 | 类别 c 的队列存“不是 c”的负例，不能把它们标成 c 用于 kNN |
| ReCo，ICLR 2022 | 当前批次类均值原型、难 query、易混淆类负样本 | 无持久跨批次库；该批没出现的稀有类也没有对应原型 |

来源：[ContrastiveSeg 原文](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Exploring_Cross-Image_Pixel_Contrast_for_Semantic_Segmentation_ICCV_2021_paper.html)及[作者代码](https://github.com/tfzhou/ContrastiveSeg)；[U²PL 原文](https://haochen-wang409.github.io/U2PL/resources/U2PL_CVPR_2022.pdf)及[作者代码](https://github.com/Haochen-Wang409/U2PL)；[ReCo 原文](https://arxiv.org/abs/2104.04465)及[作者代码](https://github.com/lorenmt/reco)。

ContrastiveSeg 公开代码把两种记忆实现为有限类队列；ReCo 则有直接的全监督入口，便于保留真实 GT。U²PL 的不可靠伪标签流程不应替换 BRIGHT 已有 GT。使用任一训练对比路线，都要明确允许更新哪些特征；全部冻结时，队列本身不能塑造前端表示。

## 7 当前工程实际保存与遗漏的信息

### 7.1 已实现的流程

```text
Train 原图 RGB＋SAR
  → 固定前端、Soft ROI、Q、B 与 hard 建筑分组
  → 固定区域 Embedding z
  → 每事件×区域多数 GT 类最多保留 64 个真实区域
  → 保存 z、四类软标签、来源 ID

查询 z → 原始 z 的归一化余弦 Top-8
       → 邻居 z＋支持标签嵌入 → Cross-Attention
       → 修正冻结区域 logits → 依固定 Q 回投
```

库不保存每个类别的均值替代真实区域。当前 3480 个是抽取的区域案例，而不是 3480 栋独立建筑。事件或图像来源相关性仍需记录。

`bank.pt` 主要字段如下：

| 字段 | 当前内容 | 用途 |
| --- | --- | --- |
| `tokens` | M×128 的区域特征，float32 | 同时承担检索表示和支持内容 |
| `distribution` | M×4 的 Train GT 软分布 | 支持标签嵌入；不是模型自预测概率 |
| `events/images/regions` | 每条记录的事件、图像、区域索引 | 来源追踪与同图／同事件排除 |
| `image_names/event_names/image_event_ids` | 名称和映射 | 解释索引 |
| `identity/format/split/smoke/test_accessed` | 固定源、版本和划分信息 | 阻止误用不兼容或非 Train 的库 |
| `per_bucket` | 每桶上限 | 建库配额约束 |

归一化后的检索向量在加载时生成，并不是另一个经过检索监督训练的 key。多数 GT 类只用于分桶；真正保存的支持标签仍是完整四类分布。标签由几何路由 Q 汇总有效 GT 得到，不使用查询 GT 排序。

代码入口：[建库](D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_region_memory_v2/build_bank.py:14)、[标签与抽样](D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_roi_followup_v1/memory.py:23)、[修正头](D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_region_memory_v2/core.py:23)。

### 7.2 现在还不能判断的原因

- 原始 z 的余弦邻域是否对损毁等级有意义，尚需最终报告。分类器可分不等于余弦近邻可分。
- 相同容量的类配额不保证每个事件真的有足够尾类支持，也不保证 Top-8 结果均衡。
- 当前允许所有正标签质量的区域入候选，没有纯度或最小质量筛选。`labelled_mass` 用于候选判断，但未逐条写进库；纯度可从 `distribution` 重算。
- 库用完整 1024 图提取，训练查询用 512 裁剪，存在上下文差异；这不等同于冻结库过期，也不是已确认的下降原因。
- Cross-Attention 在 Top-8 之后学习，不能找回没有被选入的候选。

`support_coverage=1` 只表示有合法支持，不是检索正确率。现有 `diagnostics.py` 已按类报告邻居真类软质量、投票一致率和同样排除规则下的可检索库先验，应该先读取这些产物。[现有检索诊断](D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_region_memory_v2/diagnostics.py:20)

## 8 推荐的 Memory 存储设计

建议把“用来找”“用来读”“用来审计”分开。下面是下一版数据规范，不代表本次已改库。

| 信息 | 建议形状或内容 | 是否直接进入分类器 |
| --- | --- | --- |
| `key` | M×Dk，归一化检索特征；起步仍可用原 z | 主要用于检索 |
| `value` | M×128，保留原 RGB–SAR 区域 z | 是 |
| `label_distribution` | M×4，GT 软分布，ignore 排除 | 作为 Train 支持标签 |
| `event_id/image_id/region_id` | 可追踪的稳定来源 | 默认不作为语义输入 |
| `scene_group_id` | 有真实场景／空间分组资料时记录 | 去重、排除重叠来源；没有则不编造 |
| `labelled_mass/route_mass` | 每条区域的有效标签质量与总路由质量 | 筛选与分层审计 |
| `gt_purity/gt_foreground_fraction` | max(label_distribution)、1−p(background) | Train 库质量审计；不要求查询 GT |
| `predicted_group/B_mean/B_uncertainty` | 预测分组及 B 的汇总统计 | 起步只审计，不硬筛掉 B 漏检区域 |
| `attention_effective_samples` | 注意力有效支持规模 | 分析 Token 是否由极少位置主导 |
| `geometry` | 图像与特征尺寸、质心、bbox、面积、坐标单位 | 起步仅定位、分层与去重 |
| `extraction_context` | 完整图或裁剪、偏移、缩放、翻转等 | 保证可复算与上下文一致 |

质量统计必须注明权重定义。几何标签分布沿用 Q；可另存注意力加权标签分布作审计，不能悄悄改变支持标签的含义。若区域归一化汇总权重为 w_i，注意力有效支持数可记录为 1/sum(w_i²)。这是支持位置数的加权代理，不是真实独立样本数。

一个区域分布为 `[0.1, 0.2, 0.7, 0.0]`，表示它汇总的有效 GT 标签中相应比例为 Damaged，不能解释为一栋建筑有 70% 面积受损。即使当前多数类是 Damaged，也应保留这四个数，而非存成强制 one-hot。

整库还应保存前端、Q/B、Embedding、原区域头、key 投影的权重哈希，Train 清单哈希，特征层名称、归一化方式、pool/read 路由、阈值、尺寸、数据类型、类别映射、ignore 值、采样种子和配额政策。当前已保存部分身份校验信息，扩展时应沿用它，而非绕过兼容性检查。

起步不建议把灾害类型、地理坐标、B 置信度全部拼进分类器。它们可能帮助，也可能成为事件类别先验的捷径；先作为审计字段，每增加一种分类输入都要单独验证，并确认真实部署时可获得。

空间成本方面，3480×128 个 float32 特征约 1.70 MiB；若独立保存 128 维 key、128 维 value 和四维标签，张量约 3.45 MiB，未计元数据和文件封装。无须为每条记录保存完整图像、密集特征、Q 或重复保存约 761 MiB 的前端 checkpoint。原图通过来源路径关联，前端保留单份及哈希。

固定 Embedding 时库不会因为修正头更新自动过期。只改变检索投影，需要同步重新投影库 keys 和 query；若改变 Q、B、Embedding 或输入预处理，必须重新提取对应内容并更换库版本。训练中不断更新前端则另需研究动量编码或刷新策略，不属于当前最小实验。

## 9 下一步只做能区分原因的实验

### 9.1 先完成现有诊断

保持原协议，不重跑 `all`。读取最终 `summary.json`、`retrieval_quality.json`、`audit_final.json` 和三种推理扰动。

优先判断：

1. Damaged 查询的邻居 Damaged 软质量，比同样排除规则下的库先验高多少？按类、按事件分别看。
2. 在原 hard 预测错误的查询上，邻居是否提供更多正确等级证据？总体近邻准确不等于有互补价值。
3. 正常检索是否优于随机邻居、标签打乱、Memory off？依赖 Memory 不等于胜过独立训练的 nomemory。
4. 新收益是否以更多背景误报、Intact/Destroyed 退化或部分事件显著下降为代价？使用混淆与配对转移，不能仅看三个 IoU 的升降就推断转移方向。

### 9.2 第一个低成本对照是直接读取标签

固定 bank、z、Q、B、合法性排除与 Top-8。对已有支持分布 d_j 和余弦分数 s_j，计算：

```text
a_j = softmax(s_j / temperature)       # 只在合法邻居中归一化
p_retrieval = sum_j a_j * d_j
p_final = (1 - lambda) * p_base + lambda * p_retrieval
```

这是 kNN 标签读取的任务适配，不是 Tip-Adapter 原公式的严格复制。`p_base` 与 `p_retrieval` 均为区域概率，再沿原 Q 回投；不能直接把概率与 logits 相加。无合法邻居时返回 p_base。温度与 lambda 只在预先限定的小范围 Val 比较，必须包含 lambda=0，不用 Test。若这条低复杂度读取也没有互补信号，不能仅靠增加 attention 层数解释问题。

### 9.3 若近邻语义不足 再只学习检索键

优先保留当前 z 作为 value，引入同一小投影 P 作用于查询与库：

```text
key_query = normalize(P(stop_gradient(z_query)))
key_bank  = normalize(P(stop_gradient(z_bank)))
value_bank = stop_gradient(z_bank)
```

可用同维线性 P、单位矩阵初始化，使初始检索尽量复现原结果；并非直接替换成新主干。用 Train 查询标签监督检索空间，可采用软标签邻域预测目标或经过明确定义的对比目标，二者择一。库／查询都需排除同图和预先约定的同事件候选；某查询没有合适正支持时不能伪造正例。

若用对比监督，混合区域不能仅凭 argmax 一律作为强同类正对；需定义质量门槛或软标签相容度，并报告少数类被排除多少。可微训练时，对全部合法库键计算分块分数与损失；不要以为离散 Top-K 索引能够自动反传并学会选择未入选的样本。冻结的 z 不接收梯度，但 P 的查询与库两侧可按同一目标更新。

分两阶段验证：先训练并锁定 P、重算全部 keys，再从相同初始化训练原规格修正头。和原 key 检索保持同一组 bank ID、容量、Top-K 和读出；增加使用同容量投影但不读外部标签的 query-only 对照，并单独报告投影训练成本，避免把新增容量和监督误称为检索收益。这是新假设，尚未在 BRIGHT 验证。

### 9.4 若证据指向库质量 则改采样而不是同时改 key

保留总容量与 key，比较原事件×多数类随机保留和质量分层、来源去重的采样。先在相同总预算内覆盖更多独立图像／对象代理，并保留不同纯度和尺度层次。不能简单只留“高置信度、面积大、纯度高”区域，否则容易删去小建筑、边界和少数难例；预测置信度低也不等于真实 GT 不可靠。

如果要验证类别平衡的训练采样或新损失，应给 memory 和 nomemory 两路同时采用相同策略。这样才能区分长尾重平衡收益与检索增量。训练期 ContrastiveSeg／PaCo 路线另立实验，不与本轮 key 或采样修改合并。

## 10 评价与停止条件

主要比较仍是完整 Val 前景 mIoU 的同预算、同一步差值。联合报告 Damaged P/R/F1，所有类别 IoU、配对错误转移、ROI 漏检和误检位置、边界两侧、小对象代理、每事件、耗时和显存。近邻语义质量按类别宏平均和有效支持质量分层，不用全图背景主导的准确率掩盖前景。

对区域检索，补充每类候选数、保留数、实际独立图像数、事件覆盖、合法排除后可用数量、邻居标签分布及原模型易错查询的互补率。近邻标签一致率不是前景 mIoU 上限。

已有单次训练多个验证点不是独立重复。出现明确正向信号后再重复训练种子与建库种子；验证集不确定性按图像／场景或事件结构做配对汇总，不把像素当独立样本，且不与种子方差混称。

若 key 的语义近邻指标改善但完整分类没有提升，说明检索改善未转化为有效利用；先停止扩展 key。若修正的错误与新增伤害相抵，不能通过不断增加残差强度掩盖。若简化标签读取、key 适配或质量采样都无稳定增量，保留 nomemory，是得到可解释研究结论，不是必须继续增加模块。

## 11 结论

Memory 的本质是存储与读取历史证据；存多少不是机制本身。长尾改善需要落实到类别供给、表示可分性、查询训练贡献或决策校准中的具体环节。当前工程已保存 Train 区域特征和软标签，主要未知点是这些特征能否检索到对原模型错误有互补性的支持。

最合适的顺序是：读取最终近邻诊断，建立简单标签读取对照，再由证据选择“独立检索键”或“固定预算的库采样”之一。存储采用 key、value、软标签、来源、质量和版本的分工。已有机制的迁移不等于原创，更不能据此承诺达到 SRAC 公告指标。

## 参考文献

1. U. Khandelwal, O. Levy, D. Jurafsky, et al. “Generalization through Memorization: Nearest Neighbor Language Models.” ICLR, 2020.
2. R. Zhang, W. Zhang, R. Fang, et al. “Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification.” ECCV, 2022.
3. X. Qi, Q. Chen, J. Jia, V. Koltun. “Semi-Parametric Image Synthesis.” CVPR, 2018.
4. Z. Liu, Z. Miao, X. Zhan, et al. “Large-Scale Long-Tailed Recognition in an Open World.” CVPR, 2019.
5. J. Cui, Z. Zhong, S. Liu, et al. “Parametric Contrastive Learning.” ICCV, 2021.
6. J. Zhu, Z. Wang, J. Chen, et al. “Balanced Contrastive Learning for Long-Tailed Visual Recognition.” CVPR, 2022.
7. B. Kang, S. Xie, M. Rohrbach, et al. “Decoupling Representation and Classifier for Long-Tailed Recognition.” ICLR, 2020.
8. W. Wang, T. Zhou, F. Yu, et al. “Exploring Cross-Image Pixel Contrast for Semantic Segmentation.” ICCV, 2021.
9. Y. Wang, H. Wang, Y. Shen, et al. “Semi-Supervised Semantic Segmentation Using Unreliable Pseudo-Labels.” CVPR, 2022.
10. S. Liu, S. Zhi, E. Johns, A. J. Davison. “Bootstrapping Semantic Segmentation with Regional Contrast.” ICLR, 2022.
