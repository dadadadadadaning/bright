# SRAC 区域、建筑标签与 Memory：混合区如何建库、检索和回投

日期：2026-10-10。问题是：BRIGHT 的损毁等级按建筑标注，而本地 SRAC 风格实现的查询与建库单位是软超像素 Q。当一个 Q 区域覆盖背景和多栋不同损毁等级建筑时，现有标签、检索、回投各表示什么，以及哪些方案有文献与代码依据？本报告核对了本地实现、BRIGHT 原始论文和相关作者论文/代码。**没有重新跑 Train、450 张 Val 或 Test，因此不能把以下可能机制写成已证实的误差归因。**

## 1. 四种单位不能混为一谈

| 单位 | 本任务中的意义 | 当前 SRAC 是否保留其 ID |
| --- | --- | --- |
| 真实建筑 | 原始多边形有各自的 Intact / Damaged / Destroyed 属性 | 当前语义数据加载及 Memory 缓存不保留建筑实例 ID |
| 标注像素 | 建筑属性栅格化后得到四类：Background、Intact、Damaged、Destroyed | 是，Train 标签用于生成区域软标签 |
| Q 区域 | 一组相邻特征位置的**软归属**，可跨建筑、跨类别；一栋楼也可覆盖多个 Q | 是，当前 key/value 汇总及回投单位 |
| 最终像素 | 最终 IoU 与混淆矩阵的统计单位 | 是，经 Q 回投和像素分类输出 |

[BRIGHT 原始论文](https://essd.copernicus.org/articles/17/6217/2025/essd-17-6217-2025.html)说明原标注由建筑多边形和逐建筑损毁属性组成。当前[语义加载器](/D:/software/dowmload/BRIGHT-master/bda_benchmark/dataset/make_data_loader.py:83)读取四类 `*_building_damage.tif`，另派生二值建筑定位标签，未把多边形实例编号传入 SRAC。当前[Q 生成](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_transformer_v1/model.py:79)由灾前结构特征、空间位置及**预测**建筑概率共同决定，训练标签不参与前向分区；其 Q 是每个特征网格位置对附近至多九个种子的权重，不是 SLIC 的硬实例分割，也没有显式“同一 Q 仅含同一栋楼”的约束。[前端训练](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_transformer_v1/train_front.py:185)主要使用回投后的像素分割损失，没有单独的实例 ID、建筑边界或 Q 类别纯度损失。

因此，“加了区域约束”与“已对齐建筑实例”是两回事。用户先前给出的固定对照中，`grid_probq` 前景 mIoU 61.8381%，高于 `pixel_grid` 的 61.4194%（+0.4187 个百分点），支持**Q 的图内概率共享有用**。但这不证明 Q 按栋切开，也不保证带标签的跨事件 Memory 有用。`region` 61.4220% 还混合了区域 token、分类头和训练差别，不能单独据此归咎于 Q。

## 2. 当前一个 Q 区域含四类时，程序究竟做了什么

设特征位置为 `i`，区域为 `r`，四类真值为 `GT_i`，软归属 `Q_ir` 且每个位置对其候选区域的权重和为 1。Train 阶段当前[区域标签函数](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_transformer_v1/common.py:160)先把像素 one-hot 以面积法下采样，再按 Q 汇总，近似表示为：

\[
m_{r,c}=\sum_i Q_{ir}\,\mathbf 1[GT_i=c],\qquad
y_{r,c}=\frac{m_{r,c}}{\sum_{c'=0}^{3}m_{r,c'}}.
\]

例如区域的有效覆盖为 20% 背景、40% Intact、25% Damaged、15% Destroyed，则 `y_r=[.20,.40,.25,.15]`。它是**四类像素组成**，不是“一栋楼分别损伤 25%、毁坏 15%”。这条记录没有办法仅凭 `y_r` 判断区域里有几栋楼、哪一部分属于哪栋。即使 `y_r=[0,1,0,0]`，也可能是两栋相邻的完整建筑，所以**类别纯度与建筑实例纯度不同**。

当前[建库代码](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_transformer_v1/build_memory.py:90)存区域 token、完整四维 `y_r`、质量及 `(事件, 图像, 区域)`等元数据。`argmax(y_r)` 用于“事件×类别”配额分桶，**没有把四维软值覆盖成硬标签**；默认 `min_purity=0`，有效混合区域可以入库。这不是数据平衡机制能解决的建筑身份问题。提高纯度门槛可能使混合支持减少，同时误删边界、小建筑、罕见 Damaged；要按类、按事件报告剩余条目和查询命中率。旧 J3 [投影训练](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_qmetric_v1/train_metric.py:54)可只取高纯 bank 样本，而推理仍在完整 bank 上检索，亦需区分。

还有一个需要实测的**key/value 对齐风险**：区域标签按原 Q 统计，区域 128 维 token 的主分支却乘可学习空间权重 `sigmoid(score(adapted))` 后汇总，并添加区域均值/方差与几何信息。混合区域的 token 可能偏重某个损坏角落，软标签却按整块面积以 Intact 为主；反向也可能。代码允许这种机制，但目前没有证据证明它就是现有误报的原因。[token 构造](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_transformer_v1/model.py:271)。

## 3. 当前“谁检索谁”，以及为什么检索无法替区域认领建筑

**查询**是 Val/Test 当前图的整块 Q 区域 token `T_r`，不带真值；**key** 是 Train bank 各条历史区域 token；**value** 是对应的 Train 四类软标签及元数据。J2 使用原 token 余弦，J3 使用投影后余弦；二者排同事件、同图像后取 Top-8。Top-8 不保证邻居足够相似、不保证来自不同事件，也不代表每条邻居恰好对应查询区的一栋楼。[选择与排除](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_qmetric_v1/metric.py:535)。

例如查询区同时跨一栋 Intact 和一栋 Damaged，当前系统只发送**一个混合 token**去检索。某个 Damaged 邻居可以对整个 token 有用，却没有任何字段说明“只给右边的损坏楼使用”。靠把邻居标签取多数、加深注意力层或增加 Top-K，都不能从一个无空间分解的向量恢复楼内边界。若将 `y_r` 强行改成主类作检索监督，还会把被覆盖的少数类当错误标签。若查询端按当前预测单一类别预先限定支持类，模型原本漏掉的 Damaged 可能再也检不到 Damaged；可并行收集各类候选，再根据绝对相似度与证据质量决定采纳。

新版[证据读出](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_hardquery_memory_v1/reader.py:23)计算 `posterior=Σ_j softmax(s_rj/τ)y_j`，并把绝对 Top-1 相似度、间隔、标签熵、支持纯度、事件多样性送入质量向量。只要非空支持存在，softmax 权重总和仍为 1，即使所有支持都差；所以**归一化标签分布不是可靠度**，必须另外学习或约束拒绝。该分支代码具备门控与有界修正，但此前正式报告只证实 E0/E1，不应把新训练收益当成已测结果。

## 4. 现在的回投怎么算，提出的新回投该怎么做

当前直接区域头先得到 `p_r=softmax(z_r)`，然后按每个像素自己的 Q 权重回投：

\[
p_i^{\mathrm{direct}}=\sum_{r\in\mathcal N(i)}Q_{ir}p_r.
\]

之后双线性插值到原图并归一化。[直接区域回投](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_transformer_v1/model.py:303)。Q 是软且重叠的，一个像素一般没有唯一“所属区域”；如果几乎全部权重压在一个跨两栋楼的 Q，而 `p_r` 是区域常数，直接回投就难以给同一区域内两栋楼不同类别。

旧 J1/J2/J3 并非把 `p_r` 直接涂满 Q：Reader 得增强区域 token，按当前 Q 回投为像素上下文，再与每个位置的 A1 像素特征 `F_i` 合并分类。所以同一 Q 内像素仍**可能**有不同输出；但检索证据的原始空间定位仍较粗。[Reader](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_joint_context_v1/model.py:94)、[像素融合](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_joint_context_v1/model.py:215)。新的 hardquery 分支则把有门控的区域 logit 修正按 Q 回投，加到固定 J1 上。[修正头](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_hardquery_memory_v1/reader.py:165)。

要检验“Memory 只在可靠时改判”，可用**固定、已验收的最终像素概率** `P0_i`（例如已显示正向效应的 Q 平滑基线）和下面的概率融合做全新对照。令每个区域有证据分布 `ρ_r` 与门控 `g_r∈[0,1]`，则

\[
A_i=\sum_r Q_{ir}g_r,\qquad
P_i^{\mathrm{out}}=(1-A_i)P_i^0+\sum_r Q_{ir}g_r\rho_r.
\]

当所有 `g_r=0`，所有像素严格回到固定基线；`Σ_c P_i^{out}(c)=1`。需在**最终输出坐标系**对齐/插值权重后融合，避免将已 Q 平滑的 `P0` 再池化回投一次。这个公式是待做的对照，不是当前 J1/J2/J3 的实现，也不能直接搬用旧门控权重。若 Q 过粗，`g_r` 仍会让证据扩散，故可进一步在像素处用 `g_{ir}` 或建筑掩膜限制。边界/低置信区域拒绝，保留 `P0_i`。

## 5. 怎么“平衡”混合区域：从小改到按栋建库

**A. 先诊断并保留 Q。** 在 Train/Val 的只读分析中记录每个 `r` 的四类质量 `m_{r,c}`、`max(y_r)`、熵、有效面积、前景比例 `b_r=1-y_{r,0}`，并在 `b_r>0` 时记录建筑内条件分布 `d_r(c)=y_{r,c}/b_r`（`c=1,2,3`）。`[.8 BG,.2 Damaged]` 与 `[0 BG,.5 Intact,.5 Damaged]` 的主类/熵性质不同；只看四类 `max(y)` 会漏掉“少量纯 Damaged 被背景淹没”。再按这些量分层统计 Top-8 的真实标签相容性、Intact→Damaged 误报与 Damaged 救回。GT 仅在 Train 用于标签和训练，在 Val 用于**只读诊断**，在 Test 推理中不能出现。

**B. 第一个可控改版：纯支持、混合查询、可拒绝回投。** 在相同 Q、前端、bank 容量、事件配额和固定像素基线下，优先让前景高质量、高类纯支持进入主库；背景、前景与背景混合、多损毁等级混合条目分开统计或作辅助软证据，不能一律按主类硬贴。查询要包括所有 Q，尤其 Train 内漏判 Damaged 和“原本预测正确却可能被 Memory 推向 Damaged”的 Intact；不能用 Val 错例训练。用跨事件/跨图像排除，记录绝对相似度、邻居标签冲突、事件多样性与正确支持覆盖；非空但坏的邻居应能触发 `g=0`。这个版本可先查 Memory 是否主要受支持污染影响，但**纯支持不自动解决混合查询的建筑定位**。纯度门槛按每类召回选，避免删除小 Damaged 后产生虚假的纯净收益。

**C. 若混合查询确实构成瓶颈，改为 Q × 预测建筑的层级检索。** 在 Train、Val、Test 都用**相同的预测实例生成器**得到建筑候选掩膜 `M_{ib}`，以 `W_{i,br}=Q_{ir}M_{ib}` 提取每栋建筑内的局部子区域 key；不要直接复用已经混合了别栋楼的整块 `T_r`。Train 端的 `GT` 只给 `(b,r)` 候选算标签和质量：`y_{br,c}∝Σ_i W_{i,br}1[GT_i=c]`；背景覆盖单独记录，建筑损毁等级则只在前景 `c=1,2,3` 条件化。可将同一预测建筑的多个子区域聚合成建筑 key，也可保留多子原型。查询端无 GT，必须仅从预测实例及图像特征构造同定义的 key。若 Train 用真实建筑掩膜裁 key 而 Test 用预测掩膜，训练/检索特征分布会不一致；可作为有标签上限诊断，不能当默认部署协议。已有[预测建筑连通域分组实验](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_building_context_v2/building_ops.py:19)可借鉴候选 ID 与区域归属接线，但不是当前 SRAC Memory，也不能保证贴邻的楼不被连通域并成一栋。

建筑层级检索仍需区分“有没有建筑”与“建筑属于哪类”。先由定位概率和候选质量决定是否允许建筑 Memory，再让 `b` 或 `(b,r)` 同时搜索 Intact、Damaged、Destroyed 的跨事件支持，不能由当前预测硬选唯一类。返回的是每栋建筑/子区域的证据分布 `ρ_{br}`，而非某个 Q 区域的真值。若多条支持来自同一图或同一事件，不能误当独立证据；可比较一事件至多一条与不限制两种方案。预测实例裂成多块时可让子区域保持局部证据；预测实例把两栋楼合并时应降低门控，回退像素基线。

建筑约束回投可写为（在最终像素坐标系取对齐后的 `Q` 与 `M`）：

\[
A_i=\sum_{b,r}M_{ib}Q_{ir}g_{br}\le1,\qquad
P_i^{\mathrm{out}}=(1-A_i)P_i^0+\sum_{b,r}M_{ib}Q_{ir}g_{br}\rho_{br}.
\]

其中预测建筑掩膜需互斥或归一化使 `Σ_b M_{ib}≤1`，`g_{br}` 含预测掩膜质量和检索可靠度。`A_i=0` 时严格保留 `P0_i`；背景和边界不必强涂损毁标签。若某栋楼的边界预测可信且有足够证据，可以再比较把其多个子区域汇成单一损毁等级的建筑级输出；若还没有可信实例分割，就**不能宣称已实现按栋 Memory**。BRIGHT 官方另有[CVPRW26 实例级基线](https://github.com/ChenHongruixuan/BRIGHT/blob/master/cvprw26/README.md)，但本地[实例审计说明](/D:/software/dowmload/BRIGHT-master/bda_benchmark/docs/H9_B0_BUILDING_ORACLE.md:20)指出其公开 697 张实例验证与此处标准 450 张语义验证划分不同；不能未经对齐把实例标注当标准 Val450 的隐藏真值。

## 6. 论文与作者代码：哪些机制真能迁移

| 工作 | 论文和作者代码的实际做法 | 对当前问题的可迁移部分与限制 |
| --- | --- | --- |
| **Gould 等，ECCV 2014，Superpixel Graph Label Transfer**：[论文](https://users.cecs.anu.edu.au/~sgould/papers/eccv14-spgraph.pdf)、[Darwin 官方项目与命令](https://users.cecs.anu.edu.au/~sgould/darwin/drwnProjNNGraph.html) | 多尺度、重叠超像素；训练区存像素标注的经验类别分布，度量学习时有时取主类；测试区找训练图邻居；每个像素汇总覆盖它的区域及其检索邻居的软标签；同一个查询尽量从不同**图像**取支持。**这是最直接的“混合超像素＋Memory 检索＋像素回投”先例。** 但它的目标是一般语义类别，不是建筑实例；多尺度可缓解单尺度混合，不能从单一混合 token 还原内部楼的身份。 |
| **Kim 等，ICCV 2023，Adaptive Superpixel for Active Learning**：[论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Kim_Adaptive_Superpixel_for_Active_Learning_in_Semantic_Segmentation_ICCV_2023_paper.pdf)、[作者代码](https://github.com/ml-postech/adaptive-superpixel-for-active-learning-in-semantic-segmentation) | 给超像素主类会因边界跨类而产生噪声，作者用自适应超像素和 sieving 排除高风险像素。支持“不把混合区主类当真值、要筛掉风险证据”的原则；其方法用于主动学习标注，**不是**现成的跨事件检索库。 |
| **Li 等，CVPR 2021，ASGNet**：[论文](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Adaptive_Prototype_Learning_and_Allocation_for_Few-Shot_Segmentation_CVPR_2021_paper.pdf)、[作者代码](https://github.com/Reagan1311/ASGNet/blob/main/model/asgnet.py#L198-L250) | 在有 GT 前景掩膜的 support 内提多个局部原型，query 做位置匹配分配。支持“历史条目先裁可信前景、多原型配局部查询”，但它是 few-shot 即时 support，**没有**当前跨事件持久 bank；query 端绝不可用 GT。 |
| **Jampani 等，ECCV 2018，SSN**：[论文](https://faculty.ucmerced.edu/mhyang/papers/eccv2018_ssn.pdf)、[作者代码](https://github.com/NVlabs/ssn_superpixels) | 可微软分区，可借助 Train 语义标签重建和紧致损失使区域更贴语义边界。仅在证据显示 Q 是瓶颈后测试此类训练约束；“贴语义类别”仍不等于“按建筑实例”。 |
| **Barsellotti 等，CVPR 2024，FreeDA**：[论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Barsellotti_Training-Free_Open-Vocabulary_Segmentation_with_Offline_Diffusion-Augmented_Prototype_Generation_CVPR_2024_paper.pdf)、[作者代码](https://github.com/aimagelab/freeda) | 离线视觉原型库与测试图超像素局部匹配，加全局语义。支持“局部与全局证据联合”思路；库不是逐 Q 的 Train GT 软标签 Top-K。 |
| **Zheng 等，ACL 2021，Adaptive kNN-MT**：[论文](https://aclanthology.org/2021.acl-short.47/)、[作者代码](https://github.com/zhengxxn/adaptive-knn-mt) | 根据邻居质量动态确定检索证据权重及邻居数。可借鉴“找到邻居不等于应采纳”，但翻译任务的概率混合不直接解决建筑边界或严格拒绝。 |

上述文献不能证明当前误报由 Q 造成；它们分别支持处理混合标注、局部多原型、可学习边界、邻居筛选等**可检验机制**。尤其 Gould 等论文的公式与当前 `Q` 回投相似，但作者使用多个尺度的重叠区域；当前实现只有固定局部候选软归属，不能把二者称为完全相同方法。

## 7. 最小实验矩阵与判别逻辑

1. **不换模型先量化问题。** 固定已复现的数值协议及 Train/Val 列表，统计 Q 区域四类组成、前景条件纯度、边界混合、区域有效质量及逐事件分布。若拿到与样本严格对齐的实例 ID，再统计每个 Q 覆盖几栋楼、一栋楼跨几个 Q；只有四类 TIF 时不能伪造这些建筑级真值。做 `Q` 软标签回投的只读 *oracle 诊断* 可估计区域标签平均的边界损失，但它不是所有像素模型的理论性能上界。
2. **固定 Q/F/基线/容量，只改支持入库规则。** 比较原 bank 与等容量的“可靠前景支持＋混合软支持”库；报每类留存数、跨事件可用量、命中标签相容性、Top-8 重复度。若干净支持显著改善检索而混合查询仍坏，才转入查询拆分。
3. **固定同一 bank，只改检索与拒绝。** 比较原 Top-8 与多类别候选/不同事件去重、绝对相似度质量门控；训练查询从 Train 产生（最好训练折外预测，减少困难样本挑选偏差），包含漏判 Damaged 与本来判对但容易被 Memory 误改的 Intact。报告“拒绝率—Damaged 救回—Intact 误报”曲线，而不是只报救回数。
4. **有证据后才做建筑层级。** 若混合区是主要错误来源，比较原 Q 区域 key、`Q×预测前景` key、`Q×预测建筑实例` key；同一版本 Train bank 与 Val/Test query **用同源掩膜算法**，重提 key 与 value，不能混用旧 bank 或旧 metric 权重。增加实例定位质量和建筑级评价；低质量实例回退固定像素基线。
5. **最终选择看共同固定基线上的像素四类 IoU、前景 mIoU、逐事件、Damaged 救回、原本正确 Intact 的伤害、Intact→Damaged 新增假阳性。** 有真实实例标注且划分对齐时再报逐栋指标。Val 用于方案选择；正式 Test 留给冻结方案，不使用 Test 标签建库、调门控或挑实验。

现有 E1 的标签直读造成的大量 Intact→Damaged 是**像素级观测**，并不能直接分摊成“因混合 Q 造成多少”与“因检索/Reader 造成多少”。本次也未测得当前 Q 中四类混合区域的频率。最先可证伪的命题是：在同一 Q 与固定基线下，清理不可靠支持或拒绝坏检索是否能减少 Intact 误报而保留 Damaged 救回；只有这一步不足以解释问题时，才投入建筑实例级 Memory。
