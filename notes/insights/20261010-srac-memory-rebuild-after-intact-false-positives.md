# SRAC Memory：从 Intact 误报诊断到建库、检索和使用方案

分析日期：2026-10-10。研究方向：BRIGHT 灾前光学与灾后 SAR 建筑损毁分割。研究问题：历史参考怎样帮助漏判的 Damaged，同时避免把真实 Intact 改成 Damaged。

本报告采用 research-paper-insight 工作流，核对本地实现、原始论文和作者代码。450 张 E0/E1 的状态及统计来自用户粘贴的服务器报告；本次没有连接服务器复算，也没有训练、修改实验源码或重建实际 bank。以下方案为待验证设计。

## 1. 已确认的问题：参考有帮助，也会误导

用户提供的正式运行目录为 /root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130。E0/E1 complete、full_validation_evidence、reproduction.passed 均为 True，450 张；E1 的六项 immutable 均为 True。此前严格设置下失败的记录不因此被改写。

| 相对固定 J1 | Damaged 净增加正确像素 | 真实 Intact 导致的净新增 D 假阳性 | 全部净新增 D 假阳性 | D IoU 变化，百分点 |
|---|---:|---:|---:|---:|
| J2 | 39 | -2,248 | -582 | +0.003176 |
| J3 | 5,947 | 11,172 | 9,499 | +0.019623 |
| 余弦标签读出 | 324,385 | 1,422,342 | 1,730,261 | -3.924348 |
| 旧投影标签读出 | 374,189 | 2,698,195 | 3,167,763 | -8.125231 |

两种标签读出的净新增假阳性分别有 82.20%、85.18% 来自真实 Intact。旧投影相对余弦多出 1,437,502 个净假阳性，其中 1,275,853 个来自真实 Intact，占 88.75%；净正确 Damaged 只多出 49,804。

通俗地说，模型从参考册里找回了一些漏判答案，但照参考答案改得太多，将更多完整建筑说成损毁。不能据此推断库中人工标注有错、Damaged 数量过多，或某个具体模块已经被证明失效。

统计按真实类别划分。真实 Intact 的新增 D 假阳性可能来自 J1 原预测 Background、Intact、Destroyed；只有转移矩阵 t[1][1][2] 才是“J1 原本正确 Intact 被改成 D”。应同步报告它除以 sum(t[1][1]) 的比例及逐事件结果。像素总量不能代替条件错误率，也不能把像素当成独立重复实验。

旧 J3 的 mIoU 为 61.657357%，仍低于 A1 的 61.686685%。旧 Reader 输出变化较小，可能抑制了误导证据，也可能遗漏有效证据，现有统计尚不能区分。标签读出不是 Reader 的注意力或性能上界。本次 E1 未验证新 hard-query 投影与六组修正头。

## 2. 当前代码中的 Memory 究竟是什么

Memory 是 Train 历史区域的“带答案参考册”，不是保存 450 张 Val 图像，也不是直接保存整栋建筑的最终预测。

| 组成 | 当前作用 | 关键代码 |
|---|---|---|
| 历史区域缓存 | 冻结前端与 Q，保存区域特征、软标签和来源 | [snapshot 建库](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_snapshot_joint123_v1/build_memory.py:95) |
| 检索器 | query 与 bank 使用相同投影，跨事件、跨图像取 Top-K | [select_ids](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_qmetric_v1/metric.py:535) |
| 旧 Reader | 支持特征、标签与元数据形成 value，交叉注意力读取 | [ContextMemoryReader](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_joint_context_v1/model.py:115) |
| E1 标签诊断 | 同 Top-8 的软标签按相似度加权，经 Q 回投影 | [retrieve_evidence](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_hardquery_memory_v1/reader.py:80) |
| 新版修正器 | 将标签证据及质量用于固定 J1 的有界修正 | [GatedMemoryCorrection](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_hardquery_memory_v1/reader.py:165) |

每条原始支持包含 128 维 tokens、四类 distribution、events、images、regions、geometry、disasters。软标签描述区域内标注类别的组成。例如 70% Intact、30% Damaged 表示区域混合，不表示某栋建筑损毁了 30%。

建库已有事件×主类别配额。源码默认每格最多 64、min_purity=0.0；实际运行参数应以服务器 manifest 为准。旧投影训练才筛选纯度不低于 0.8 的子集，不能说整个推理 bank 都经过 0.8 过滤。

纯度衡量类别混合度，不等于人工标注准确性、视觉难度或跨事件匹配可靠度。J1 判错的 Train 区域也可能有可靠标注和很高参考价值，不能因为模型判错就删掉。

## 3. 为什么“找到八个相似参考”仍可能误判

Top-8 只表示现有候选中排名靠前，不保证相似度足够高。当前标签读出在有效邻居间使用 softmax，权重总和为 1；相似度整体下降而差距不变，权重不变。

例如两组相似度 [0.90, 0.89] 与 [0.20, 0.19]，在相同温度下得到完全相同的归一化权重。这是数学示例，不是本次实测相似度。若弱匹配的邻居恰好都偏向 Damaged，标签意见也会很一致。意见一致不等于匹配可靠。

该局限存在于 E1 直接标签读出；新版修正头已另输入最高绝对相似度等质量信息，并非整个现有系统只看 softmax。旧投影为何变差，还需区分邻居集合、分数尺度、标签混合与 Q 回投影，不能只凭假阳性汇总归因。

最低成本新诊断是固定特征、库和 Q，比较余弦/旧投影的 Top-8 各自使用均匀权重与当前相似度权重。跨检索器看候选集合差异，同一检索器内看权重作用。更改诊断实现需独立版本，保留当前已通过 E0 的源码。

## 4. 重建方案：一份推理参考库，一份训练查询集合

两份数据的用途不同。参考库回答“有什么真实案例可借鉴”；训练查询集合回答“查找器和使用者最该练习哪些情形”。困难查询不能简单全部塞进推理 bank。

### 4.1 推理参考库

1. 继续只用 Train。冻结当前前端与 Q，先复用已有完整 Train 区域缓存，保留原 bank。
2. 每条支持保留原始特征、真实软标签、事件/图像/区域来源；附区域有效质量、混合程度和选择原因。事件 ID 优先用于排除、覆盖分析和来源追踪，是否输入分类器另做对照。
3. 初次对照保持总容量和事件×类别实际配额。比较原随机抽样与同容量多样性选样，先选真实代表样本，保留可追溯标签；不把所有类内样本平均成单个模板。
4. 特别检查是否覆盖“标签明确、外观容易与 Damaged 混淆的 Intact”，同时覆盖轻微损毁等多种 Damaged 外观。高纯度不等于外观简单，二者需分开看。
5. 混合区域可单列质量层，仍保存软标签。若测试高纯度过滤，应独立于多样性选样；不能只提高纯度、丢掉困难和小区域后就宣称建库质量更好。某事件某类不足预算时明确记录，在可比容量下做实验。
6. 不因 Intact 误报多就无条件增加 Intact 库容量或降低 Damaged 配额。当前只有误报结果，没有证据证明类别数量失衡是原因。

只修改检索投影时，可从已有 tokens 重新计算检索 keys；query 与 bank 必须使用同一个新投影，旧投影建立的索引必须更新。修改前端、Q 或区域特征定义时，才需要同步重提取查询与库特征。不能使用新 query 表示去查旧空间的索引。

### 4.2 训练查询集合

从完整 Train 记录以下三种查询及其真实软标签、固定 J1 判断、检索结果和支持来源：

- 需要救回：真实 Damaged，但 J1 漏判。
- 容易被参考误导：真实 Intact，J1 正确，但 Memory 倾向 Damaged。先用像素转移与区域统计量化覆盖；混合区域不强行用单类定义。
- 普通查询：保持四类与事件覆盖，防止模型只会处理被挑出的错误。

当前 [query_hardness](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_hardquery_memory_v1/metric_learning.py:33) 依据真实软标签与 J1 概率的不一致程度，没有专门突出第二种。普通采样仍可能包含第二种，不能说完全没有覆盖。

训练检索器时，真实 Damaged 查询应接近合适的跨事件 Damaged 支持；真实 Intact 查询应能区分外观近似的 Damaged 支持。可靠纯区域可以定义明确正负例，混合区域依据软标签相容性处理。跨事件、跨图像限制继续保持，推理检索不接收查询真实标签。

这些风险样本必须在 Train 中产生；Val 的错误只用于分析方法，不可直接加入训练或建库。固定 J1 已见过 Train，因此这些误差不是全模型的独立折外误差估计。

## 5. 使用方案：先保留原判断，再决定如何参考

推理流程为：当前图像提取区域 → 固定 J1 给原判断 → 跨事件检索 Top-8 → 汇总软标签和质量 → 小修正器判断是否采纳 → 经同一个 Q 回投影到像素。

初版继续固定 Top-8，先避免同时改变太多因素。参考质量可以包含绝对相似度、标签冲突、支持混合程度、事件多样性以及与 J1 的分歧；这些都可以在推理时得到。标签一致但绝对相似度很低，也应作为风险情况学习。

输出保留现有形式：固定 J1 logits + Q 回投影后的门控有界修正。关闭全部新增修正时严格返回 J1。单个区域关闭不保证所有重叠像素不变，因为还有其他区域贡献。当前 sigmoid gate 主要是连续减弱，不能把它描述为已经验证有效的硬拒绝器。

新增重点是“有参考，但参考有害”的训练，而非再次增加普通 gate。用 Train 中实际检索得到的冲突证据，监督哪些修改降低分割损失、哪些损害原本正确的预测；查询软标签/像素标注只参与训练监督。空支持 dropout 只模拟参考缺失，不等于拒绝错误参考。

可对原本正确 Intact 的不必要 D 修正加入适度保护，但必须同时训练 Damaged 救回。保护过强会导致永远关闭 Memory。阈值与权重应在 Train 内预留部分或遵循明确的验证选择协议确定，不能逐张根据 Val 真值开关。

## 6. 经核验的论文与作者代码：借什么，不照搬什么

| 论文与收录 | 可迁移模块及作者代码 | 适配、难度与限制 |
|---|---|---|
| ReCo，ICLR 2022：[原文](https://arxiv.org/abs/2104.04465) | [compute_reco_loss](https://github.com/lorenmt/reco/blob/main/module_list.py#L65)：低真实类置信度查询，按类原型相似度采混淆负类 | 分割任务匹配较高，迁移难度中；原法不是本项目的跨事件推理库。加入“J1 正确但 Memory 误导”查询是本报告提出的适配 |
| Tip-Adapter，ECCV 2022：[原文](https://arxiv.org/abs/2207.09519) | [build_cache_model](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/utils.py#L38) 保存特征和标签；[run_tip_adapter](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/main.py#L27) 把缓存证据加入基线 | 读出机制匹配，CLIP 小样本图像分类与本任务不同；诊断迁移难度低。它不是本地 softmax Top-8 投票 |
| Adaptive kNN-MT，ACL 2021：[原文](https://aclanthology.org/2021.acl-short.47/) | [Meta-k 网络](https://github.com/zhengxxn/adaptive-knn-mt/blob/main/fairseq/models/transformer.py#L779) 根据距离和标签多样性，混合不同 k 与原模型 | 可借选择性融合，迁移难度中；原任务机器翻译，词标签特征需换成区域软标签质量。包含 k=0 是软混合，不保证精确硬拒绝 |
| ProtoSeg，CVPR 2022：[原文](https://arxiv.org/abs/2203.15102) | [prototype_learning](https://github.com/tfzhou/ProtoSeg/blob/main/lib/models/nets/hrnet.py#L86) 和 [原型损失](https://github.com/tfzhou/ProtoSeg/blob/main/lib/loss/loss_proto.py)：每类多个代表，配合表征学习 | 分割任务匹配较高；同容量选样实验难度中，完整迁移更高。原法在线更新原型并训练特征，不等于冻结特征离线聚类必然有益 |

Tip-Adapter 作者实现使用未归一化的 exp[-beta(1-s)] 标签累加，再加到原模型 logits；直接读取整个小样本缓存，不先做 Top-K。这与本地归一化标签投票有实质差别：整体低相似度会降低原公式的缓存量，但其量还受库大小、重复与类别密度影响，仍不是校准可信度，也不是自动拒绝。若移植应保持容量并单独校准，不照搬其参数或实验选择流程。[作者读出代码](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/main.py#L27)

ReCo 是困难查询与混淆负类采样的依据，当前 J3 已有对比学习，不能把“加入对比学习”重复作为新贡献。ProtoSeg 的正确像素更新策略也不应被解释为“只允许 J1 判对的样本入本项目 bank”。[ReCo 作者代码](https://github.com/lorenmt/reco/blob/main/module_list.py#L65)、[ProtoSeg 作者代码](https://github.com/tfzhou/ProtoSeg/blob/main/lib/models/nets/hrnet.py#L86)

近期同任务核验：IGARSS 2026 官方页面列出 SRAC: Superpixel Retrieval-Augmented Classification for Multi-Modal Disaster Building Damage Assessment。因此单纯“超像素＋历史 Memory”不是足够的新贡献。官方页面没有可下载资源，本次没有验证其完整官方实现，不能声称当前代码严格等同作者方法。[会议页面](https://2026.ieeeigarss.org/view_paper.php?PaperNum=2851&SessionID=1356)

本次为围绕已观测误报的定向检索，不是穷尽综述，也不宣称以上机制是 2026 年最新方法。论文提供可迁移依据，不提供本任务性能保证。

## 7. 最小实验顺序

| 顺序 | 操作 | 固定条件与判断依据 |
|---|---|---|
| 0 | 从已有报告拆真实 Intact 中原本正确者的误改，报告逐事件比例 | 只读报告，不重跑已通过的 E0/E1 |
| 1 | 固定邻居诊断：两种检索器各做均匀与相似度加权 | 固定前端、bank、Q，分开检查候选集合与权重 |
| 2 | 完成现有 query/bank/hard × ungated/gated 六头实验 | 同预算最终步；与 query-only、固定 J1 和 A1 比较；当前尚无新版效果结论 |
| 3 | 仅增加 Train 风险查询采样 | bank、读出、投影大小固定，检验双向混淆改善 |
| 4 | 固定有效检索分支，单独加入非空错误支持的拒绝/保护训练 | 检验减少误改是否保留救回；必要时加固定幅度缩小修正的对照 |
| 5 | 若确认覆盖或重复问题，再做同容量 bank 多样性选样 | 质量过滤与多样性分开；改变bank后匹配训练，而非直接换库宣称完整方法增益 |

所有候选保留四类指标，重点记录 Damaged 救回、原本正确 Damaged 的伤害、真实 Intact 的 D 假阳性、原本正确 Intact 被改成 D、Background/Destroyed 误报与逐事件结果。

固定真值时，Damaged IoU 改善要求净 TP 增量大于 J1 的 Damaged IoU 乘净 FP 增量。救回多不等于 IoU 高。小幅收益进一步检查训练种子与事件稳定性；1e-5 是复现门槛，不是统计显著性门槛。

旧代码、bank、报告保持不变。新建库/新训练机制另立版本、记录所用缓存、权重、源码和配置；不覆盖当前已通过的正式运行，也不放宽复现门槛。

## 8. 可执行的研究题目备选

1. 面向跨事件建筑损毁分割的误改感知区域检索训练。借鉴 ReCo，在 BRIGHT Train 中同时采漏判 Damaged 与 Memory 易误导的 Intact，固定库、容量和头部比较采样策略。预期贡献是检验基线错误难度之外的检索风险监督；创新性需由独立消融与泛化结果支持。
2. 基于可拒绝历史证据的建筑损毁分割。借鉴显式标签缓存与自适应参考权重，固定 J1，在非空冲突参考上学习是否修正。用 query-only、无门控、连续门控及新保护机制对照，检验历史证据的净增量与误改代价。仅增加 gate 本身不是充分创新。
3. 等容量多样性区域 Memory 对建筑损毁误报的影响。借鉴类内多个代表，在原事件×类别预算内比较随机支持与真实多代表支持，评估 Intact 混淆、困难 Damaged 覆盖和逐事件效果。预期贡献是区分容量与覆盖作用；聚类选样本身创新较弱，可作为主方法消融。

本次产出是分析和设计文档，未将以上备选机制写入训练代码，未运行新训练或验证。
