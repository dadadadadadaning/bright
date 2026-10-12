# SRAC Memory 本身的改进：固定 Q 与 Reader 的诊断和消融

日期：2026-10-11。范围：历史案例的构造、区域特征与标签的对应、检索键和 Top-8 排序；不训练采纳／拒绝门控。以下数值来自已经完成的 Val450/E1 和同次前向对照，**不是**新建 Memory bank v2 的验证成绩。

## 证据先于方案

| 已有观察 | 可得结论 | 不可得结论 |
| --- | --- | --- |
| Val 前景 mIoU：A1 61.6867%，`grid_probq` 61.8382%，实际 `region` 61.4220%，J1 61.6380%，J3 61.6574% | Q 的概率共享有用；实际区域表征／输出链路以及历史 Memory 仍有改进空间 | 不能说 Q 必须重做，也不能把 J1→J3 当成 Memory 的纯因果增益 |
| J3 相对 J1：Damaged 救回约 1.61 万像素，Damaged 假阳性净增约 9500，其中真 Intact 净增约 1.12 万 | Intact↔Damaged 混淆必须成为检索评价的主要切片 | 无法仅凭像素转移断定错误出在建库、排序或 Reader |
| 直接近邻标签投票：余弦／旧投影的 Damaged 假阳性分别净增约 173 万／317 万像素；其中真 Intact 分别约占 82%／85% | “近邻的 Damaged 标签”不能直接视为可靠答案 | 这种直接投票与 J3 Reader 不同，不能当成 Reader 上界或检索器单独失效的证明 |
| 前景充足的混合 Q 中，Intact→Damaged 的错误率高于高纯 Q；但高纯 Q 的错误绝对数也多 | 应按区域纯度分层诊断 | 混合 Q 与错误相关，不证明混合是主要因果原因；Q 也不等于建筑实例 |

旧库的 `build_memory.py` 默认 `min_purity=0`，用 Train Q 区域按事件×主类随机选样，保留四维软标签。`common.py:region_targets` 用原 Q 对训练真值像素面积汇总。`model.py:SuperpixelEmbedding` 的主 token 分支却用 `Q × learned spatial_weight` 池化适配特征，另有原 Q 下的均值／方差支路。因此在混合区，检索 token 与软标签可能偏重不同位置。这是待检验的 **key–label 对齐假设**，不是已证实故障。

原 J3 的 `train_metric.py` 默认只用纯度至少 0.8 的 bank 行训练跨事件投影，推理查询则包括全部 Q 区域。已有 `hardquery_memory_v1/metric_learning.py` 改为从 Train 查询中抽困难样本，但困难度定义为 `1−GT·P_J1`，尚未专门挖掘“J1 判对的 Intact，却被 Top-8 Damaged 误导”的检索错误。因此后续不能再把“加对比学习”视为新方案；新增点只能是查询与负例关系。

v2 库目前建成 3411 条，296 条放宽了优先的单图上限；`all_checks_passed=true` 只表示来源、配额、容量等结构检查通过。评估在额外 A1 重放检查处停止，尚无 v2 Val 指标。v2 按四类最大占比优先选纯区域，未保证外观覆盖；v2 的 Top-64→多来源 Top-8 限制单图／单事件重复，但没有直接解决 Intact↔Damaged 类别混淆。先修复或部署此前的 A1 同次前向重放修正并完成正式 Val450，不能从未完成报告中报收益。

## 最先完成的无门控四格实验

固定前端、Q、投影、Reader、每事件×主类配额和 3411 总容量，比较：旧库＋原 Top-8、旧库＋来源多样 Top-8、质量库＋原 Top-8、质量库＋来源多样 Top-8。四格必须同一次前向、通过正式 E0 复现门槛，不能用独立重跑时的 Top-8 ID 完全一致当门槛。

同时报告完整四类 IoU、前景 mIoU、每事件结果、真 Damaged 的救回／伤害、真 Intact→Damaged 的新增与旧误报消除。检索诊断要对同一查询分别统计：正确类支持是否在整个库／Top-64／Top-8 出现，高纯查询的正确类 Hit@8，混合查询与支持软标签的相容性、支持来源重复、绝对相似度以及 Q 前景占比／前景内部纯度。Val 真值只在预测与邻居 ID 固定后用于诊断，绝不能进入 Val 检索。

## 根据四格结果，只改 Memory 的下一刀

1. **整个库缺合适参考**：保持容量和事件×类配额，先按前景占比、前景内部类别纯度和 Q 面积分层，保证可靠的小 Damaged 与容易看成 Damaged 的真 Intact 都有代表。每桶在可靠样本中做特征空间覆盖选样（如 greedy coreset／k-center）并控制单图重复，与“纯度优先”单独对照。混合样本继续保存四维比例，不能按主类强行改成硬标签。纯度并不等于按栋。
2. **Top-64 有正确案例，Top-8 没有**：固定库，先比较余弦排序与来源去重；再在 Train 挖掘真实 Intact 近邻被 Damaged 占据、真实 Damaged 找不到 Damaged 的查询与跨事件困难负例，只训练检索投影。现有普通 SupCon／J1 困难度采样都不能代替这类“被检索误导”的查询。混合查询使用软标签相容度或降低权重，不能强行按主类压到一个中心。
3. **正确案例在 Top-8，却与查询软标签不相容或 J3 不改善**：先检验 token 主分支的空间加权与 Q 面积软标签是否错位，并分离检索 key 与 Reader value。可在同一 Q 上用原 token 和 Q 池化的 A1 语义特征分别建索引，而 Reader 仍读取原 token／软标签；Train、Val 的 key 必须由同一前端和规则提取。若改 key，不可沿用旧库索引。检索更好而固定 Reader 输出未涨，也可能是 Reader 输入分布改变，不能据此否定 Memory。
4. **错误主要由混合 Q 引起**：先用更小或重叠的局部支持条目／多尺度支持表示做同容量消融。只有支持侧与查询侧按同一预测建筑范围拆分，才能测试建筑片段 Memory；Val/Test 真值不能切查询。不要先重训 Q 或强制“一块 Q＝一栋楼”。

## 论文与作者代码如何借鉴

- [Gould 等，ECCV 2014：Superpixel Graph Label Transfer](https://users.cecs.anu.edu.au/~sgould/papers/eccv14-spgraph.pdf)；[作者 Darwin 实现说明](https://users.cecs.anu.edu.au/~sgould/darwin/drwnProjNNGraph.html)：多尺度重叠区域保存经验类别分布，在像素端合并区域参考。支持混合软标签与局部证据，不会凭空产生建筑 ID。
- [PatchCore，CVPR 2022](https://arxiv.org/abs/2106.08265)；[作者代码](https://github.com/amazon-science/patchcore-inspection)：greedy coreset 保留训练特征空间覆盖。可迁移到同容量、同事件×类别配额选样；原任务是正常样本异常检测，不能照搬增益。
- [ReCo，ICLR 2022](https://arxiv.org/abs/2104.04465)；[作者采样代码](https://github.com/lorenmt/reco/blob/main/module_list.py)：困难查询与混淆负类采样。可用于重训检索投影，不涉及采纳／拒绝门控；原方法不是跨事件历史库。
- [ASGNet，CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Adaptive_Prototype_Learning_and_Allocation_for_Few-Shot_Segmentation_CVPR_2021_paper.pdf)；[作者代码](https://github.com/Reagan1311/ASGNet/blob/main/model/asgnet.py#L198-L249)：支持前景内多个局部原型，按查询匹配。适合思考同类多外观覆盖，但其 few-shot 支持有 mask，和此处持久 Train 库不同。
- [Tip-Adapter，ECCV 2022](https://arxiv.org/abs/2207.09519)；[作者缓存读出代码](https://github.com/gaopengcuhk/Tip-Adapter/blob/main/main.py#L645-L653)：绝对相似度保留在未归一化缓存读出中；只可借鉴诊断，当前直接标签预测已证明会大量误改 Intact。

本轮分析不训练采纳／拒绝门控，不扩大库，不以 Test878 挑选方案。首次通过正式 Val 的新版 Memory 结果出现前，所有“新库更好／更差”的说法都是未验证假设。
