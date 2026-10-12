# A2 局部损毁 MIL 监督改造：从整包分类到局部受损发生辅助任务

- 用户研究方向：灾前 RGB / 灾后 SAR 建筑损毁像素分割（BRIGHT）。
- 用户问题：A2 如何避免 Bag 汇聚吞掉小面积损毁；为何 Bag 仅作训练监督；如何以可验证方式改造。
- 日期：2026-09-16。

## 当前实现的事实

1. A0--A3 全部从同一 D0 checkpoint 初始化，不是 A2 从训练完成的 A1 继续初始化。
2. A2 使用 `roi_alpha=0.1`、`use_bags=True`、`use_hierarchy=False`；因此在结构上等同 A1 的 ROI 路径加 Bag/MIL 辅助损失。
3. 训练 Bag 不是完整建筑标注：`build_training_bag_specs` 从 GT 的 1/2/3 类分别提取四连通分量。一个 Damaged 分量的 Token 不会被同一建筑的 Intact 像素平均淹没，但相邻同类建筑可能合并，且该分量选择依赖 GT。
4. A2 将 `mil_residual_scale=0`，因此不把 Bag 类别广播回像素；训练时只通过 `L_mil` 改变共享 RGB/SAR/fused 特征与 MIL head。

## 现有损失

`L = L_location + L_damage + s(t) * 0.25 * L_mil`，其中 `s(t)=0`（前 1000 optimizer steps）或 1（之后）。A2 不包含 A3 的 existence、severity、guard loss。

## 为什么不直接写回 Bag 预测

一栋/一个连通分量内的 Bag 概率没有像素位置；将它广播回所有 Token 会让一个局部 Damaged 结论覆盖结构边界、完整屋顶或邻接区域。已有完整建筑 Memory 像素写回的负结果也说明这种广播在 BRIGHT 中不安全。

## 可迁移证据

| 来源 | 可迁移模块 | 对 A2 的含义 |
|---|---|---|
| Ilse et al., ICML 2018, Attention-based Deep MIL, https://proceedings.mlr.press/v80/ilse18a/ilse18a.pdf | 可学习注意力而非均值汇聚 | 必须检查受损 Token 权重是否集中；但不能将注意力当分割标签。 |
| Li et al., CVPR 2021, DSMIL, https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Dual-Stream_Multiple_Instance_Learning_Network_for_Whole_Slide_Image_Classification_CVPR_2021_paper.pdf ; code https://github.com/binli123/dsmil-wsi | critical-instance stream + bag stream | A2 已有同类思想；当前 hard critical token 需要改为 top-k soft critical evidence，避免单个 SAR 噪声 Token 主导。 |
| Durand et al., CVPR 2016, WELDON, https://openaccess.thecvf.com/content_cvpr_2016/papers/Durand_WELDON_Weakly_Supervised_CVPR_2016_paper.pdf | top-k positive 与低分实例共同聚合 | 对小损毁使用 top-k positive occurrence，而非整包平均或单一最大值；低分项只能作为 Bag 分类约束，不可反向标注为背景。 |

## 推荐的 A2-R1（只在 A2-A1 通过后实施）

### 模型

保留 P3/A1 像素主路径。给每个 GT 建筑/连通组件的 feature-grid Token 增加一个二类 `affected_token_head`，只输出“是否受损发生”，不输出四类最终标签。

### 训练目标

对每一个类别为 Damaged/Destroyed 的 Bag：

`L_topk_occ = BCE(mean(top-k(p_affected)), 1)`。

对 Intact Bag：

`L_intact = BCE(mean(p_affected), 0)`。

像素级真实 Affected Token 仍使用现有 `L_damage`；新增 Bag loss 只是让少量高分 token 不能被整包平均稀释。候选 k 使用 `max(1, ceil(0.05*N))`，上限 8，且 k 的选择只在 meta_train 固定。

### 禁止项

- 不能用 GT Bag mask 推理；
- 不能把 Bag 标签广播给所有像素；
- 不能以 A2 的验证集结果继续调 k；
- 不训练 A2-R1，除非先确认 A2 实际 Bag 数和 MIL loss 非零，并完成 A2-A1 比较。

## 最小实验

1. B0=A1：ROI 像素路径。
2. B1=A2：当前 DSMIL Bag 监督。
3. B2=A2-R1：B1 + top-k local occurrence loss。
4. B3=B2，随机选 top-k Token（参数/额外损失对照）。
5. B4=B2，均值池化替代 top-k。

仅当 B2 同时超过 B1、B3 和 B4，且提高小损毁/边界 Affected recall 而不降低 Damaged F1 时，才能说明局部 MIL 真的解决了“少量受损 Token 被稀释”。
