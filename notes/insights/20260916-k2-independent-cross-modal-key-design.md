# K2：独立跨模态关系 Key 设计

## 研究问题

在冻结 P3 的前提下，构造一个不读取 P3 最终四分类概率、但能利用分类前 RGB 与 SAR 证据的检索 Key，判断 Memory 是否能提供与 P3 互补的跨事件损毁证据。

## 核心边界

“独立”不是完全不用 P3，而是：

1. 允许读取 P3 分类器之前的 RGB 与 SAR 特征；
2. 对这些特征 `detach`，不反向修改 P3；
3. 禁止把 P3 的最终 logits、softmax 类别概率和 argmax 类别放入 Key；
4. 使用单独训练的模态投影、局部对齐和 Key Encoder；
5. P3 概率只能在最后的安全 Verifier 中使用，不能参与近邻检索。

## 推荐数据流

```text
灾前 RGB -> 冻结 P3 RGB encoder -> 分类前 F_rgb --detach-->
                                                       RGB projector
灾后 SAR -> 冻结 P3 SAR encoder -> 分类前 F_sar --detach-->
                                                       SAR projector
                                                           |
                                      RGB 邻域内搜索最相关 SAR
                                                           |
                                    aligned SAR + entropy + offset
                                                           |
                         [RGB, SAR, |RGB-SAR|, RGB*SAR, cosine,
                          norm ratio, alignment confidence]
                                                           |
                                建筑内 SLICO 区域统计汇聚
                         mean/std/max/attention/core-ring/geometry
                                                           |
                                  独立两层 MLP -> 128D L2 Key
                                                           |
                                   Train-only Memory 检索
```

## 与当前 V2 的区别

当前 V2 已包含 RGB/SAR 独立投影、局部容错对齐、区域多统计量和 128D Key，是 K2 的雏形。但其关系输入还包含 P3 的 `fused` 特征，并复用训练过的 `cross_token`；Key Encoder 同时带三分类头。因此它处于 K1 与 K2 之间，可能与 Direct Head 学到相似分类边界。

纯 K2 应进行以下修改：

- 删除关系输入中的 `fused`；
- 新建 K2 专属 `rgb_projector`、`sar_projector`、`relation_encoder`；
- Key Encoder 与 Direct 分类头解耦；
- 检索 Key 不读取 P3 logits；
- 将 P3 entropy/margin 留到最终 Verifier，而不是放入 Key。

## 建立步骤

### 阶段 A：固定查询单位

为了只验证 Key，不同时改变多个因素，K0-K4 应使用完全相同的建筑条件 SLICO 区域、可靠性筛选、Memory 容量和 Top-k。第一轮不要同时改成整栋建筑 Memory。

### 阶段 B：训练 Key 空间

只使用完整 Train：同类别、不同事件的区域为正样本；不同类别区域为负样本。建议损失为：

```text
L = L_cross_event_supcon
  + 0.2 * L_class_ce
  + 0.1 * L_shift_consistency
```

`L_shift_consistency` 要求同一区域在人工 SAR 位移后仍产生接近的 Key。GT 只用于 Train 的监督与 Memory value，Val/Test 禁止写入 Memory。

### 阶段 C：建立 Memory

对 Train 中可靠建筑区域生成 Key，保存 Key、软标签、事件、图像、建筑/区域编号、纯度和对齐质量。按类别与事件均衡采样，防止某个大事件或 Intact 类占满库。

### 阶段 D：只读检索

Val 查询时生成同样的 K2 Key，取 Top-k 邻居。机制验证时排除同图和同事件，并限制单一事件最多贡献两个邻居。输出类别 posterior、第一第二类 margin、邻居 agreement、来源事件数/熵和最大相似度。

### 阶段 E：安全使用

Memory 不直接覆盖像素类别。P3 logits、P3 entropy/margin、Memory 置信度、对齐熵只进入低容量 Verifier。只有 Verifier 预测修改效用为正时，才给 P3 一个零初始化的小残差；否则保持 P3。

## 必须做的消融

| 变体 | Key 内容 | 目的 |
|---|---|---|
| K0 | P3 logits/probability | 测量“复制 P3”上限 |
| K1 | P3 fused feature | 测量融合特征重复程度 |
| K2 | 独立 RGB/SAR 关系，不含 fused/logits | 核心方法 |
| K2-no-align | K2 但不做局部对齐 | 证明容错对齐作用 |
| K2-shuffle | K2 但打乱 Memory 标签 | 证明正确检索作用 |
| K3 | K2 局部 Token 组成建筑 bag | 第二阶段测试建筑标注单位 |
| K4 | K3，P3 uncertainty 仅进入 Verifier | 最终安全结构 |

所有变体必须使用相同查询区域、训练步数、Key 维度、Memory 容量和 Top-k。

## 评价重点

裸检索 Macro-F1 不是主结论。需要统计四种条件：

1. P3 正确、Memory 正确；
2. P3 正确、Memory 错误（潜在伤害）；
3. P3 错误、Memory 正确（真正互补）；
4. P3 错误、Memory 错误。

核心指标是 P3 错误子集上的 Memory correction rate、P3 正确子集上的 harm rate、Oracle recovery、Memory 相对 Direct Head 的额外恢复以及跨事件事件胜率。

## 第一阶段准入建议

- K2 在 P3 错误子集上的正确率明显高于 K0/K1；
- K2 的正确建议减错误建议为正；
- K2 Oracle recovery 至少达到 25%；
- K2 与 Direct Head 的 Union 比 Direct Head 单独至少多恢复 0.20pp；
- 随机标签 Memory 后互补收益显著消失；
- 至少 10/14 事件不下降。

若 K2 裸检索略低但在 P3 错误子集更强，仍值得继续；若只在 P3 已正确区域表现好，则说明它仍在重复 P3，不能进入回写阶段。

## 风险

1. 分类前特征仍来自 P3，因此是“避免最终决策复制”，不是统计意义上的完全独立模型。
2. RGB 与 SAR 的投影若未先学到共同关系空间，直接余弦对齐可能不可靠。
3. SLICO 只是局部证据容器，不等于建筑实例，也不应直接决定整个区域类别。
4. 若同时改变 Key、区域划分和 Memory 单位，将无法归因提升来源。

## 推荐推进顺序

```text
先固定现有查询区域，完成 K0/K1/K2/K2-no-align
    -> 分析 P3 错误子集互补性
    -> K2 通过后再把多个局部 Token 聚合为建筑 bag（K3）
    -> 最后训练带 P3 uncertainty 的安全 Verifier（K4）
```

