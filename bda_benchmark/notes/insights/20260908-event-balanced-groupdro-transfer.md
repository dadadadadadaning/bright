# BRIGHT 事件均衡与 GroupDRO 方法迁移建议

- 用户研究方向：灾前 RGB—灾后 SAR 的跨事件建筑损毁评估
- 用户研究问题：通用事件/域泛化方法中，哪些适合 P3 + fixed S2 + Metric-aligned Cluster Verifier
- 分析日期：2026-09-08
- 证据来源：BRIGHT ESSD 2025；GroupDRO ICLR 2020；DomainBed ICLR 2021；REx ICML 2021；SWAD NeurIPS 2021；MixStyle ICLR 2021；RobustNet CVPR 2021；Fishr ICML 2022；WILDS ICML 2021

## 核心判断

事件均衡不能等同于类别均衡。前者让每个灾害事件具有近似相同的训练影响，后者处理 Intact/Damaged/Destroyed 的数量差异。BRIGHT 中还需要区分“同事件新图像”和“训练中完全未见事件”。只有把整个事件留出，才能支持真正的 cross-event 结论。

当前 MACU `sample_weights` 已实现簇级事件逆频率加权，并乘以 `sqrt(|metric utility|)`。因此它应命名为 Event-balanced ERM，而不是普通 ERM；实验中必须补充不含事件权重的纯 ERM。

## 通用方法与迁移优先级

| 方法 | 通用做法 | 迁移到当前方法 | 优先级 |
|---|---|---|---|
| Event-balanced sampling/loss | 均匀采样域或对域内平均损失再平均 | 事件→图像→修改簇三级采样，或直接计算事件平均BCE | 立即做 |
| GroupDRO | 动态提高高损失group的权重 | event作为group，优化最难训练事件的Verifier损失 | 立即对照 |
| V-REx | 惩罚事件损失方差 | 降低不同事件之间Verifier损失差异 | GroupDRO不稳时 |
| CVaR/top-k event risk | 优化最差一部分事件而非单个最差事件 | 优化最差25%训练事件，降低小事件噪声 | 条件性 |
| Leave-one-event-out | 每次完整留出一个域 | 先做Verifier级LOEO；最终需要端到端事件留出 | 验证必做 |
| Event-balanced threshold | 用事件宏平均和最差事件选阈值 | 固定一个全局阈值，不使用测试事件专属阈值 | 立即做 |
| MixStyle | 跨域混合特征均值/方差 | 仅在P3的RGB/SAR分支内分别跨事件混合 | P3重训阶段 |
| RobustNet whitening | 去除域特定风格协方差 | 分别作用于RGB和SAR浅层特征 | 第二阶段 |
| CORAL/MMD | 对齐不同域特征分布 | 对齐晚期损毁特征，避免强行对齐原始RGB/SAR | 条件性 |
| SWAD | 平均训练轨迹中的平坦区域权重 | 用于重训P3，不适用于已冻结P3后的线性Verifier | 低成本对照 |
| Fishr | 对齐各域梯度方差 | 仅在基础方法仍不稳定且算力允许时尝试 | 后期 |

## 推荐的 Verifier 损失

对事件 `e` 内修改簇 `i`：

```text
cluster_loss_i = clipped_sqrt_abs_utility_i × BCE(score_i, label_i)
event_loss_e   = sum(cluster_loss_i) / sum(utility_weight_i)
```

事件均衡 ERM：

```text
L_EB = mean_e(event_loss_e)
```

GroupDRO：

```text
q_e ← q_e × exp(eta × EMA(event_loss_e))
q   ← normalize(q)
L_GDRO = sum_e q_e × event_loss_e
```

V-REx：

```text
L_VREx = mean_e(event_loss_e) + lambda × Var_e(event_loss_e)
```

不要同时使用事件均匀采样和再次乘完整的事件逆频率，否则会双重补偿。若batch已经按事件均匀采样，损失内只保留效用权重和必要的类别/正负标签权重。

## 最小实验矩阵

| 编号 | 方法 | 目的 |
|---|---|---|
| G0 | 普通簇级ERM | 确认大事件支配程度 |
| G1 | Event-balanced ERM | 当前MACU近似方案 |
| G2 | G1 + 事件均衡阈值选择 | 测试训练均衡和决策均衡是否互补 |
| G3 | GroupDRO | 提升最难训练事件 |
| G4 | V-REx | GroupDRO被噪声小事件支配时替代 |
| G5 | CVaR-25% | 单个最差事件噪声太大时优化最差四分位 |

所有方法使用相同特征、Metric Utility标签、模型容量、训练步数、拒绝比例上限和候选阈值。第一轮只使用 Logistic Regression；不要同时改P3或加入TTA。

## 批次构造

低容量Verifier推荐三级采样：

1. 均匀选择事件；
2. 在事件内均匀选择图像；
3. 在图像内选择修改簇，并尽量同时包含正效用和负效用簇。

这样可以防止“某事件图像多”“某图像簇多”两次支配训练。若采用全批量训练，则直接先算每个事件的平均损失，再对事件平均。

## 阈值选择

未知事件没有标签，因此最终只能使用一个全局阈值。meta_calibration上先筛掉拒绝率>70%、Damaged F1下降的阈值，再要求至少10个事件不下降；满足约束后，优先最大化事件宏平均mIoU增益，最后才以总体mIoU破平局。

不应为每个已知事件保存单独阈值，因为部署时新事件的最优阈值未知。

## 验证协议

### Verifier级跨事件验证

冻结现有P3和S2缓存，每次拿一个完整事件作为验证事件，其余事件训练Verifier。该实验成本低，回答“Verifier能否迁移到没有参加Verifier训练的事件”。它不能证明P3本身没见过该事件。

### 端到端跨事件验证

训练P3时也完整排除目标事件，随后重建S2和修改簇，再训练Verifier。这才回答“完整系统能否推广到未见事件”。成本高，可先预注册3至5个代表事件，再决定是否做14折。

### 指标

- 总体mIoU及相对fixed S2增益；
- 事件宏平均增益；
- 最差事件和最差25%事件；
- 非下降事件数；
- Damaged F1；
- Oracle recovery、拒绝率；
- 图像级和事件级bootstrap CI；
- GroupDRO最终事件权重，检查是否被单个小/噪声事件垄断。

## 准入建议

相对Event-balanced ERM，GroupDRO至少应满足：

1. 最差事件或最差四分位明显改善；
2. 总体mIoU下降不超过0.10pp；
3. Damaged F1不下降；
4. 负增益事件数减少；
5. meta_confirm置信区间下界不恶化；
6. 没有单一事件长期占据绝大部分DRO权重。

若GroupDRO只改善训练事件最差损失而不改善留出事件，停止使用；这说明它记住了最难已知事件，不等于学会了未知事件规律。

## P3阶段的迁移

在重训P3时，依次比较：ERM、事件均衡采样、GroupDRO、GroupDRO+SWAD。CE+Lovász/ordinal必须在四个方法中保持一致。MixStyle只能在RGB分支之间或SAR分支之间跨事件混合，不应直接混合两种模态的统计；CORAL/MMD优先放在晚期损毁特征，而不是强行对齐原始RGB与SAR。

## 风险

- GroupDRO保护的是最差“已知事件”，不能自动保证未知事件；必须LOEO验证。
- 小事件标签噪声会吸走DRO权重；使用EMA、权重上限、早停，必要时改用V-REx或CVaR。
- 事件均衡可能加剧类别不均衡；事件内还需监控三类分布，但不要把`事件×类别×修改方向`全部定义成group，样本会过度稀疏。
- 重采样会改变输出概率含义，最终Verifier分数需要在独立事件上重新校准。

## 建议题目

1. **Metric-Aligned Group-Robust Cluster Verification for Cross-Event Building Damage Assessment**：在修改簇层面结合指标效用和事件鲁棒风险。
2. **Event-Balanced Selective Context Refinement for Unseen Disaster Mapping**：突出全局阈值、安全回退和完整事件留出。
3. **Worst-Event Aware RGB–SAR Building Damage Segmentation**：将GroupDRO迁移到P3并与Verifier级稳健性联动。

## 原文与趋势证据

- BRIGHT ESSD 2025：https://essd.copernicus.org/articles/17/6217/2025/
- GroupDRO：https://arxiv.org/abs/1911.08731
- V-REx：https://proceedings.mlr.press/v139/krueger21a.html
- DomainBed：https://github.com/facebookresearch/DomainBed
- WILDS：https://proceedings.mlr.press/v139/koh21a.html
- MixStyle：https://openreview.net/forum?id=6xHJ37MVxxp
- RobustNet：https://openaccess.thecvf.com/content/CVPR2021/html/Choi_RobustNet_Improving_Domain_Generalization_in_Urban-Scene_Segmentation_via_Instance_Selective_CVPR_2021_paper.html
- SWAD：https://proceedings.neurips.cc/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html
- Fishr：https://proceedings.mlr.press/v162/rame22a.html
