# Event-Generalized P3–S2–Verifier 一体化改进方案

- 用户研究方向：BRIGHT 跨事件 RGB–SAR 建筑损毁语义分割
- 用户问题：事件均衡能证明什么，如何真正提高未见事件，以及P3、S2和Verifier如何形成统一创新
- 日期：2026-09-08
- 证据：BRIGHT ESSD 2025；当前P3/fixed S2结果；GroupDRO、V-REx、DomainBed、WILDS、MixStyle、RobustNet、SWAD

## 核心判断

事件均衡只是训练干预，不是未见事件泛化的证据。它能证明的最多是：数据量大的已知事件是否曾经支配训练，以及均衡后已知事件间的性能差距是否缩小。只有完整留出事件，且P3、S2参数选择和Verifier训练都未访问该事件，才能支持未见事件效果的结论。

当前应把系统重构为三个可单独验证的层次：

```text
Event-Robust P3
    ↓
Calibrated Boundary-aware S2 Candidate Bank
    ↓
Metric-aligned Multi-action Cluster Verifier
```

## 第一层：Event-Robust P3

不改变P3主干结构，先改变训练风险：

1. 事件→图像→crop三级均衡采样；
2. 比较普通ERM、事件均衡ERM、GroupDRO，GroupDRO不稳时使用V-REx；
3. 损失逐项比较CE、CE+Lovász、CE+ordinal，只有单项有效才组合；
4. 使用SWAD或固定早停降低最差事件过拟合；
5. 第二轮才加入模态内MixStyle：RGB事件间混合、SAR事件间混合，不跨模态混合。

P3准入不仅看总体mIoU，还要求事件宏平均、最差事件、最差四分位和D-F1改善。GroupDRO若只改善已知最差事件而不改善完整留出事件，则停止。

## 第二层：改进S2而非只审核S2

fixed S2的已知问题是整体有效但有大量有害修改，且LocCC可能粘连。建议从单一强度后处理改为候选库：

```text
A0：P3，完全不传播
A1：weak S2，低强度上下文
A2：fixed S2，中等强度上下文
A3：hard context，强区域一致化
```

所有候选先使用训练集冻结的概率校准。伪实例第一版保持LocCC，第二版使用距离变换+RGB边界的轻量拆分。不要直接训练Panoptic头。

为保证决策单位一致，可以先由fixed S2相对P3的变化并集定义修改簇，再在同一簇上计算各候选动作的反事实mIoU效用。若多强度候选实现复杂，第一版保留三动作：回退P3、减弱S2、接受fixed S2。

## 第三层：Metric-aligned Multi-action Verifier

旧二分类只判断接受/回退；新版本预测每个动作的效用：

```text
U(C, revert)
U(C, damp)
U(C, accept)
```

训练目标可使用效用回归或三动作排序损失，推理时选择预测效用最高的动作；当最大效用低于安全阈值时默认回退P3。事件均衡/GroupDRO作用在每个事件的动作效用损失上。

这样P3、S2和Verifier形成统一逻辑：P3产生稳健底线，S2产生结构候选，Verifier控制传播强度和风险。

## 实验矩阵

| 编号 | 模型 | 回答的问题 |
|---|---|---|
| B0 | 原P3 | 原始基线 |
| B1 | 原P3+fixed S2 | 现有结构收益 |
| B2 | Event-balanced P3 | 事件数量偏置是否重要 |
| B3 | GroupDRO/V-REx P3 | 最差源事件风险是否可改善 |
| B4 | 最佳P3+fixed S2 | P3改进是否传递到S2 |
| B5 | 最佳P3+calibrated S2 | 概率可靠性是否改善传播 |
| B6 | 最佳P3+boundary-aware S2 | LocCC污染是否限制候选质量 |
| B7 | B6+binary metric verifier | 接受/回退收益 |
| B8 | B6+multi-action verifier | 自适应传播强度收益 |

每一步只改变一项，并同时报告相对原P3、相对当前阶段P3以及相对该阶段fixed S2的变化。

## 跨事件验证

1. Verifier级LOEO：P3固定，每次完整留出一个事件不训练Verifier；成本低，但不能证明P3没见过事件。
2. 端到端LOEO：训练P3时也排除目标事件，重新生成S2和簇，再训练Verifier；这是完整证据。
3. 开发阶段预注册3–5个代表性事件做端到端pilot；方法冻结后再做全部事件或官方zero-shot协议。
4. 内层source-event validation选择超参数，外层held-out event只做评估；不得使用目标事件选择阈值。

统一报告：总体mIoU、事件宏平均、最差事件、最差25%、D-F1、三个类别IoU、非下降事件数、拒绝/减弱/接受比例、Oracle recovery、图像级和事件级bootstrap CI。

## 事件均衡的可证明边界

若Event-balanced ERM优于普通ERM，可以说明训练曾受事件样本量偏置影响；若已知事件方差下降，可以说明源事件训练更加均衡；若LOEO也改善，才能提供跨事件迁移的经验性证据。它不能证明因果不变性，也不能保证任意未来事件有效。

## 论文创新表达

事件均衡、GroupDRO、Lovász、MixStyle都不能单独声称新颖。核心创新应写成：

1. 将固定建筑上下文传播重述为未见事件下的结构化选择风险问题，而不是对所有建筑无条件平滑；
2. 提出mIoU对齐的多动作修改簇效用，使模型能在接受、减弱和回退之间选择；
3. 在事件鲁棒P3与边界感知伪实例候选之上，使用统一的事件留出风险—覆盖率协议验证平均性能和最差事件性能。

可用题目：

1. **Event-Robust Metric-Aligned Selective Context Refinement for Building Damage Assessment**
2. **Risk-Controlled Pseudo-Instance Context Propagation for Unseen Disaster Events**
3. **From Uniform Smoothing to Selective Structural Refinement in Cross-Event RGB–SAR Damage Mapping**

避免“首次使用GroupDRO”“首次使用Lovász”等无法成立的表述；使用“we formulate”“we introduce a task-specific utility/policy”“we establish an event-held-out evaluation”更稳妥。

## 风险与停止条件

- P3事件鲁棒训练若总体下降>0.10pp且LOEO不改善，不进入后续系统。
- S2候选库若Metric Oracle不高于fixed S2 Oracle至少0.20pp，不训练多动作Verifier。
- 边界拆分若降低定位IoU或产生大量碎片，保持LocCC。
- 多动作Verifier若不超过二分类Verifier，不作为主模型；保留较简单版本。
- 所有复杂化都必须在完整留出事件上有效，不能只在同事件随机划分上有效。

## 来源

- BRIGHT：https://essd.copernicus.org/articles/17/6217/2025/
- GroupDRO：https://arxiv.org/abs/1911.08731
- V-REx：https://proceedings.mlr.press/v139/krueger21a.html
- DomainBed：https://github.com/facebookresearch/DomainBed
- WILDS：https://proceedings.mlr.press/v139/koh21a.html
- MixStyle：https://openreview.net/forum?id=6xHJ37MVxxp
- RobustNet：https://openaccess.thecvf.com/content/CVPR2021/html/Choi_RobustNet_Improving_Domain_Generalization_in_Urban-Scene_Segmentation_via_Instance_Selective_CVPR_2021_paper.html
- SWAD：https://proceedings.neurips.cc/paper/2021/hash/bcb41ccdc4363c6848a1d760f26c28a0-Abstract.html
