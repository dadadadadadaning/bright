# BRIGHT Memory：四类条件应在哪里设置

目前V3是固定成员的新FP32建库与诊断实验。设置是否有效要看执行代码是否读取它；不能把概念参数当现有环境变量。

| 条件 | 当前代码状态 | 正确操作 |
| --- | --- | --- |
| 评价查询前景≥0.8 | diagnose._weighted_evidence_diagnostic已实现 | 读取foreground_pure/foreground_mixed，检索本身仍使用全部查询 |
| 评价纯度≥0.8 | 已实现，区分纯净与混合前景 | 读取已有分组，无需改源码或重跑 |
| Damaged/Destroyed单独统计 | 已实现，类ID2/3 | 读取类软证据、IoU、rescue/harm/FP |
| Train支持前景≥0.5 | 尚未接入V3排名 | 独立预览筛选库存，再在新实验中明确候选策略 |
| Reader质量特征 | 已保存6维quality，Reader未训练实现 | 在E2真实接入foreground比例、foreground内部纯度等特征 |

第一、二项只是评价分组，使用Val真实标签是事后评分；不能用它们决定部署时哪些query进行检索。最终仍用全Val450混淆矩阵和前景三类mIoU，保留背景→建筑的假阳性。

支持筛选预览采用已知Train标签：

```python
support_fg = labels[:, 1:].sum(1)
support_fg_purity = labels[:, 1:].max(1).values / support_fg.clamp_min(1e-12)
keep = original_valid & (support_fg >= 0.5)
```

此处0.5表示“总前景软标签质量至少50%”，不是标签的最大类别不是背景。背景0.45、三个前景0.20/0.20/0.15会通过，却仍以背景为主类。实际应逐成员计算，不可直接用总条数减896。

PREVIEW_MEMORY_SUPPORT_FILTER.py默认 --min-fg 0.5 --min-purity 0.0。0.0表示本轮不额外用纯度硬筛选，只审核前景比例这一个因素；如独立比较纯度0.8，可以显式传 --min-purity 0.8，但不将两种改变混在主线结论里。

预览只读库和小元数据，并验证paired_bank SHA。仅生成内存中的mask，保留原行号和member_ids，不修改bank、协议、检索排名或Val结果。输出筛后条数、类别质量和跨事件候选量；不提供新检索性能。

现V3 select_neighbors通过两key共同有效范数和事件/图像排除决定候选，不读取bank.eligible作为质量过滤开关。修改eligible不能正确接入fg规则，也违反原库与协议绑定。实际筛选需要在独立实验把候选mask作用于排名、控制及产物验证，T128/F256共用同一个mask。

单独剔除背景支持不保证更好：背景query可能只能找到前景案例，从而增加误报。E2主线先保持原库，把Train支持fg比例、foreground内部纯度、邻居标签冲突等输入拒绝门，使用当前基础预测不确定性；不能使用Val真实标签做门控。阈值筛库作为独立消融。

当前quality[4]是四类标签的最大概率加权均值，包含背景；背景纯案例也会接近1，它不等于foreground内部纯度。两项应分别计算，不直接改名使用。
