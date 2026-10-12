# 推理 Memory 与全局重排序的可挽救边界

## 判断

被数据否定的是当前实例，而不是整个方法家族：P3 Mean Key + 余弦/kNN + 全局阈值或共享评分器，不能稳定区分 Damaged 救援与 Intact 误伤。继续扫描 Top-k、alpha、margin、epoch 不构成新方法。

## 推理 Memory 仍可研究的版本

1. 一侧选择性检索：P3 保持主预测，Memory 只提出 Non-Damaged→Damaged 建议；独立验证器使用 P3 未包含的新证据，并允许绝大多数样本拒绝。
2. 支持集交叉验证器：Mean 仅负责召回 Top-20；验证器读取 Query 与多个支持建筑的局部 token、边界、SAR 统计及邻居间一致性，不再只读取 Mean 距离。
3. 风险控制：阈值在 Train-LOEO 上锁定，Validation 只执行一次；若无法产生正收益覆盖点，系统自动退化为纯 P3。

## 全局 Embedding 仍可研究的版本

1. 分层空间：先学习 Intact/Affected，Affected 内再区分 Damaged/Destroyed，避免一个三类空间承担两种语义。
2. 多原型 Damaged：每种跨事件可复现的局部损毁模式一个代理；只有聚类稳定性通过审计后训练。
3. 局部候选重排序：冻结 Mean 粗检索，只训练 token-level Query–Support 验证器，不改写整个向量空间。

## 不可挽救的版本

- 同一 P3 Mean Key 上继续扫描 k、权重、温度和阈值。
- 仅靠 conformal/abstention 给不可分分数套校准；它可以拒绝，却不能创造判别信息。
- 继续共享全局 Adapter，同时希望只移动 Damaged 而不影响 Intact/Destroyed。

## 决定性实验

V0：Top-20 局部 token Oracle 与新证据审计。仅对 P3=Intact、Memory=Damaged 的 Train-LOEO proposal，比较 Mean-only 与 token/local-SAR 验证器。要求 Combined AUROC 至少 0.70、相对 Mean-only 至少 +0.05，并存在 precision≥55%、净纠正>0、非负事件≥10/14 的锁定点。V0 不通过则终止推理 Memory。

E0：Occurrence/Severity 分层嵌入审计。冻结 P3，分别审计二分类 Occurrence Key 与 Affected 内 Severity Key；只有二者跨事件指标均优于原 Mean，才训练分层度量空间。
