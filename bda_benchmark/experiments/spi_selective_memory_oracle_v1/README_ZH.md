# Selective Memory Oracle V1

## 目的

本实验不训练 Verifier，也不访问 Test。它只回答：

> 在固定 P3、固定 Metric Encoder 和固定训练集 Memory 的条件下，如果有一个理想裁判知道每次 Memory 建议是否有利，现有建议池最多包含多少可恢复的验证集 mIoU？

## 严格边界

- Memory 建议只由 P3 特征、SLIC、Metric Encoder 和训练集 Memory 产生。
- GT 仅在建议产生之后用于 Oracle 接受或拒绝，不能参与检索。
- 第一版只修正 `Intact ↔ Affected`，不让 Memory 决定 `Damaged ↔ Destroyed`。
- P3像素基线严格使用锁定配置 `roi_alpha=0.1`；Memory描述子继续使用与既有V2缓存兼容的“无ROI加权 fused 特征”。
- Affected 建议给 Damaged/Destroyed 添加相同 logit，保持 P3 的严重度顺序。
- 报告的是“给定建议空间的贪心上限”，不是可部署结果，也不是数学上的全局组合最优。

## 关键比较

- `Memory Oracle gain`：现有 Memory 建议池的理论空间。
- `Direct-head Oracle gain`：相同 Metric Encoder 分类头、不检索 Memory 时的理论空间。
- `Memory Oracle - Direct-head Oracle`：Memory 本身是否提供额外候选价值。
- `Oracle recovery`：P3 漏掉的 Affected 像素中，理想选择可找回多少。

若 Memory Oracle 增益很小，停止训练复杂 Verifier，先修复描述子、Memory 采样或区域定义。若 Memory Oracle 明显高于 Direct-head Oracle，再按照相同逻辑在 **Train** 上另建训练目标并训练低容量 Verifier。

`val450_metric_targets_*.pt` 只允许做 Oracle 误差分析，禁止用于拟合 Verifier；否则会把验证集标签泄漏进模型。Train 目标还必须排除同图检索，建议同时做排除同事件版本。

## Complementarity阶段

`complementarity`与`complementarity_event`额外计算：P3漏检恢复漏斗、Direct/Memory建议重叠、联合Oracle、Memory独有候选Oracle。只有联合Oracle相对Direct Oracle继续提高至少0.20pp，才进入Residual Utility Memory实现。
