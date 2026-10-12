# 面向 BRIGHT 的指标对齐伪实例与修改簇验证方案

- 用户研究方向：灾前 RGB—灾后 SAR 多模态建筑损毁语义评估
- 用户研究问题：优化 LocCC 伪实例内部统计，并以规范的 Train/Validation/Test 流程把 Cluster Oracle 空间转化为可部署增益
- 分析日期：2026-09-08
- 论文来源：BRIGHT 数据论文；Lovász-Softmax；用户现有 P3、fixed S2、Building/Cluster Oracle、SMP、D1 实验
- 趋势来源：BRIGHT 2026 Challenge 总结

## 论文可迁移模块

| 模块 | 原文位置 | 可迁移方式 | 适配条件 |
|---|---|---|---|
| 数据集级 IoU 对齐目标 | Lovász-Softmax Sec.3/3.1 | 用训练集混淆矩阵计算簇级 mIoU 边际效用 | 权重只能由 meta_train 冻结 |
| 跨事件评估 | BRIGHT 数据论文与 2026 Challenge | 按灾害事件分层并报告事件宏平均、最差事件 | 查询事件不得进入训练或 Memory |
| 光学主导定位、模态专用编码、晚融合 | BRIGHT 2026 Challenge 总结 | 保持 P3 定位和双编码器冻结，只在候选修改验证端融合 | 不重新堆叠早期 RGB-SAR 对齐模块 |

## 匹配与可行性评估

- 收录信息：BRIGHT 正式发表于 ESSD 2025；Lovász-Softmax 发表于 CVPR 2018；BRIGHT 2026 Challenge 总结为 arXiv 预印本。
- 有无代码：BRIGHT 官方提供基准代码；当前项目已有 P3/fixed S2/簇缓存；本方案主要复用缓存。
- 任务匹配程度：高。直接针对 BRIGHT 的 mIoU、跨事件泛化和伪实例一致性。
- 创新度：中高。创新不在增加一个注意力块，而在“候选修改—指标边际效用—安全回退”的闭环。
- 迁移难度：中。第一阶段不需重新运行 P3；第二阶段仅训练低容量模型。
- 可能出现的问题：Cluster Oracle 是 GT 上限而非实际效果；伪实例统计可能事件相关；Validation 阈值过拟合；测试集不可用于反复调参。

## 落地建议

- 建议采用的模块：多分布伪实例描述、簇—建筑对比描述、metric-aligned cluster utility、低容量分层验证器。
- 需要替换的数据/任务：把旧的 corrected-harmed 标签替换为由 meta_train 混淆矩阵冻结的 mIoU 边际效用标签。
- 最小可行实验：直接读取现有 D1 `meta_train.pt` 和 `meta_calibration.pt`，不重跑 P3，比较 pixel-net 标签与 metric-aligned 标签的 Oracle 和跨事件可预测性。
- 关键对照实验：fixed S2；方向/面积规则；旧 Utility；metric-aligned Logistic；两层 MLP；标签或空间打乱。
- 风险控制：meta_train 拟合；meta_calibration 选唯一阈值；meta_confirm 一次确认；Val450 仅在确认后运行；Test 只做最终一次提交。

## 建议题目方案

1. 题目：Metric-Aligned Pseudo-Instance Refinement for Cross-Modal Building Damage Assessment
   - 核心思路：把 S2 修改簇的接受决策与数据集 mIoU 边际变化直接对齐。
   - 可落地实验：P3 + fixed S2，比较旧像素效用和新指标效用。
   - 预期贡献：减少类别不平衡下“像素净收益为负但 mIoU 有益”的监督错位。
2. 题目：Safe Cluster-wise Context Propagation for RGB-SAR Building Damage Mapping
   - 核心思路：伪建筑内生成候选、修改簇级安全接受、失败时回退 P3。
   - 可落地实验：建筑级、簇级、像素级 Oracle 与实际恢复率对照。
   - 预期贡献：结构一致性与保守部署相结合。
3. 题目：Distribution-Aware Pseudo-Building Calibration for Event-Generalized Damage Assessment
   - 核心思路：以类别概率分布、分位数、熵、碎片度及簇—稳定区差异替代简单均值。
   - 可落地实验：Logistic 与小型 DeepSets，严格留事件评估。
   - 预期贡献：提高跨事件伪实例统计的表达能力。

## 证据

| 类型 | 来源 | 支撑内容 |
|---|---|---|
| 原文 | Berman et al., CVPR 2018, Lovász-Softmax Sec.3/3.1 | 像素损失、图像 mIoU 与数据集 mIoU并不等价，应显式考虑 IoU 目标 |
| 原文 | Chen et al., ESSD 2025, BRIGHT | 数据覆盖多事件，跨事件泛化是核心难点 |
| 趋势 | Chen et al., arXiv 2607.22746 | 2026 Challenge 的未见事件性能明显下降；领先方案偏向模态专用编码、晚融合和光学主导定位 |
| 项目实验 | Cluster Oracle / D1 | Cluster Oracle 有 +1.2093pp 空间，但现有单变量高分辨率证据未达到跨事件准入门槛 |
