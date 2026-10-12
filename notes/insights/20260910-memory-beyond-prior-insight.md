# Memory 不作为语义先验时的可迁移方案

- 用户研究方向：灾前 RGB—灾后 SAR 建筑损毁评估
- 用户研究问题：完整建筑 Memory 的建筑级分类有效，但辅助先验、整栋像素写回和 S2 修改选择失败后，Memory 还能承担什么职责
- 分析日期：2026-09-10
- 现有证据：完整建筑 Memory 建筑级 Macro-F1 +5.24pp；像素写回 M4 相对 fixed S2 -0.2029pp；完整卡 AUC 仅比 P3 全局检索高 0.0046

## 可迁移模块

| 模块 | 可迁移方式 | 适配条件 | 风险 |
|---|---|---|---|
| 跨事件原型度量教师 | Memory 仅在训练期构造同类跨事件正对和异类困难负对，学习检索专用投影空间 | 训练批次按事件和类别均衡 | 伪实例多数标签噪声 |
| 类别原型正则化 | 用鲁棒类别原型约束 P3 分类器权重或建筑 token，不在推理期执行 KNN 写回 | 保持定位分支冻结 | 原型可能过度平滑事件特有损毁 |
| 角色 token 支持交互 | 查询建筑的像素/角色 token 与检索病例的 core/ring/change/stable token 做局部交叉注意力 | 先证明学习后检索优于 P3 全局检索 | 计算量和错误邻居传播 |
| 困难病例路由 | Memory 只判断当前建筑是不是 P3 的高风险病例，再调用一个局部专家头 | 需要 train-only 的 P3 错误类型标签 | 路由器可能跨事件失效 |
| 训练样本课程与重采样 | Memory 搜索稀有类、困难正样本和近邻异类，用于对比学习与批次构造 | 不改变推理结构 | 更像训练策略而非网络模块 |

## 推荐落地顺序

1. R-M1：训练检索专用投影头。正样本为同损毁类、不同事件的完整建筑；负样本为不同损毁类但 P3 特征相近的建筑。使用软类别分布或纯度权重降低多数标签噪声。
2. 只做检索准入：相对 P3-global AUC 至少 +0.02、邻居纯度至少 0.70、Damaged/Destroyed Recall@8 提升、至少 10/14 留出事件改善。
3. R-M2：通过后，将 Memory 变为训练期类别原型正则，推理时移除病例库；比较 P3 与 P3+prototype training 的像素 mIoU、Damaged F1 和事件稳定性。
4. R-M3：只有 R-M1 检索空间通过，才让查询像素/角色 token 对检索到的支持角色 token 做置信门控交叉注意力；禁止整栋类别广播。
5. R-M4：将“P3 是否可能错”作为独立路由任务，Memory 只触发困难建筑专家，不直接输出最终类别。

## 最小实验矩阵

| 编号 | 方法 | 目的 |
|---|---|---|
| B0 | frozen P3 key | 当前强捷径基线 |
| B1 | fixed complete-card key | 已知结果 |
| B2 | learned cross-event metric key | 验证 Memory 能否获得 P3 之外的检索空间 |
| B3 | B2 去掉事件均衡 | 验证跨事件采样作用 |
| B4 | B2 去掉困难负样本 | 验证损毁等级边界作用 |
| B5 | B2 使用硬多数标签 | 与软标签/纯度加权比较 |

R-M1 不通过时停止推理期 Memory，保留其作为稀有类训练采样工具。R-M1 通过后，再在原始数据划分上训练 R-M2/R-M3；不得用已消费的 meta-calibration 反复选择结构。

## 建议题目方案

1. **Cross-Event Prototype Metric Learning for RGB–SAR Building Damage Assessment**：Memory 作为跨事件度量教师，提高稀有损毁表征。
2. **Role-Aligned Retrieval Interaction for Multimodal Building Damage Segmentation**：检索完整建筑后，以角色 token 局部交互代替整栋广播。
3. **Memory-Routed Hard-Case Experts for Cross-Event Disaster Damage Assessment**：Memory 只负责识别 P3 高风险建筑并路由到专家。

## 趋势依据

- Pin the Memory（CVPR 2022）将类别 Memory 与域泛化元学习、类别内聚和 Memory 分离约束结合，说明 Memory 可以首先承担跨域表征学习职责，而不是直接替换输出。
- SCCA（ICCV 2023）指出把支持样本压缩成单个原型会损失空间细节，并采用对齐后的支持 patch 交叉注意力；这与本项目“建筑分类提高但整栋写回损害像素”的结果一致。
- Retrieve and Segment（CVPR 2026）采用学习式、按查询融合支持特征，而不是固定后期融合，支持将病例用于轻量查询适配器。
- RMem（CVPR 2024）强调有界 Memory 中样本信息量与新鲜度的平衡；本项目可迁移为类别—事件均衡及代表性/困难病例共同维护。
