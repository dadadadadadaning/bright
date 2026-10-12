# H6-B1R 类别条件稀疏残差路由

## 问题与假设

H6-B1 的 stage4 传感器原型能够降低 RGB--SAR gap，但共享空间门控在建筑区域趋于饱和。原生 SAR 残差随后持续改变 P3 的类别边界，验证集对 destroyed 的收益没有稳定迁移到测试集，damaged 反而退化。

B1R 只检验一个变量：将共享门控替换为低开启率、类别条件的稀疏路由。它不修改 P3、A2 原生 SAR 专家、stage4 传感器原型或损失函数。

## 结构约束

- P3 完整推理锚点冻结；
- A2 原生 SAR 专家及 stage4 传感器原型冻结；
- RGB 编码器、定位头和全部 BN 冻结；
- 仅训练 `safe_residual_router`；
- intact、damaged、destroyed 使用三个独立空间门；
- background 门和残差恒为零；
- 门控偏置为 `-4`，初始开启概率约 `1.8%`；
- 类别转换系数由 `0.1*tanh(parameter)` 限制在 `[-0.1, 0.1]`；
- 初始输出严格等于 P3；
- 仍只使用原有 CE + Lovasz，不增加辅助损失。

## 实验协议

Pilot 为 1500 optimizer steps，每 250 步运行一次完整验证集。运行 3407、2027、7777 三个种子。开发阶段禁止访问 Test。

每个种子的验证集最佳权重必须同时满足：

- Valid mIoU 不低于 60.5019%；
- Damaged IoU 不低于 43.0309%；
- Destroyed IoU 不低于 62.8957%；
- Event mean 不低于 52.7280%；
- Event worst 不低于 40.7300%。

至少两个种子同时通过，才允许按 Valid mIoU 从通过者中锁定一个 checkpoint，并执行一次 Test。Test 只用于最终报告，不能反向选择步数、种子或超参数。

已经查看过的历史 Test 结果应标记为开发期探索结果，不再视为完全独立的盲测结果。若论文要求严格无偏结论，应额外建立从未查看过的事件级 holdout。
