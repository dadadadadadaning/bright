# Complete Building Memory V1

本实验先为每个 P3 LocCC 建立完整建筑卡，再分别判断 Memory 适合哪项任务。

完整建筑卡包含整栋/核心/边界/高变化/稳定区角色 token、P3 全局与关键实例表征、几何、P3 类别概率和训练期标签。建筑是否进入缓存只由 P3 LocCC 和面积决定，绝不使用 GT 纯度筛选。

两个任务完全分开：

1. `semantic_task`：整栋建筑检索，检查固定 25% Memory 后验能否改善 P3 损毁分类。
2. `utility_task`：先以完整父建筑路由，再用修改簇证据排序，检查能否判断 S2 修改保留或回退。

所有邻居强制来自不同事件。输出决策为：只适合语义、只适合 S2、两者都适合（但必须使用两个独立库）、两者都不适合。

开发阶段只允许 `meta_train -> meta_calibration`，禁止访问 meta_confirm、Val450 和 Test。

## E1：像素级写回

`run.sh e1` 在既有完整建筑卡上运行两部分验证，不重建缓存，也不训练新模型。

第一部分是 C0--C7 捷径对照：正常完整建筑检索、打乱标签、类别先验、仅几何、仅 P3 全局特征、仅 RGB、仅 SAR、RGB+SAR+角色 token。它用于确认建筑级结果确实来自跨模态完整建筑病例，而不是类别先验或几何捷径。

第二部分将同一个冻结 Memory 后验以 `beta=0.25` 写回语义图，比较：

- M0：P3；
- M1：P3 + fixed S2；
- M2：P3 + Memory severity-only；
- M3：P3 + Memory full-semantic；
- M4：fixed S2 + Memory severity-only（主实验）；
- M5：fixed S2 + Memory full-semantic。

Severity-only 严格保持来源预测的前景/背景掩膜，只在 Intact、Damaged、Destroyed 三类之间重新分配概率。主准入要求 M4 相对 M1 的 mIoU 至少提高 0.30pp、Damaged F1 不下降、Destroyed F1 下降不超过 0.10pp、至少 10/14 事件不下降、配对 bootstrap 置信区间下界大于零，并且定位掩膜逐像素完全不变。
