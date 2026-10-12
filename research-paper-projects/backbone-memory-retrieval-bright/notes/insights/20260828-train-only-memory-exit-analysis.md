# BRIGHT Memory 路线出口分析

## 当前判断

推理时 Memory 直接改写 P3 预测已经被本项目数据否定：检索能救回少量 Damaged，但无法可靠拒绝 Intact 误报。该版本应停止。Train-only Memory 尚未被否定，因为 H1 还没有进入一次有效优化；当前阻塞来自区域回放契约和服务器代码版本，而不是验证指标。

## 可迁移论文模块

1. XBM：动态队列保存近期训练嵌入，跨批次挖掘困难负样本；适合替换静态旧特征缓存。
2. 类别级像素 Memory：只保存高质量标注特征，持续更新，测试时移除 Memory。
3. OHEM / JTT：先由基线定位当前困难样本，再直接重加权主任务损失；不强迫 Damaged 成为单一嵌入簇。
4. Group DRO：按事件×类别控制最差组损失，直接对应 BRIGHT 的跨事件不稳定。
5. 非各向同性/多原型思想：Damaged 是多模态类，若再做度量学习，应保留多个局部模式。

## 推荐的最小实验链

### H1-OHEM0（第一优先）

使用 H0-B 的事件上限 D→I 锚点，但不读取跨事件正样本 Key，不训练 projector。只对当前 Student 的 Intact/Affected 发生分数施加排序或困难 BCE/Focal 损失，并对原本正确的 Intact/Destroyed 使用 Teacher 蒸馏保护。该对照回答“收益来自困难样本重采样，还是来自 Memory 相似关系”。

### H1-Mem+（仅当 OHEM0 有效）

在完全相同锚点与训练权限上，增加动态类别/事件队列的跨事件 Damaged 支持。队列使用 stop-gradient、定期刷新和半困难负样本，避免静态旧 Key 与当前 Student 漂移。

### H1-Group（独立稳健性对照）

按事件×Occurrence 类别构造训练组，优化最差组或进行最后分类头重加权。该实验不使用检索，可判断 Memory 是否真的优于更直接的事件鲁棒训练。

## 暂停项目

- 推理期 kNN、Cross-Attention、阈值扫描：已有数据直接否定。
- 共享全局 Embedding Adapter：反复以 Intact/Destroyed 换取 Damaged。
- 将 IM4CD、超像素、跨模态对齐同时并入 H1：会破坏单变量归因。IM4CD 若继续，应单独立项。

## 关键工程修改

不要在每个训练步重新以连通域启发式匹配缓存建筑。预先把 Teacher 建筑掩膜压缩为 RLE/行区间 sidecar，并记录图像、bbox、面积、组件哈希。训练时使用固定掩膜池化 Student 特征；若进行几何增强，对图像与掩膜执行同一变换。这样把“重复猜建筑组件”改成“按不可变ID读取训练目标”。
