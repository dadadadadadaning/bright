# Memory设计范式向BRIGHT的迁移建议

## 用户问题

在强P3、14个跨事件域、Damaged类内异质且Mean Top-20召回高但安全决策失败的条件下，哪些Memory范式可以迁移？

## 四类范式与匹配度

1. 推理Key-Value Cache（RAC、Tip-Adapter）：任务邻近但数据假设不匹配。BRIGHT的事件偏移和Intact误报比标准长尾/少样本更强；只有独立验证器通过后才能使用残差融合。
2. Train-only动态队列（XBM、Class-wise Memory）：匹配度最高。应使用动量Teacher、FIFO、质量过滤、类别×事件上限和stop-gradient，替换静态一次性Key。
3. 多原型/层级代理（SoftTriple、HPL）：适合Damaged多形态和Occurrence/Severity层级，但训练前必须做原型稳定性与事件纯度审计。
4. 粗检索＋局部验证（CVNet、LOCORE、DMML）：与当前Top-20 Oracle证据最匹配。Mean保留为召回Key，局部Token/相关体只负责验证Rescue/Harm，不直接三分类。

## 最推荐方案

### V0-LTV：Local Token Verification Memory

Mean Key召回跨事件Top-20；从P3 Stage2/3提取建筑内局部Token；对每个Query-Support构造归一化相关矩阵，并按支持事件聚合；小型验证器输出“这次Non-Damaged→Damaged建议是否安全”。Train-LOEO训练和阈值锁定，Validation只执行一次。该方案借鉴CVNet的retrieve-then-verify，但目标改为救援/伤害而非通用图像同一性。

### M0-DQ：Dynamic Event-Class Queue

Memory仅用于训练。每类×事件维护FIFO队列，由EMA Teacher写入当前高质量区域Key；Damaged锚点只选择不同事件、相似度处于半困难区间的正样本，Intact负样本同样事件上限；推理删除队列。该方案综合XBM、类别Memory和STEAM的动量编码器思想。

### E0-HMP：Hierarchical Multi-Prototype Memory

先分Occurrence（Intact/Affected），再分Severity（Damaged/Destroyed）；Affected下设置多个Damaged代理。先冻结P3进行无训练聚类审计，要求每个代理跨至少5个事件、最大单事件占比不超过30%、多种子ARI稳定，再允许训练。

## 不建议直接迁移

- RAC/Tip-Adapter式Cache logits直接加到P3：RM6P已在当前Key上失败。
- 共享三类Proxy/Adapter：M1系列已证明会破坏Damaged或Intact/Destroyed。
- 大型LOCORE直接端到端训练：当前proposal数量和事件数有限，先从小型相关性验证器做起。

## 建议论文题目

1. Event-Robust Retrieve-and-Verify Memory for RGB–SAR Building Damage Assessment。
2. Dynamic Event-Class Memory for Hard Damaged Building Representation Learning。
3. Hierarchical Multi-Prototype Memory for Heterogeneous Disaster Damage Recognition。
