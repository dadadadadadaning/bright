# SRAC图示与当前V2的逐阶段对照

研究问题：BRIGHT RGB–SAR建筑损毁评估中，当前DamageFormer四阶段改编与用户提供的SRAC图示有何不同。日期2026-10-03。

证据界限：图示直接可见四阶段模块；团队新闻报告75.65%mIoU与8.08pp增益，会议目录确认论文。此次未获得完整论文与官方实现，图中未说明的共享权重、损失、训练顺序、实例约束、评估协议不能补写为事实。

## 模块与匹配

1. Stage1 VMamba+UNet+++Differentiable SSN；V2共享A1 RGB Swin、小型嵌套解码、三轮局部软中心更新。思想匹配，骨干/解码容量/聚类公式及训练监督未证实一致。图并不能证明分区限制在单栋建筑内，亦不能证明一个超像素就是一个实例。
2. Stage2图示建筑掩膜背景压暗和超像素边界突出，然后编码；V2已经在原始RGB上调制并重新编码，不是早期V1仅特征层调制。V2只调制RGB，SAR保持原图编码；共享RGB编码器与fusion1/3。图里无法确认原方法是否同样共享、具体怎样融合光学与SAR。V2边界为硬Q argmax，B/E detach，0.1背景底值和0.15提亮强度为自定义。
3. Stage3图示注意力加权区域汇总及mean/std旁路；V2具备这些组成，128维、sigmoid注意力、可学习tanh alpha、LayerNorm等是具体自定义选择。图中E[w*x]归一化细节不能直接等同当前按权重质量归一化的Pooling。
4. Stage4图示Memory、预计算邻居表、MMAP Gather、支持标签/灾害/地域/几何元信息、Cross-Attention和FFN。当前V2已实现邻居mmap表及这些元信息的对应项，不能继续用旧V1“没有邻居表/元信息”的说法。地域用事件ID嵌入，不是真实地理坐标；几何为归一化面积、位置均值与标准差。Train查询用预计算表，评价检索支持在线余弦Top-k。同事件/图像排除是当前自定义协议，不知论文是否相同。

## 运行阶段与解释

- regions训练原前端、Q、调制、注意力与统计、简单区域头；没有历史检索。损失loc+region+0.25pixel，best按region选择。
- train_heads是固定快照的诊断：同一调制融合特征，独立新像素MLP/均值区域MLP；没有原注意力统计，也没有Memory；其58%左右不能叫完整SRAC改编最终结果。
- train_retrieval才冻结前端训练memory/nomemory头。原文是否同样分段训练未知。

最新证据：新region相对同容量新pixel在500/1000/1500步分别+0.4852/+0.5510/+0.4253pp。不能归咎所有区域汇总。固定Q关闭调制使原region59.1447→60.3895，Damaged改善但其他类别取舍存在。缺少完整Memory结果，不能提前归因其有效或无效。

## 迁移可行性、创新与风险

任务与模态匹配高；代码组成大致覆盖，数值等价未成立。改编成本中高；权重共享、SAR融合、标签/评估口径和Memory数据范围都是关键风险。替换VMamba为Swin或画出同样四框本身不是充分创新。75.65与当前Val前景三类指标需先核对split、类别定义、GT使用及支持库范围才能比较。

## 三个候选题目与最小实验

1. BRIGHT结构调制迁移失配研究：保持Swin和Q算法，normal/off匹配初始化与调度，检验Damaged伤害及整体取舍；贡献是机制证据，不预设涨分。
2. BRIGHT建筑约束多子区域表示：验证预测建筑背景分组并匹配Token预算；不宣称原SRAC具有实例限制；检查混合与漏检风险。
3. BRIGHT区域检索的独立贡献：冻结同前端，memory/nomemory匹配头及支持标签干预，验证额外历史证据而非容量增益；不使用Val/Test标签入库。

建议先完成现有5000步探针，再进行normal/off短程训练或同快照Memory对照，不改运行中的源代码。

来源：用户架构图；本地fourstage_v2/model.py、train_regions.py、retrieval.py、metadata.py、build_neighbors.py、train_retrieval.py与region_evidence_v1/train_heads.py。

- 团队消息：https://jianghai.ucas.ac.cn/index.php/zh/xshde/xsdte/461-igarss-2026
- 会议目录：https://2026.ieeeigarss.org/search.php?show=search
