# 区域退化、建筑约束与Memory：证据与最小改进路线

日期：2026-10-03。研究对象：BRIGHT RGB–SAR建筑损毁评估，DamageFormer/Swin四阶段改编。依据本地fourstage_v2代码与用户提供的step3000、Val450审计；未访问服务器，未修改训练代码。

## 已确认的结果

- pixel 60.8998%；normal区域59.1447%；off区域60.3895%。固定Q关闭图像调制恢复1.2449pp。
- Damaged IoU normal37.9269%→off42.8333%；但Background、Intact、Destroyed相对normal下降，不能称全面改善。
- Damaged中pixel正确→region Intact：normal498442，off77279。不能由此断言唯一原因是纹理抹除。
- 低纯度桶的伤害率14.64%，高纯度桶0.277%；属于关联，类别、边界及难度可能混杂。
- 原像素概率经Q投影60.8982%接近pixel，不能证明所有区域汇总无损，也不支持将整体下降全归因于Q。

## 代码核验

model.py使用RGB Swin多尺度特征、嵌套解码器、三轮局部中心更新产生Q。传入全零building，因此没有建筑/背景分组。RGB及XY决定直接归属，SAR仅经损毁任务梯度间接影响分区。

建筑概率来自定位头；图像调制为raw*(0.1+0.9B)，再在硬Q边界处加入0.15*(255-dimmed)。这是提亮超像素边界，不是真实建筑/损毁边界监督。B与边界在此路径detach。

调制RGB再次通过共享RGB编码器及融合层，与SAR融合；注意力Pooling和mean/std旁路产生128维Token；线性四类头后按Q回投。区域头没有额外的区域间self-attention。最终region不与pixel残差相加。

train_regions.py目标为定位CE+Lovasz、区域NLL+Lovasz、0.25像素CE+Lovasz；全部前端联合更新。数值系数不等于实际梯度贡献比例。best按region验证mIoU选，不按pixel选。entropy是日志不是损失。正式50k、EBS16、base1e-5/new1e-4、1k warmup后cosine。实际进程当前步数需服务器日志。

train_retrieval.py单独冻结前端，训练memory/nomemory分类头，面积加权软区域标签CE，完整图像Val选权重。Memory不会自动随regions训练启用，也不会在冻结阶段重塑Q。当前V2支持标签、事件/灾害/几何元信息，不应套用旧V1缺元信息的描述。

## 论文模块与适配

|论文|可借鉴内容|匹配及风险|
|---|---|---|
|SSN，ECCV2018|可微聚类、任务重建与紧致度目标|分区机制匹配高；紧致度过强会切断细小结构，不能直接照搬权重。官方论文可获取，当前代码非官方复现。|
|ChangeOS，RSE2021|建筑对象定位与对象级语义约束|BDA任务匹配高；双光学到RGB–SAR有模态差异；连通域不能直接等同实例。作者代码公开。|
|Superpixel Transformers for Efficient Semantic Segmentation，IROS2023|局部cross-attention聚合、区域self-attention、区域预测回投|分割结构匹配高；并非BDA验证；不依赖外部Memory，额外上下文需容量对照。代码发布状况本次未核验。|
|SPFormer: Enhancing Vision Transformer with Superpixel Representation|内容自适应中间特征表示|Transformer匹配；原图分类结果不能外推BDA。arXiv及TMLR版本可检索，注意不同同名论文。|
|SRAC，团队IGARSS2026录用新闻|分区—区域表示—历史检索整体框架|多模态BDA任务匹配；本次未获得完整论文与官方代码，不能核定训练预算、损失、无Memory消融或声称75.65可直接复现。|

## 最小落地建议

1. 不修改运行中的模型，不新增长训练。保留同一快照，补充off与normal按类别×混合程度的救回/伤害统计。
2. 同一A1及新模块初始化对照normal与off短程3k–5k；使用同一50k调度前缀，不能一个5k余弦另一个50k余弦。先核验实现，不把当前固定协议入口当作已有off训练开关。
3. 在off条件下再比较无分组/预测建筑背景分组，Pooling和回投均隔离；保持背景可预测，不用GT掩膜前向。匹配实际区域预算，审计漏检建筑。
4. 不一开始强制每建筑一个Token；对象内多子区域，外部上下文作为独立证据保留。监督一致性必须与BRIGHT标签粒度一致，不能声称数据必有建筑内细粒度GT。
5. 若pixel退化仍在，比较A1单独微调与联合训练，再单独测试像素辅助权重0.25/1.0。已有残差T3未超T0，不能重新把残差当成保证。
6. Memory可在冻结同一前端后做一次memory/nomemory匹配预算对照以回答交互问题，无需先假设区域必须超过pixel；但有效增益需同时对照同容量头和A1。随机邻居/支持标签打乱仅推理依赖检查。

## 三个可执行研究题目候选（假设，不是已成立创新）

1. BRIGHT跨模态建筑损毁评估中的结构调制失配分析：normal/off及类别转移对照，目标解释信息保留与类别取舍；成本低，创新主要在可重复机制证据。
2. 建筑概率约束的多子区域损毁表示：off基线加入预测建筑背景分组，区域预算匹配，目标减少混合伤害；成本中，先排除只是Token数量增加。
3. 区域纯度与跨事件检索可靠性的联合评估：冻结同前端，memory/nomemory及干预、分纯度召回统计；成本中，测试Memory是否帮助困难区域而非仅加强多数类。GT纯度仅分析，不能作为Val/Test检索输入。

上述方向均不以堆模块为新颖性依据；需要真实增益、多种子及公平协议。阶段式诊断优先于同步加入新损失、实例分区和Memory。

## 原文来源

- SSN： https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.pdf
- ChangeOS论文： https://www.sciencedirect.com/science/article/pii/S0034425721003564
- ChangeOS作者代码： https://github.com/Z-Zheng/ChangeOS
- Superpixel Transformers： https://arxiv.org/abs/2309.16889
- SPFormer： https://arxiv.org/abs/2401.02931
- SRAC团队消息： https://jianghai.ucas.ac.cn/index.php/zh/xshde/xsdte/461-igarss-2026
