# Memory作用、替代路线及最新探针对照

研究方向：BRIGHT RGB–SAR建筑损毁评估。日期2026-10-03。证据为用户最新附件0c900dd4与本地V2检索代码；未修改运行代码，未执行服务器训练。

## 最新效果

附件最后2400/5000步，最后完整Val为2000步。新pixel/newregion：500步57.5641/58.0492，1000步58.2731/58.8241，1500步58.5037/58.9290，2000步58.6676/59.1218%。区域领先0.4253–0.5510pp。2000步距旧region59.1447仅-0.0229pp，距旧pixel60.8998为-1.778%。两新头共享同一冻结调制特征和Q；旧pixel输入路径不同。中期单种子证据，不代表统计显著或已超过原A1。新区域是均值Pooling，不是原attention+stats。没有Memory参与本轮训练。

## Memory目的与限制

当前V2属于外部Train样本检索：按特征余弦选邻居，支持Token、软标签和灾害/事件/几何元信息通过Cross-Attention辅助分类。训练前端冻结；不是修复分区、配准或恢复丢失像素细节。每事件×多数类reservoir有配额，但全局Top-k不保证查询得到类别均衡支持。支持头训练为先采图再采区域、面积加权软标签CE，与前端像素NLL+Lovasz不同；不能把旧region到Memory的全部变化归因历史证据。公平主要比较同前端、同损失、同预算、同参数的memory/nomemory。

## 原文可迁移模块

|来源|机制|适配与风险|
|---|---|---|
|Rethinking Semantic Segmentation: A Prototype View，CVPR2022|每类多个训练特征原型，非参数分类；原文提供代码链接|可将样本库压缩为多原型，成本低中，适合磁盘紧张；原型可能丢失罕见形态，冻结后聚类是简化探针不是严格复现|
|Exploring Cross-Image Pixel Contrast，ICCV2021|跨图同类拉近、异类拉远，塑造特征空间|改进Damaged/Intact判别；推理可不查库，但训练需更新表示，不能只加损失到完全冻结特征；不是本次固定头实验直接替代|
|Object-Contextual Representations，ECCV2020|从当前图像汇总语义区域并回送像素上下文，作者实现公开|避免外部库，保留像素预测；不提供新历史案例，错误初始语义可能自我强化；成本中|

## 最小建议

先完成现有探针。若验证历史信息，固定同一前端和Token版本，比较无Memory头、轻量标签加权kNN诊断、原样本Memory头。必要时以每类多原型压缩作为第四组，不改分区/调制/损失。kNN失败不能证明attention必失败，仅反映当前距离结构。使用相同候选排除策略，训练原型/索引不含Val/Test；训练查询应排除自身及按协议排除同事件贡献，可用leave-event-out原型。验证看类均衡检索命中与软标签一致性、不同支持图像/事件数量、各类别救回伤害及最终全图mIoU。不要以检索同类率取代分割指标。

若Memory有问题：先判断检索是否语义有效；语义无效再学投影/对比，而非堆库。若邻居可靠但输出变差，再逐项去掉元信息、测试有无标签、支持融合可靠性。事件嵌入可能成为捷径；当前同事件排除并不自动消除事件风格偏差。不能按Val查询真标签选库/选Top-k/判定是否调用Memory。

## 三个候选题目（待验证）

1. BRIGHT多原型区域检索分类：从Train按类构建多个原型，对照同容量头和真实样本检索，检验存储/精度权衡。组合本身新颖性有限，需效率和机制结果。
2. BRIGHT跨事件损毁判别嵌入：对Damaged/Intact加入跨事件监督对比，同预算对照，目标改善语义邻居而非事件相似。成本中，勿同步改变采样/损失多个因素。
3. BRIGHT无外部库语义上下文增强：借鉴OCR在原始融合特征上构建当前图像上下文，与A1同容量适配器比较；目标保留像素细节，不预设能解决长尾。

## 证据来源

- https://arxiv.org/abs/2203.15102
- https://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Rethinking_Semantic_Segmentation_A_Prototype_View_CVPR_2022_paper.pdf
- https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Exploring_Cross-Image_Pixel_Contrast_for_Semantic_Segmentation_ICCV_2021_paper.html
- https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/5021_ECCV_2020_paper.php
- 本地fourstage_v2/build_memory.py、retrieval.py、train_retrieval.py。

这些论文是机制参考，不是已在用户BRIGHT协议验证的改进。除Prototype原文和OCR此前作者库外，本次没有额外核验各方案软件环境兼容性。
