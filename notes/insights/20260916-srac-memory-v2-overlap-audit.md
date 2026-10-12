# SRAC 与 Superpixel Exemplar Memory V2 重合度及转向建议

- 用户研究方向：灾前 RGB、灾后 SAR 的建筑损毁评估；超像素区域检索与 P3 安全回写。
- 用户研究问题：当前 Memory V2 是否与 IGARSS 2026 SRAC 重复，是否构成抄袭，怎样验证并形成可独立投稿的贡献。
- 分析日期：2026-09-16
- 论文来源：[IGARSS 2026 accepted papers](https://2026.ieeeigarss.org/papers/accepted_papers.php)、[中国科学院大学学院新闻](https://emse.ucas.ac.cn/index.php/zh/xwzxe/535-igarss-2026)
- 趋势来源：[DrsNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC7988710/)、[CN109360190A](https://patents.google.com/patent/CN109360190A/zh)、[IEEE author ethics](https://conferences.ieeeauthorcenter.ieee.org/author-ethics/ethical-requirements/)

## 证据边界

已核实 SRAC 被 IGARSS 2026 接收，公开介绍称其使用 VMambaSSN 生成超像素，并利用外部标注库检索强化稀有损毁类别，在 BRIGHT 上报告 75.65% mIoU。当前未检索到可公开核对的全文、公式或官方代码，因此不能判断其具体 Key/Value、训练损失、检索规则和像素回写是否与 Memory V2 相同。

## 论文可迁移模块与重合审计

| 模块 | SRAC公开描述 | Memory V2 | 判断 |
|---|---|---|---|
| 任务与模态 | RGB–SAR多模态BDA | 相同 | 高重合 |
| 数据集 | BRIGHT | 相同 | 高重合 |
| 区域单位 | VMambaSSN学习超像素 | 灾前RGB SLIC超像素 | 概念重合、实现不同 |
| 外部知识 | 标注库检索 | train-only exemplar Memory | 高重合 |
| 目标 | 稀有严重损毁、结构一致性 | 跨事件损毁检索、P3漏检修正 | 高重合 |
| 区域描述 | 未公开 | 局部容错RGB–SAR关系、核心/外围、边界、形状、SAR分位数 | 尚不能判断，可能差异点 |
| 检索训练 | 未公开 | CE+软标签+跨事件SupCon | 尚不能判断 |
| 使用方式 | 检索增强分类 | 计划作为不确定性门控Verifier，失败回退P3 | 潜在关键差异 |
| 鲁棒验证 | 未公开 | 同图/同事件排除、随机标签、leave-event、错位审计 | 潜在关键差异 |

## 匹配与可行性评估

- 收录信息：SRAC已列入IGARSS 2026官方接收名单和会议日程。
- 代码情况：本次检索未发现官方公开代码；不能据此断言永远没有代码。
- 任务匹配程度：极高。
- 当前创新度：若将“超像素+标注Memory+损毁分类”作为主贡献，创新度偏低且会被SRAC直接覆盖；若转为“配准不确定性、跨事件风险与选择性回退”，仍有形成独立贡献的空间。
- 迁移难度：中等；已有检索基础，但缺少像素安全回写、直接分类器对照和真正LOEO验证。
- 主要风险：最近邻工作重合、Noto跨事件崩溃、纯净GT超像素审计偏乐观、Memory不一定优于参数分类器、Test提前使用。

## 抄袭判断

仅仅独立使用相同的通用组件不自动构成抄袭；未注明来源地复用对方的文字、图、代码、公式、具体算法流程或将已知核心思想声称为自己的首创，才会产生严重学术伦理风险。现在已有公开的SRAC先行工作，后续论文必须明确引用并把它列为最接近工作。应保存代码提交、实验日志和研究笔记的时间记录，但时间记录不能替代引用，也不能解决创新性重合。

## 落地建议

1. 停止将“Superpixel Retrieval-Augmented Classification”作为自己的核心命名和主创新。
2. 将SRAC纳入最强相关工作/基线；在获得全文后逐项核查超像素生成、Key/Value、检索、损失、回写和评估协议。
3. 把主问题改为：像素模型何时应信任跨事件区域Memory，以及错误Memory如何安全回退。
4. 第一阶段仅修正Occurrence，不用Memory硬判Damaged/Destroyed；采用P3不确定性、检索margin、邻居一致性和来源事件多样性联合门控。
5. 增加直接分类头、随机标签、随机近邻、同/跨事件Memory、错误类别Memory、无门控回写、规则网格/SLIC等对照。
6. Val冻结后才能一次性运行Test；Test不得用于阈值选择。

## 最小可行实验

- R0：Metric Encoder直接分类头，无Memory。
- R1：正确Memory裸检索。
- R2：类别数保持不变的随机标签Memory。
- R3：排除同事件Memory。
- W0：冻结P3。
- W1：P3+直接分类头残差。
- W2：P3+Memory，无门控。
- W3：P3+Memory+不确定性/检索双门控。
- W4：W3+来源事件多样性门控。
- W5：随机Memory+相同门控。

准入：W4相对P3的Val mIoU至少+0.30pp，Damaged F1不下降，至少10/14事件不下降，bootstrap CI下界大于0，且W4显著优于W1和W5。

## 建议题目方案

1. **Risk-Calibrated Cross-Event Memory Verification for RGB–SAR Building Damage Assessment**
   - 核心：把Memory作为可拒绝的Verifier，而不是分类器。
   - 实验：跨事件排除、错误Memory、coverage-risk、P3回退。
   - 贡献：检索风险控制与可验证的非伤害机制。
2. **Alignment-Uncertainty-Aware Regional Retrieval for Optical–SAR Damage Assessment**
   - 核心：局部容错对齐、核心/外围关系和错位不确定性进入检索键。
   - 实验：0/4/8/16像素位移、SAR打乱、无对齐描述符对照。
   - 贡献：解决RGB–SAR配准残差，而不是泛化的超像素检索。
3. **Event-Diverse Memory Routing for Robust Building Damage Segmentation**
   - 核心：限制单一来源事件支配Top-k，并在事件外推风险高时拒绝。
   - 实验：Noto/Ukraine诊断、来源事件熵、LOEO或选定困难事件留出。
   - 贡献：事件捷径检测和跨事件路由。
4. **Metric-Aligned Selective Memory Correction for Pixel-wise Building Damage Models**
   - 核心：依据修改簇的真实mIoU效用训练接受/拒绝器。
   - 实验：直接检索分类与metric-aligned verifier、oracle recovery、拒绝率。
   - 贡献：把检索正确率转化为安全的像素指标改进。

## 证据

| 类型 | 来源 | 支撑内容 |
|---|---|---|
| 官方接收名单 | IGARSS 2026 | SRAC题目和接收事实 |
| 机构新闻 | 中国科学院大学 | VMambaSSN超像素、外部标注库、BRIGHT 75.65% mIoU等公开描述 |
| 既有研究 | DrsNet | 超像素、检索与稀有类别并非全新组合 |
| 早期专利 | CN109360190A | RGB/SAR建筑损毁中的超像素融合已有更早先例 |
| 伦理规范 | IEEE Author Center | 使用他人思想、过程、结果或算法必须明确归因 |
