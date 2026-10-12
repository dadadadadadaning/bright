---
paper_id: SRAC-IGARSS2026
title: "SRAC: Superpixel Retrieval-Augmented Classification for Multi-Modal Disaster Building Damage Assessment"
year: 2026
authors: "Siheng Ning, Yuan Li, Xiaodan Zhang, Jianbin Jiao"
venue: "IGARSS 2026"
url: "https://2026.ieeeigarss.org/papers/accepted_papers.php"
pdf: "未获取"
code: "未证实"
status: "非全文精读"
---

# SRAC: Superpixel Retrieval-Augmented Classification for Multi-Modal Disaster Building Damage Assessment

## 基本信息

| 字段 | 内容 |
|---|---|
| 标题 | SRAC: Superpixel Retrieval-Augmented Classification for Multi-Modal Disaster Building Damage Assessment |
| 作者/团队 | Siheng Ning, Yuan Li, Xiaodan Zhang, Jianbin Jiao |
| 发表年份 | 2026 |
| 收录于 | IGARSS 2026，录用名单 paper 2851 |
| 关键词 | multimodal BDA, superpixel, retrieval-augmented classification, long-tail recognition |
| PDF | 未获取 |
| 代码 | 未证实 |

## 论文主要研究问题

- 面向灾前光学与灾后 SAR 的建筑损毁评估，处理跨模态差异、配准残差、像素预测碎片化和稀有严重损毁类别。
- 证据：用户提供的实验室新闻稿；用户提供的方法总览图。

## 使用的方法或模型

- Stage 1：VMamba Backbone、UNet++ 与 differentiable SSN 生成可学习超像素。
- Stage 2：通过 building mask、background dimming、superpixel boundary highlighting 注入结构先验，再用 VMamba Backbone 提取像素特征。
- Stage 3：空间注意力加权超像素池化；主投影路径与 mean/std raw skip path 形成超像素嵌入。
- Stage 4：从 memory bank 和 semantic-neighbors table 检索 Top-R 支持超像素；注入标签、灾害类型、区域和几何元数据；用 cross-attention 与 FFN 分类。
- 证据：用户提供的方法总览图。由于未获取全文，训练细节、损失、数据划分和各模块消融均未证实。

## 数据集与结果

| 数据集 | 用途 | 规模/设置 | 原文证据 |
|---|---|---|---|
| BRIGHT | 多模态建筑损毁评估 | 具体划分、训练协议和指标口径未获取 | 新闻稿称 75.65% mIoU、相对所用基线 +8.08 pp |

## 核心发现

1. 新闻稿报告 SRAC 在 BRIGHT 上达到 75.65% mIoU，并称相对主流基线提高 8.08 pp。
   - 证据：实验室新闻稿。
2. 方法图显示提升机制不是单独的超像素平滑，而是“可学习结构分区 + 超像素嵌入 + 标注记忆检索 + 元数据条件分类”的完整系统。
   - 证据：用户提供的方法总览图。

## 主要局限与待核验项

- 正式 PDF、消融表、损失函数、memory bank 构造、训练/验证/测试划分均未获取。
- 75.65% 的确切指标定义、是否为官方 test、所比较基线及其数值需要全文确认。
- Memory bank 是否仅由训练标注构建、是否使用目标域或验证/测试信息，未证实。
- 官方代码可用性未证实。

## 原文证据索引

| 位置 | 证据内容 | 支撑结论 |
|---|---|---|
| IGARSS 2026 accepted papers list | paper 2851 标题 | 录用事实 |
| 实验室新闻稿 | 研究动机与 75.65% / +8.08 pp | 摘要级结果 |
| 用户提供的方法图 | Stage 1-4 结构 | 模块级结构解读 |
