# SRAC 图示到 P3 的安全超像素迁移判断

## 问题与证据边界

- 目标：RGB 灾前图像与 SAR 灾后图像的 BRIGHT 像素级建筑损毁分割。
- 参考：SRAC（*Superpixel Retrieval-Augmented Classification for Multi-Modal Disaster Building Damage Assessment*）方法图、实验室新闻稿、IGARSS 2026 accepted-paper list。
- 证据边界：IGARSS 2026 录用事实可核验；论文全文与官方代码尚未获取。因此图中模块结构可以分析，但 SRAC 的损失、划分、Memory 建库规则和新闻稿中的指标口径均不能视为已证实。

## SRAC 的四阶段（由图示解读）

1. **可学习超像素**：灾前光学图进入 VMamba/UNet++ 与 differentiable SSN，得到超像素掩码。其职责是提供局部结构边界，不等同于建筑实例或损毁类别。
2. **像素表征**：光学和 SAR 分支分别提取特征；图中把建筑掩码、背景抑制和超像素边界作为结构先验。
3. **超像素嵌入**：对每个超像素内的像素特征做空间注意力加权池化；主投影与 mean/std 统计跳连共同形成区域向量。
4. **检索增强分类**：查询超像素在标注 Memory 中查 Top-R 邻居，再把邻居标签、灾害类型、区域和几何等元数据嵌入，通过交叉注意力和 FFN 输出分类。

## 为什么不能整套搬到 P3

既有实验已经显示：P3 的建筑/簇级 Memory 在推理期没有提供超出 P3 概率的独立信息，尤其无法稳定恢复 `Background/Intact -> Affected` 漏判；直接写回会降低最终像素 mIoU。故 SRAC 的 Stage 4 不是当前可迁移的首选。

此外，P3 漏判中约 32% 不属于边界、小损毁、高 SAR 噪声或 S2 簇。故超像素只能改善局部结构传播，不能替代全建筑 Affected Occurrence 主分支。

## 推荐迁移：只迁移 Stage 1--3 的“结构关系”

### 定位

`Superpixel Relation Adapter (SPR)`：超像素是由**灾前 RGB**生成的无类别局部关系图；P3 仍对每个像素输出四类 logits。不得把一个超像素命名为建筑、赋予多数损毁标签，或将区域类别广播写回。

### 最小模型

```text
灾前 RGB -> 固定 SLIC ID 图
P3 Stage-2 特征 + ID 图 -> 超像素池化 r_s
像素特征 f_i, r_s, f_i-r_s -> 小型 1x1 adapter -> Delta_i
f'_i = f_i + gamma * gate_i * Delta_i, gamma 从 0 开始
P3 原像素分类头 -> 四类像素输出
```

`gate_i` 应在预测建筑区内使用，并在超像素边界和 P3 高熵区域衰减。它不是语义答案，只决定局部关系信息是否可信。

### 训练与评估边界

- GT 只用作像素 CE/Lovasz、训练期小损毁权重与超像素质量审计；推理不得使用 GT 区域。
- 超像素 ID 图须和 RGB/SAR/label 做完全相同的裁剪、翻转、旋转，且只可 nearest-neighbor 变换。
- 先修正 AO-H-R1 的候选 token 监督；否则残差始终接近零，无法公平判断超像素信息是否有用。
- 固定 SLIC 比可学习 SSN 更适合第一版：成本低、变量少。只有固定分区与随机 ID/规则网格对照成立时，再考虑 SSN/SpixelFCN。

## 最小、可证伪的实验

| ID | 方法 | 验证问题 |
|---|---|---|
| SP0 | 冻结 P3 + AO-H-R1 | 基线 |
| SP1 | SP0 + 参数量匹配 1x1 residual | 排除“参数变多” |
| SP2 | SP0 + 固定 SLIC 关系 adapter | 局部结构关系是否有用 |
| SP3 | SP2 + 高可信区域一致性损失 | 一致性正则是否有额外价值 |
| SP4 | SP3 + 打乱 SLIC ID | 是否真正利用分组关系 |
| SP5 | SP3 + 规则网格 | 是否优于普通平滑窗口 |

先在 meta_train 拟合和冻结 SLIC 参数/损失权重，只在 meta_calibration 做一次内部评估；不访问 Val450/Test 调参。报告 mIoU、Damaged/Destroyed F1、Affected 漏判率、ring/small/noisy recall、14 事件指标和 bootstrap。

通过需至少满足：SP2>SP1，SP3>SP2，SP3 明显优于 SP4/SP5，且总 mIoU 与 Damaged F1 不牺牲。否则超像素只是在做普通平滑，不构成方法贡献。

## 可执行题目

1. **Boundary-Safe Superpixel Relation Adaptation for RGB--SAR Building Damage Segmentation**：固定灾前 RGB SLIC + 零门控关系残差；证明边界安全的区域上下文。
2. **Uncertainty-Gated Region-to-Pixel Occurrence Learning for Cross-Modal Building Damage Assessment**：全建筑 Affected Head 为主，超像素仅辅助局部关系传播。
3. **Pre-disaster Structural Superpixels for Small-Damage-Aware Cross-Modal Damage Mapping**：训练期小损毁重加权，推理期不含任何 GT 区域信息。

## 原始和趋势证据

- IGARSS 2026 accepted-paper list: https://2026.ieeeigarss.org/papers/accepted_papers.php
- SRAC 实验室新闻稿: https://jianghai.ucas.ac.cn/index.php/zh/xshde/xsdte/461-igarss-2026
- SSN (ECCV 2018): https://openaccess.thecvf.com/content_ECCV_2018/html/Varun_Jampani_Superpixel_Sampling_Networks_ECCV_2018_paper.html
- RegionProxy (CVPR 2022): https://openaccess.thecvf.com/content/CVPR2022/html/Zhang_Semantic_Segmentation_by_Early_Region_Proxy_CVPR_2022_paper.html
