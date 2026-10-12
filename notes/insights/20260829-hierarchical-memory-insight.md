# 分层原型与实例 Memory 对 BRIGHT RGB–SAR 损毁评估的启发

- 用户研究方向：灾前 RGB、灾后 SAR 的建筑损毁评估
- 用户研究问题：如何用分层原型与实例库先判断 Intact/Affected，再判断 Damaged/Destroyed，并避免误伤 Intact
- 分析日期：2026-08-29
- 论文来源：ASGNet（CVPR 2021）、Prototype View（CVPR 2022）、Pin the Memory（CVPR 2022）、Prototype Memory Network（WACV 2023）
- 趋势来源：CVF Open Access 与论文原文页面

## 论文可迁移模块

| 模块 | 原文依据 | 可迁移方式 | 适配条件 |
|---|---|---|---|
| 多原型而非单均值 | ASGNet指出单原型难以覆盖对象内部多样性 | Intact/Affected各自保存多个模式原型 | 原型必须跨事件混合，不能退化成事件簇 |
| 查询引导的原型分配 | ASGNet的Guided Prototype Allocation | 根据Query决定读取哪个Occurrence原型 | 只在训练LOEO上选择参数 |
| 每类一组非参数原型 | Prototype View使用多个训练特征均值表示类别 | 原型只由Train区域建立 | Validation/Test不能更新原型 |
| 域无关Memory约束 | Pin the Memory强调域泛化Memory | 原型加入事件分布审计和跨事件约束 | 必须报告事件NMI、最大事件占比和LOEO结果 |
| 写入质量筛选 | Prototype Memory Network筛选有用原型并淘汰低质量原型 | 只写入定位可靠、纯度高、跨事件稳定的实例 | 不能只按分类置信度写入 |

## 匹配与可行性评估

- 收录信息：上述工作分别发表于CVPR 2021、CVPR 2022和WACV 2023。
- 有无代码：部分工作提供公开实现；本项目不应直接复刻其任务代码，而应迁移多原型、分配和质量控制思想。
- 任务匹配程度：方法层面中高；模态和类别定义不同，不能直接照搬相似度函数。
- 创新度：中高。Occurrence原型与Severity实例的非对称分层在当前RGB–SAR跨灾害设置中具有明确任务动机。
- 迁移难度：中高。主要难点不是Memory容量，而是事件去偏、Key解耦和拒绝校准。
- 可能问题：Affected仍包含多种灾害模式；Mean特征会形成事件簇；类别均衡Memory会改变真实先验；实例检索可能再次过预测Damaged。

## 落地建议

### 层级一：Occurrence Prototype Bank

目标只判断Intact/Affected。Key来自Stage2局部Occurrence Encoder，不使用旧Mean Key。每个原型保存中心、方差、事件直方图、支持数量和质量分数。通过事件上限采样拟合原型，并拒绝单事件占比过高的原型。

### 层级二：Severity Instance Bank

仅在Occurrence已经判为Affected后启用。使用更深层MICB/Critical局部Token保存Damaged/Destroyed实例。Mean负责粗召回，局部Token负责Top-k验证；这一层没有权限把Intact改成Affected。

### 层级三：拒绝机制

Memory必须同时满足原型margin、邻居标签一致率、跨事件覆盖和局部验证分数，否则返回P3。所有阈值在Train-LOEO上锁定，Validation只执行一次。

### 最小实验

1. 先完成P3-S2O 500步Screen，确认Stage2局部Occurrence表示存在增量。
2. O-M0：使用冻结S2O Embedding，无训练建立I/A多原型，审计Global/Event AUROC和事件主导。
3. O-M1：Train-LOEO锁定Occurrence拒绝点，检查救援/伤害净收益。
4. S-M0：在GT Affected的Train-only审计中比较D/X Mean与Critical实例Key。
5. H-M1：只有前两层分别通过后才进行层级组合，不直接训练Cross-Attention。

## 建议题目方案

1. 题目：Cross-Event Hierarchical Prototype–Instance Memory for Optical–SAR Building Damage Assessment
   - 核心思路：Occurrence使用跨事件多原型，Severity使用局部实例库。
   - 最小实验：与一步三分类kNN、单原型和无拒绝层级Memory比较。
   - 预期贡献：证明不同决策层需要不同Memory粒度和权限。
2. 题目：Event-Debiased Occurrence Prototypes for All-Weather Building Damage Recognition
   - 核心思路：通过事件上限拟合和LOEO约束学习域无关I/A原型。
   - 最小实验：报告原型事件NMI、事件主导率、Occurrence AUROC和逐事件结果。
   - 预期贡献：解决类别原型退化为灾害事件原型的问题。
3. 题目：Retrieve Globally, Verify Locally: Safe Damage Memory for RGB–SAR Assessment
   - 核心思路：Mean粗召回，局部Token验证，拒绝不可靠建议。
   - 最小实验：Top-20 Oracle、局部验证增量、risk–coverage与Intact误伤。
   - 预期贡献：将候选召回与安全决策显式解耦。

## 证据

| 类型 | 来源 | 支撑内容 |
|---|---|---|
| 原文 | [ASGNet, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Adaptive_Prototype_Learning_and_Allocation_for_Few-Shot_Segmentation_CVPR_2021_paper.html) | 单全局原型会产生歧义，多原型和查询引导分配可覆盖对象尺度与形态变化 |
| 原文 | [Rethinking Semantic Segmentation: A Prototype View, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhou_Rethinking_Semantic_Segmentation_A_Prototype_View_CVPR_2022_paper.html) | 可用每类多个训练特征原型进行非参数分类 |
| 原文 | [Pin the Memory, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_Pin_the_Memory_Learning_To_Generalize_Semantic_Segmentation_CVPR_2022_paper.html) | Memory需要抽象跨域稳定的类别信息而非域风格 |
| 原文 | [Prototype Memory Network, WACV 2023](https://openaccess.thecvf.com/content/WACV2023/html/Lee_Unsupervised_Video_Object_Segmentation_via_Prototype_Memory_Network_WACV_2023_paper.html) | Memory写入需要质量评分和有用原型筛选 |
