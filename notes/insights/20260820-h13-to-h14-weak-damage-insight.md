# 从 H13-SPER 到 H14-UWDC：弱损毁证据迁移设计

- 用户研究方向：灾前 RGB + 灾后 SAR 的建筑三级损毁评估
- 当前问题：H13 区域聚合略升总体 mIoU，但降低 Affected Recall、增加 Damaged gate-miss，且仅 5/14 事件非负
- 分析日期：2026-08-20
- 论文来源：HeteCD（ISPRS JPRS 2025，官方代码）；FlowMamba（IEEE TCSVT 2025，官方代码）

## 诊断结论

H13 的区域划分 Oracle 超过 99%，但 S1/S2 的训练收益几乎为零，说明瓶颈不是区域边界。当前区域头只使用 `stage2_rgb_common` 与 `stage2_sar_common`，并把基础模型的 occurrence 均值作为节点输入；这容易复用 H6-A2 的保守判断，而没有显式利用 `rgb_unique/sar_unique` 中可能保留的模态特有弱损毁证据。S1 与 S2 几乎相同，也否定了继续增加超像素图传播的必要性。

## 可迁移模块

| 模块 | 来源 | 迁移方式 | 适配条件 |
|---|---|---|---|
| 非共享模态编码与差异挖掘 | HeteCD | 保留 H6-A2 非共享编码；使用 Stage2/3 common gap 与两侧 unique 特征 | 不做全图一致性对齐，不直接复制二值 3D 模块 |
| 多尺度损毁表征 | HeteCD / FlowMamba | Stage3 上采样至 Stage2 后进行轻量融合 | 只进入 Intact/Affected 边界，不修改 Damaged/Destroyed 比值 |
| 类别条件监督思想 | FlowMamba | 用一次 error-conditioned occurrence BCE 替换通用层级 NLL | 不增加损失项；Damaged→Intact 漏检赋更高权重 |

## 建议模型：H14-UWDC

1. H6-A2 全部冻结，继续作为安全基线。
2. 删除超像素划分、区域均值/最大值和图传播。
3. Stage2/3 分别构造：`|rgb_common-sar_common|`、common product、`rgb_unique`、`sar_unique`；unique 特征只拼接，不假设两者可逐通道直接相减。
4. Stage3 上采样到 Stage2，使用小型卷积融合，输出有符号 occurrence residual。
5. 从证据编码器中移除 base occurrence，避免捷径；base occurrence 仅在最后被残差修正。
6. 使用完整的软 RGB building probability 约束残差，不腐蚀建筑边缘。
7. 保持零初始化安全接入，且给 Damaged/Destroyed 同加残差，严格保持严重度赔率。
8. 用单一 error-conditioned weighted BCE 替换现有层级 NLL：重点监督真实 Damaged 但基线预测 Intact 的像素。

## 实验链

- U0：精确 H6-A2。
- E0：验证集无训练审计 common gap 与 unique 特征对“Damaged 被预测为 Intact”的 AUROC/AUPRC及逐事件一致性。
- U1：Stage2 common+unique，保持普通 occurrence BCE。
- U2：U1 + error-conditioned weighting（替换，不叠加损失）。
- U3：仅 U2 通过后加入 Stage3 自顶向下融合。

准入门槛沿用：Damaged IoU +0.30 pp、Affected Recall +0.50 pp、gate-miss -0.50 pp、event mean/worst 不下降、至少 10/14 事件非负。验证候选锁定前不得访问测试集。

## 风险与创新判断

- 创新度：中等。创新不在“使用 unique 特征”，而在针对异构 RGB-SAR 三级损毁，把一致性分量用于稳定上下文、模态特有分量用于弱损毁纠错，并通过 occurrence/severity 解耦只修正第一层漏检。
- 主要风险：unique 分量可能主要编码 SAR 材料和入射角，而非损毁；因此 E0 审计必须先于训练。
- 迁移难度：中等。H6 已输出 Stage2/3 common/unique，新增代码主要是轻量证据金字塔、一次损失替换和独立评估。

## 建议题目

1. Modality-Unique Weak-Damage Correction for Heterogeneous RGB-SAR Building Damage Assessment
2. Event-Robust Occurrence-Severity Decoupling for Optical-SAR Damage Mapping
3. Error-Conditioned Multi-Scale Difference Mining for Weak Building Damage Recognition

## 证据

- HeteCD 官方仓库与论文信息：https://github.com/weiAI1996/HeteCD
- FlowMamba 官方仓库与论文信息：https://github.com/flying318/FlowMamba
