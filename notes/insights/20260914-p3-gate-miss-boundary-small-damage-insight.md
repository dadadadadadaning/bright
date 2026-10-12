# P3 漏判归因后的主网络设计建议

- 用户研究方向：灾前 RGB 与灾后 SAR 的建筑损毁评估（四类语义分割）。
- 当前问题：P3 的 Affected 漏判在建筑边界、小面积损毁与高 SAR 局部 MAD 区域富集；fixed S2 的修改簇准确但仅覆盖少量漏判。
- 证据来源：用户给出的 `SPI_P3_GATE_MISS_SPATIAL_AUDIT_V1` meta-calibration 输出。

## 可迁移模块

| 模块 | 原始工作 | 迁移方式 | 适配条件 |
|---|---|---|---|
| 双时相、多尺度特征交互 | ChangeFormer | 保留 DamageFormer 主干，新增轻量 Stage2/Stage1 局部变化分支；不更换整个主干 | 不把 RGB–SAR 当作同分布影像直接相减 |
| 全局—局部联合变化建模 | CDMamba / ChangeMamba | 用现有 P3 深层特征提供上下文、以高分辨率卷积处理边界和小损毁 | 先以局部 Head 验证，不直接替换为 Mamba 主干 |
| 灾前信息作为结构锚点 | MapFormer | RGB 屋顶/边缘特征为查询锚点，SAR 只提供质量门控后的变化证据 | 不要求 RGB 与 SAR 像素外观一致 |
| 质量感知融合 | CAFGFNet 的方向 | SAR 局部 MAD 生成连续可靠性门，对 SAR 变化支路做降权 | 仅在高 MAD 风险经开发审计确认后启用 |

## 推荐的唯一主方案

题目：**Boundary- and Quality-Conditioned Hierarchical Affected Occurrence Decoder for RGB–SAR Building Damage Assessment**

1. P3 的 RGB Stage2 高分辨率结构特征进入边界增强卷积支路。
2. P3 的 SAR Stage2 特征进入局部变化支路；局部 MAD 只作为连续门控，不能作为硬阈值删除 SAR。
3. 两支路预测 Affected occurrence logit；该 logit 只改变 Intact 与 Affected 的边界。
4. 原始 P3 的 Damaged–Destroyed 相对 logit 差保持不变；以层级概率组合恢复四类输出。
5. fixed S2 修改簇只作为停止梯度的高置信提示通道，且必须提供“无簇提示”消融；不得将 S2 类别广播为监督标签。

## 必经审计与实验

1. **D1 overlap audit**：计算 `ring × small-damage × high-SAR-noise × S2-change` 的 P3 漏判交集。若 ring 与 small-damage 高度重叠，用一个边界—小损毁分支；若高 SAR 噪声主要独立，才加入质量门控。
2. **M0 identity guard**：新增 Head 的残差系数为零时，输出必须逐比特等于 P3。
3. **M1 pilot ablation**：只训练 occurrence Head；比较 P3、边界支路、边界+小损毁损失、边界+质量门控。所有损失权重在 meta_train 冻结。
4. **M2 causality**：真实 MAD 门控、恒等门控、随机质量图三者比较；真实修改簇提示、零提示、打乱提示三者比较。
5. **M3 evaluation**：仅在 meta_calibration 报告 valid-mIoU、Damaged/Destroyed F1、边界 Affected Recall、小损毁 Affected Recall、事件均值与 bootstrap CI。通过后才允许冻结模型并进行一次 Test。

## 风险

- 边界与小损毁可能是同一批像素；先看交集，不能把风险倍数相加。
- S2 修改簇覆盖率仅约 7.6%，不应作为主分支输入或硬掩码。
- 高 SAR MAD 可能也反映真实灾后散射变化；门控必须连续、零初始化，不能把高 MAD 直接当噪声抹除。

## 证据与趋势来源

- ChangeFormer：双时相 Siamese Transformer 与开源实现，[GitHub](https://github.com/wgcban/ChangeFormer)，[论文](https://arxiv.org/abs/2201.01293)。
- ChangeMamba：为 BCD、SCD 和 BDA 设计时空关系建模；其公开仓库说明包含 BDA 训练代码，[GitHub](https://github.com/ChenHongruixuan/ChangeMamba)，[论文](https://arxiv.org/abs/2404.03425)。
- CDMamba：主张将局部卷积细节与全局状态空间建模联合用于变化检测，[论文](https://arxiv.org/abs/2406.04207)。
- MapFormer：利用灾前信息增强变化检测，[论文](https://openaccess.thecvf.com/content/ICCV2023/papers/Bernhard_MapFormer_Boosting_Change_Detection_by_Using_Pre-change_Information_ICCV_2023_paper.pdf)。
- CAFGFNet：光学—SAR 双分支、频域门控和 SAR 去噪/几何形变缓解的近期方向；这里仅借鉴“质量感知融合”的思路，不迁移其完整网络，[IEEE Xplore](https://ieeexplore.ieee.org/document/11340679/)。
