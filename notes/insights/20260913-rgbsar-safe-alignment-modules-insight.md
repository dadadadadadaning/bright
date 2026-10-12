# 2024–2026模块迁移与RGB–SAR安全对齐方案

日期：2026-09-13

## 用户方向与核心问题

任务是BRIGHT建筑损毁分割，输入为灾前RGB与灾后SAR。当前SALT-D1约58.11% validation mIoU；SEC-S2在合成位移下能学到正确方向，并降低部分x8错位误差，但位移幅度响应过弱，同时Damaged F1下降约1.88pp、Affected Recall下降约4.52pp。问题不是“网络没有注意力”，而是对齐监督与损毁分类共同更新全网络，导致为了适应错位样本而破坏已经学到的类别边界。

## 本地模块检查结果

| 年份/模块 | 原始用途 | 匹配度 | 当前决定 |
|---|---|---:|---|
| 2026 CAFR / MODA | 多光谱检测中的物体引导特征细化 | 中 | 对齐稳定后再试；它不估计空间位移 |
| 2026 LWGA、Strip、MHCB | 遥感表征、条带卷积、全色锐化 | 低 | 暂不加入，不能直接解释RGB–SAR错位 |
| 2025 DSA | 可变形空间注意力 | 中高 | 借鉴显式采样位移；本地实现依赖额外DSCN算子，暂不直接复制 |
| 2025 Mona | 参数高效视觉适配器 | 高（训练策略） | 借鉴冻结主干、只训练小适配部分，不整套硬塞 |
| 2025 Geometry Self-Attention | RGB-D分割，用深度构建几何关系 | 低 | SAR强度不是深度，直接迁移会引入错误物理假设 |
| 2025 CMA、MSC | 通道/跨模态注意力、全局匹配 | 中低 | 能增强融合，但不能单独证明解决错位；MSC有显存风险 |
| 2025 Contrast-Driven Aggregation | 医学前景/背景差异聚合 | 中低 | D1已有建筑引导，新增价值有限 |
| 2024 ELA、CPAM、RCM、SSA/FSA | 通用局部/通道/恢复注意力 | 低 | 不进入当前主线，容易成为无法解释的堆模块 |

## 外部论文与代码证据

1. XoFTR（CVPRW 2024）采用粗到细匹配和亚像素细化，说明跨模态对齐需要明确对应关系与尺度层级。论文：https://openaccess.thecvf.com/content/CVPR2024W/IMW/html/Tuzcuoglu_XoFTR_Cross-modal_Feature_Matching_Transformer_CVPRW_2024_paper.html；代码：https://github.com/OnderT/XoFTR
2. MINIMA（CVPR 2025）强调精确对应标签与多模态覆盖，并通过合成数据保留匹配标签，支持当前“人为平移产生精确位移监督”的方向。论文：https://openaccess.thecvf.com/content/CVPR2025/html/Ren_MINIMA_Modality_Invariant_Image_Matching_CVPR_2025_paper.html；代码：https://github.com/LSXI7/MINIMA
3. HOMO-Feature（ICCV 2025）使用主方向等模态相对不变信息，提示后续可用梯度方向/结构边缘构建轻量匹配证据。论文：https://openaccess.thecvf.com/content/ICCV2025/html/Gao_HOMO-Feature_Cross-Arbitrary-Modal_Image_Matching_with_Homomorphism_of_Organized_Major_Orientation_ICCV_2025_paper.html；代码：https://github.com/MrPingQi/HOMO_Feature_ImgMatching
4. MODA/OSSDet（AAAI 2026）用物体感知掩码抑制背景并跨光谱细化，适合迁移为建筑区域内融合，但不能替代位移估计。论文：https://ojs.aaai.org/index.php/AAAI/article/view/42457；代码：https://github.com/shuaihao-han/MODA
5. 2026年的SAR–Optical配准评测显示专用跨模态匹配器和通用预训练匹配器都值得作为外部上限对照。论文：https://openaccess.thecvf.com/content/CVPR2026W/IMW/html/Corley_Are_Pretrained_Image_Matchers_Good_Enough_for_SAR-Optical_Satellite_Registration_CVPRW_2026_paper.html

## 最值得迁移的方案

### A. S2R安全匹配探针（已实现）

冻结D1除Stage-2/3 query/key之外的全部参数，只使用已知合成位移优化相对offset。优点是成本最低、归因清楚、无需重跑D1。难度低，创新性中等，但它是后续结构创新成立前必须完成的机制验证。

### B. 建筑约束的有界残差offset adapter（条件实现）

仅在S2R仍不能把位移误差降低30%时做。在原offset上增加零初始化小残差分支，输入局部相关分布、相关熵、峰值margin和建筑概率，输出限制在局部窗口内的dx/dy修正。它借鉴DSA的显式位移与Mona的小容量适配原则，但针对RGB–SAR建筑损毁重新设计。难度中，创新性中高。

### C. 结构方向匹配证据（条件实现）

在query/key前加入无参数或极低参数的梯度幅值与主方向编码，让RGB和SAR通过几何边缘而非绝对亮度比较。该思路来自HOMO-Feature，但要限制在建筑区和Stage-2/3。风险是SAR斑点产生伪边缘，因此必须门控融合并做滤波对照。

## 验证顺序与准入标准

1. 先跑S2R 5000 optimizer steps，不重跑D1/S1/S2。
2. 用latest checkpoint做none、x8、y8、x16、y16审计。
3. 第一关：x8/y8位移轴MAE相对D1下降至少30%。不通过则进入方案B。
4. 第二关：clean mIoU相对D1不低于-0.30pp，Damaged F1不下降，至少10/14事件不下降。若位移通过而分类失败，加入clean教师蒸馏，不立即解冻全网络。
5. 只有A通过才训练B；只有B仍受模态纹理干扰才加入C。每次只改变一个因素。
6. 最终确认要求事件bootstrap CI下界大于0，并报告最差事件、Damaged→Intact、Affected Recall和位移响应曲线。

## 风险

- “即插即用”不代表适合RGB–SAR物理关系；通用注意力涨点不能证明解决配准。
- 仅在合成平移上提高可能学成数据增强技巧，必须同时测x/y、8/16像素及错误配对。
- 增加训练步数不能修复错误目标；S2说明联合训练更久可能继续伤害Damaged。
- Val450只用于既定准入流程，不访问official instances/test。

## 可投稿题目候选

1. Safe Correspondence Adaptation for Misregistered RGB–SAR Building Damage Assessment
2. Building-Guided Residual Alignment for Robust Cross-Modal Disaster Damage Mapping
3. Decoupling Correspondence Learning from Damage Recognition in RGB–SAR Assessment
4. Structure-Invariant Local Matching for Optical–SAR Building Damage Segmentation

## 最终落地建议

不要把CAFR、Mona、DSA、CMA、MSC同时塞进DamageFormer。先用S2R验证“冻结分类、只学匹配”能否解决已观察到的弱点；结果再决定改匹配器表达能力还是加分类保护。论文链条应是：发现联合对齐造成类别退化 → 解耦训练验证因果 → 必要时提出建筑约束的显式残差对齐。

## 在建筑定位基础上的具体实现

当前代码已经采用两种互补的建筑约束：

1. 网络前向约束：冻结的RGB定位分支产生soft building probability，经detach后送入Stage-2/3；它参与建筑外环、融合输入和可靠性信息。推理时只使用这张预测概率图。
2. 训练损失约束：GT location只作为offset loss的support mask，使位移监督集中在真实建筑像素；它不作为推理输入，因此不造成测试标签泄漏。

第一版S2R应保持这个设计，不把GT mask直接送入前向网络，也不把预测图硬阈值化。soft probability能保留定位不确定性，建筑边缘通过outer ring提供邻域信息。

论文迁移顺序如下：MINIMA对应“已知合成平移监督”；Mona对应“冻结主干、少量参数适配”；MODA对应“建筑soft mask抑制背景”；XoFTR对应S2R通过后的“Stage-3粗位移、Stage-2细化”；HOMO-Feature对应粗细匹配仍受模态差异时增加的梯度方向证据；2026 SAR–Optical评测中的匹配器只作为外部上限或预配准对照。

建议消融：D1；原S2全模型联合训练；S2R只训练query/key；S2R去掉建筑support；S2R使用预测soft support。首先跑前三项，只有S2R通过位移和类别安全门槛才增加后两项，避免重复训练无效组合。

代码现已提供五个隔离变体：`s2r0`全图support、`s2r`训练期GT建筑support、`s2rp`预测soft建筑support、`s2rr`预测建筑加16像素外环、`s2rcf`预测建筑外环加Stage-3粗匹配/Stage-2残差匹配。`s2rcf`不增加新参数，保持D1 state-dict兼容，从而把收益归因于匹配顺序而不是模型容量。
