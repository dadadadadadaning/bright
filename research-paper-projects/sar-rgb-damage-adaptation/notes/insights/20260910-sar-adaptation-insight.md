# 光学-SAR文献对当前P5停滞的迁移启发

- 用户研究方向：灾前RGB、灾后单通道显示型SAR的建筑损毁四分类。
- 用户研究问题：E1在1000步达到49.4142%后，1500步降至49.0078%，如何修改SAR路线并保持P3强基线。
- 分析日期：2026-09-10
- 论文来源：M2CD、SARDet-100K、CROMA、DeCUR、AnySat、X-STARS、GLCD-DA。

## 当前证据判断

1000到1500步不是全面崩溃，而是类别间转移：Intact IoU +0.14pp、Damaged IoU +0.51pp、Destroyed IoU -1.86pp；Damaged→Intact改善1.02pp，但Destroyed→Intact恶化3.18pp，Affected recall下降0.90pp。训练损失已很低，输出蒸馏损失仅0.012，但SAR特征和融合特征距离仍为0.494和0.208，说明当前主要矛盾可能是“要求新的SAR专用表示模仿旧伪RGB-Swin表示”。

## 论文可迁移模块

| 模块 | 原文依据 | 可迁移方式 | 适配条件 |
|---|---|---|---|
| 训练期桥接路径 | M2CD O2SP | 用小投影器连接P3伪RGB-SAR与原生SAR，仅对桥接表示蒸馏 | 不直接约束全部SAR特征 |
| 公共/独有表示 | DeCUR | 只对齐64维公共子空间，保留SAR独有纹理通道 | 对齐应限制在稳定建筑区 |
| 独立雷达/光学编码 | CROMA | 保留双编码器思想，不直接套Sentinel预训练权重 | BRIGHT单通道显示SAR需自行适配 |
| 多阶段域桥接 | SARDet-100K | 先做train-only SAR自监督或轻量桥接，再进行损毁训练 | 固定滤波不能替代多阶段预训练 |
| 传感器投影对齐 | X-STARS | 在低维投影后做稠密对齐 | 真实变化区不能强制相同 |
| 边缘辅助 | GLCD-DA | 把梯度/距离特征送入浅层边界辅助Head | 只作为辅助证据，不替换语义主干 |
| 尺度自适应 | AnySat | 借鉴patch尺度与模态专用输入适配 | 先用冻结探针验证收益 |

## 为什么当前改变可能不合适

1. 当前P5用随机初始化的SAR编码器直接替换P3已训练好的伪RGB-SAR分支，缺少“P3输出 + 零门控SAR残差”的安全锚点，因此初始化精度从60.5%掉到接近0，再艰难恢复。
2. `loss_kd_sar`直接让原生SAR特征逼近P3伪RGB-Swin特征。它一方面要求SAR专用编码，另一方面又要求其像旧RGB式表示，目标互相冲突。
3. MSFA论文的收益来自“滤波增强 + 多阶段预训练 + 域迁移”，当前C1只有简化滤波，不能把MSFA完整结论归因给四个固定通道。
4. BRIGHT文件是0--255显示型TIFF，不是已确认的物理幅度/功率数据；Gamma散斑模型、log幅度变换或Sentinel预训练不能未经核验直接套用。
5. RGB与SAR跨时间且灾害造成真实结构变化。对所有位置强制对齐可能把真正的损毁差异也消除，应该只在稳定区对齐，在候选变化区保留差异。

## 最小可行实验

先完成当前E1到2000步，不改配置。若2000步仍未超过49.4142%、Affected recall仍约55%且Destroyed继续下降，则停止扩展E2--E4，转入以下三组500--1000步诊断：

- K0：当前损失。
- K1：保留输出KD，关闭SAR特征KD与融合特征KD。
- K2：P3 logits加零门控SAR Occurrence残差；SAR只能在有证据时修正P3，不从零重建全部四分类。

若K1优于K0，证明旧特征蒸馏错位；若K2优于K1，说明主要问题是缺少安全锚点。之后再增加低维公共投影和稳定区对齐，不先上MoE、CROMA或AnySat全模型。

## 建议题目方案

1. 题目：Building-Guided Safe Residual Adaptation for Optical-SAR Damage Assessment
   - 核心思路：冻结P3，以建筑概率和SAR质量门控控制零初始化Occurrence残差。
   - 最小实验：P3、全替换P5、输出KD残差、质量门控残差。
   - 预期贡献：证明SAR应作为安全增量证据，而不是替换强光学基线。
2. 题目：Change-Preserving Common-Unique Alignment for Cross-Modal Building Damage Mapping
   - 核心思路：稳定区对齐公共特征，变化区保留SAR独有特征。
   - 最小实验：全特征KD、低维公共KD、稳定区公共KD。
   - 预期贡献：避免跨模态对齐抹除真实灾害变化。
3. 题目：Occurrence-First Quality-Gated SAR Fusion for Unseen-Event Building Damage Assessment
   - 核心思路：SAR只先修正Intact/Affected，再在高置信Affected区域判断严重程度。
   - 最小实验：flat、Occurrence残差、Occurrence+Severity、加入事件宏平均。
   - 预期贡献：同时改善损毁漏检与未见事件鲁棒性。

## 风险控制

- 不使用验证集选择SAR归一化或预训练统计。
- 不在当前E1中途改损失，保持现有曲线可解释。
- 不直接宣称固定滤波无效；必须等待E2对照。
- 不访问Official Test，先用Val450和逐事件bootstrap确认。
