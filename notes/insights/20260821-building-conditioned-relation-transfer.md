# 建筑条件 SAR 区域证据：论文迁移与最小实验

## 研究背景与当前证据

当前任务使用灾前 RGB 与灾后单极化 SAR 进行建筑损毁三级分类。Q0 已证明：冻结 H6-A2 的 SAR Stage4 特征在明确限定建筑区域后，Damaged 条件召回可以由旧 `sar_aux` 的 10.61% 提高到约 35.41%。因此，现阶段的问题不再是“SAR 是否完全没有损毁信息”，而是“如何把稀疏、局部的损毁响应汇总成稳定的建筑级证据，并判断它是否能够纠正 RGB 主路径”。

## 可迁移论文与边界

### LPEM

- 论文/代码：<https://github.com/yulisun/LPEM>
- 可迁移思想：不要只比较单像素；使用局部区域之间的关系与一致性组织异构模态证据。
- 不直接迁移部分：其目标是无监督异构变化检测，不是 BRIGHT 的建筑三级损毁；不能把其伪标签和全景超像素流程直接并入当前模型。

### CGSL

- 论文/代码：<https://github.com/TongfeiLiu/CGSL-for-MCD>
- 可迁移思想：异构图像应分别表示，再在区域图上建立公共结构，而不是强迫 RGB 与 SAR 像素值直接一致。
- 风险：全景图会把背景、道路和植被关系带入建筑损毁头；在本项目中必须受建筑支持约束。

### HFA-PANet

- 论文/代码：<https://github.com/TongfeiLiu/HFA-PANet-for-MCD>
- 可迁移思想：非共享编码和分层特征组织，支持后续受控的 Stage3/Stage4 建筑区域对照。
- 风险：特征对齐若施加在 Damaged/Destroyed 区域，可能抹去真正的损毁差异。

### RIEM

- 论文/代码：<https://github.com/yulisun/RIEM>
- 可迁移思想：先验证关系或规则是否能形成可靠的变化证据，不应默认直接相减就是最优表示。
- 风险：其规则诱导任务仍是二值变化，不能直接承担 Damaged/Destroyed 严重度判断。

### BRIGHT 建筑引导方法

- DFC 2025 建筑引导官方实现：<https://github.com/Henryjiepanli/Building-Guided-Pseudo-Label-Learning-for-Cross-Modal-Building-Damage-Mapping>
- BRIGHT Challenge 2026 官方汇总：<https://github.com/ChenHongruixuan/BRIGHT/blob/master/cvprw26/README.md>
- 可迁移思想：RGB 负责稳定建筑支持，SAR 只在建筑候选内提供损毁证据；最终决策应逐步转向建筑/候选级。
- 协议边界：伪标签或目标域适配必须与纯监督设置分表报告；验证标签和测试数据不得用于模型选择。

## 本轮最小可行实验：Q1-R0

Q1-R0 不训练网络，只对同一 Q0 概率图和同一建筑区域划分比较七种聚合：

1. `pixel_vote`：区域内像素类别多数投票；
2. `mean`：区域平均概率，作为稠密证据基线；
3. `max`：每类最大响应；
4. `top10` / `top20`：每类最高 10% / 20% 响应均值；
5. `lse`：平滑最大值，降低单个噪点支配；
6. `area`：超过阈值的响应面积比例。

使用两个支持协议：

- `oracle`：真实语义建筑掩膜，仅测“聚合方法上限”，不属于可部署推理；
- `p3`：冻结 P3 建筑概率阈值后生成连通区域，属于实际可用诊断，但相邻建筑可能被合并，因此仍不是官方实例标签。

准入规则锁定为：最佳局部聚合相对 `mean` 的 Damaged Recall 至少提高 2 pp，Macro-F1 下降不超过 1 pp，并且至少 10 个事件非负。只有通过后，才实施 Q1-S3/S4/MS；否则停止区域关系路线。

## 后续迁移次序

1. Q1-R0：无训练聚合审计；
2. Q1-S3/S4/MS：固定区域协议，只比较 SAR Stage3、Stage4、Stage3+4；
3. Q1-O：建筑级 P3/SAR 互补 Oracle；
4. Q2：仅在 Oracle 通过后训练安全的建筑级融合；
5. Q3：event-held-out 与跨事件校准。

这个顺序把“区域划分有效”“SAR 能被建筑级读取”“SAR 能纠正 P3”“可学习融合能逼近互补上限”拆成四个独立问题，避免模块拼接后无法解释结果。

## 可行研究题目

1. **Building-Conditioned Sparse SAR Evidence Aggregation for Optical–SAR Damage Assessment**
2. **From Pixel Responses to Building Decisions: Reliable SAR Evidence for Cross-Modal Damage Mapping**
3. **Event-Robust Building Query Fusion for Pre-Disaster Optical and Post-Disaster SAR Imagery**
4. **Optical-Anchored Region Evidence Mining for Heterogeneous Building Damage Assessment**

## 风险与停止条件

- 若 Oracle 支持下 Top-k/LSE 也不优于 Mean，说明“局部稀疏响应”假设不成立，不开发图关系模块。
- 若 Oracle 有效而 P3 支持无效，瓶颈是建筑实例/区域生成，不是 SAR 聚合。
- 若 Q1 独立 SAR 很强但 P3+SAR Oracle 很弱，说明两者错误高度重合，不训练融合器。
- 若收益只来自少数事件，不进入测试集，先执行事件互斥验证。
