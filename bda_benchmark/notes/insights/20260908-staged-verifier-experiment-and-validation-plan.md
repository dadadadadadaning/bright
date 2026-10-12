# BRIGHT 指标对齐修改簇验证器：分阶段实验与验证计划

- 用户研究方向：灾前 RGB—灾后 SAR 建筑损毁语义分割
- 用户研究问题：如何依次验证 Metric-aligned Utility、低容量 Verifier、概率校准、TTA、伪实例几何与上游损失，且不污染独立验证集
- 分析日期：2026-09-08
- 项目证据：P3、fixed S2、D1 修改簇缓存、现有 MACU 实验代码
- 论文证据：MetaSeg、SelectiveNet、Lovász-Softmax、IRNet、BESTIE、Panoptic-DeepLab、BRIGHT 相关工作

## 总原则

只允许三条逐级升级路线：

1. 选择器路线：Metric Utility → M2 → M3 → 概率校准/TTA → M4 → DeepSets；
2. 伪实例路线：LocCC → 边界/距离拆分 → 条件性 center/offset 或 SAM；
3. 上游模型路线：P3 → CE+Lovász/ordinal → 事件均衡/GroupDRO。

每次只改变一个因素。meta_train 用于拟合，meta_calibration 用于选择模型和唯一阈值，meta_confirm 只允许在整套配置冻结后确认一次，Val450 只用于最终准入，Official Test 最后访问。

## 数据流

```text
RGB(pre) + SAR(post)
       ↓
冻结 P3 → logits/probability → 概率校准 → 可选 TTA
       ↓
P3 建筑定位 → LocCC 伪实例 → A6/fixed S2
       ↓
比较 S2 与 P3 → 修改图 → 修改簇 C
       ↓
面积≤32：保留 fixed S2
面积>32：提取簇级特征 → M2/M3/M4/DeepSets
       ↓
accept：采用 S2；reject：回退 P3
       ↓
逐图混淆矩阵 → mIoU、D-F1、类别 IoU、事件指标、bootstrap CI
```

## 第一阶段：不重跑 P3

### E0 协议复现

- 对照：P3、fixed S1、fixed S2。
- 必须复现 P3=60.5020、fixed S2=61.5716 附近的既有结果。
- 检查缓存指纹、图像数、事件数、类别映射和禁用数据访问标志。
- 复现失败时停止后续实验。

### E1 Metric-aligned Utility 审计

- 对每个大簇计算：旧 pixel-net utility、训练集冻结梯度得到的 linear metric utility、精确撤销该簇得到的 exact marginal utility。
- 比较三种标签的正负号一致率、类别/方向分布和事件分布。
- Oracle：分别按三种 utility 的真值符号选择簇，和 fixed S2 比较。
- 准入：metric Oracle 增益≥0.30pp；至少10个事件不下降；CI下界>0；linear/exact 符号一致率≥95%；正负样本均≥100。
- 未通过：说明候选修改空间不足，停止训练 Verifier，转向伪实例或 P3。

### E2 M2 固定规则

- 特征仅用修改方向和面积分箱，不使用视觉特征。
- 规则候选必须在 meta_train 预定义；meta_calibration 只选一个规则，不能无限搜索。
- 目的：判断简单先验是否已经足够，并为学习模型提供必须超过的基线。

### E3 M3 Logistic Regression

- 输入：面积/形状、S2角色、RGB边缘、SAR纹理、簇—稳定区关系；训练标准化统计只能来自 meta_train。
- 标签：metric utility 正负；近零效用簇忽略或低权重。
- 权重：事件逆频率 × `sqrt(|utility|)`，并截断极端权重。
- 损失：效用加权 BCE；输出接受概率。
- 阈值：meta_calibration 上在拒绝率≤70%的候选中，先最大化 mIoU，再以 D-F1 和较低拒绝率破平局。
- 准入：相对 fixed S2 ≥+0.30pp；D-F1不下降；≥10/14事件不下降；Oracle recovery≥25%；CI下界>0；拒绝率≤70%。

## 第二阶段：增强可靠性证据

### E4 概率校准

- 最小方案：temperature scaling；扩展方案只允许再比较 vector scaling，避免大量校准器搜索。
- 为避免泄漏，在 meta_train 内按事件做交叉拟合，生成训练用校准概率；最终校准器仅用 meta_train 拟合并应用到 meta_calibration。
- 校准质量：NLL、Brier、ECE，并按类别和事件分别报告。
- 下游对照：原始概率 M3 vs 校准概率 M3；网络、特征和阈值搜索空间保持一致。
- 通过条件：不只 ECE 下降，还要最终 mIoU/事件稳定性提高；只改善 ECE 而不改善选择决策时，不进入主模型。

### E5 TTA 稳定性

- 对 RGB/SAR 成对执行相同几何变换，并在输出端逆变换对齐；第一版只用水平/垂直翻转与 90°旋转，不使用会改变 SAR 物理分布的颜色增强。
- 簇特征：类别投票一致率、概率方差、熵方差、margin方差、Jensen-Shannon差异、S2修改方向一致率。
- 三个必要对照：无TTA；只对TTA概率求均值；TTA均值+稳定性特征。这样可以区分“集成收益”和“Verifier证据收益”。
- 通过条件：相对已校准M3进一步提升≥0.10pp，或在mIoU不降时显著增加非下降事件/提高CI下界。

### E6 M4 两层 MLP 与 E7 DeepSets

- M4固定 hidden=32，不做大规模宽度搜索；只有M3已正增益且训练/校准均显示线性欠拟合才运行。
- DeepSets仅在统计量表达不足时运行；每个簇确定性采样最多128或256个像素，使用共享小MLP、mean/max pooling和小分类头。
- 必须对照参数量相近的普通MLP、像素顺序打乱不变性，以及仅汇总统计特征。
- DeepSets进入主模型的条件：在独立校准事件上稳定超过M4，而不是只提高训练AUROC。

## 第三阶段：候选和上游改进

### E8 边界/距离辅助伪实例

- 先做无训练版本：P3建筑掩膜距离变换 + marker-controlled watershed；可再叠加RGB边缘作为切割代价。
- 对照：LocCC、仅距离、距离+RGB边界。
- 无实例标注时不宣称真实 instance AP；使用语义边界F1、组件面积/细长度异常率、修改簇Oracle、fixed S2和最终Verifier收益评估。
- 通过条件：Oracle空间或fixed S2提高≥0.20pp，且建筑定位IoU不下降；否则停止，不进入center/offset。

### E9 CE+Lovász/ordinal

- 使用相同P3架构、数据、训练轮数和种子依次比较：CE；CE+Lovász；CE+ordinal；只有单项均有证据时才比较三者组合。
- ordinal将 Intact→Damaged→Destroyed 表示为两个累计二分类阈值，惩罚跨两级错误高于相邻一级错误。
- 同时评价原始P3、fixed S2、校准误差和Cluster Oracle；只看P3 mIoU不够。
- 启动条件：Metric Oracle不足或错误主要来自类别表征，而不是Verifier无法识别好簇。

### E10 事件均衡/GroupDRO

- 对照：普通ERM；event-balanced sampler；GroupDRO。
- GroupDRO的group只使用训练事件；不允许根据meta_confirm或Val450事件表现调权重。
- 报告总体mIoU、事件宏平均、最差事件、最差四分位、D-F1和平均性能损失。
- 准入建议：最差事件明显提升，同时总体mIoU下降不超过0.10pp、D-F1不下降；否则保留事件均衡而不用GroupDRO。

### E11 center/offset 与 SAM

- center/offset只有在边界/距离已证明粘连是主要瓶颈且能构造可信伪中心时开展；否则伪标签会把LocCC错误重新学一遍。
- SAM只作为冻结后处理对照：固定模型版本、提示生成方式、候选过滤和算力预算；只在灾前RGB上生成建筑候选，再与P3建筑掩膜相交。
- 二者都不能使用官方实例真值调参；报告运行时间、显存、失败率和协议依赖。

## 统一实验矩阵

| 编号 | 变量 | 必要对照 | 主要问题 | 通过后去向 |
|---|---|---|---|---|
| E1 | utility标签 | pixel/linear/exact Oracle | 旧监督是否错位 | E2/E3 |
| E2 | 规则选择 | P3/S2 | 简单规则是否足够 | E3 |
| E3 | Logistic | M2 | 低容量模型能否恢复Oracle | E4 |
| E4 | 概率校准 | raw vs calibrated | 概率是否可信 | E5 |
| E5 | TTA证据 | none/mean/features | 稳定性是否可预测收益 | E6或冻结 |
| E6 | MLP | M3 | 是否存在非线性 | E7或冻结 |
| E7 | DeepSets | 参数匹配MLP | 分布编码是否必要 | 冻结 |
| E8 | 边界/距离 | LocCC | 粘连是否限制候选 | center/offset或冻结 |
| E9 | 损失 | CE | P3类别表征是否不足 | 重建下游链路 |
| E10 | GroupDRO | ERM/event-balanced | 未见事件是否更稳 | 重建下游链路 |
| E11 | center/offset/SAM | 最佳轻量伪实例 | 高成本几何是否值得 | 最终候选 |

## 最终冻结与验证

1. 所有模型、特征、温度、阈值、拒绝上限和随机种子先冻结；
2. meta_confirm只运行一次；失败后不得根据其结果调参；
3. 通过标准：≥+0.30pp、D-F1不下降、≥10个事件不下降、Oracle recovery≥25%、拒绝≤70%、bootstrap CI下界>0；
4. 通过后才运行Val450，比较P3、fixed S2和唯一冻结模型；
5. Val450只允许确认，不再选阈值；Official Test最后一次执行。

除主指标外，应固定报告Verifier AUROC/AUPRC、风险—覆盖率曲线、可靠性图、每个修改方向的接受率、每事件收益和最差事件结果。统计单位必须以图像或事件为重采样单位，不能把大量相关像素/簇当成独立样本进行bootstrap。

## 建议题目方案

1. **Metric-Aligned Selective Context Refinement for Cross-Event Building Damage Assessment**：以指标边际效用和安全回退为主线。
2. **Calibrated Cluster Verification for RGB–SAR Building Damage Mapping**：强调概率校准与TTA稳定性证据。
3. **Boundary-Aware Pseudo-Building Refinement without Instance Supervision**：只在轻量边界/距离路线被实验证实时使用。

## 风险

- 校准集同时被用于校准概率、选模型和选阈值会产生二次过拟合，必须用meta_train内交叉拟合缓解。
- 多个模块一起加入会破坏贡献归因，应采用顺序门控实验。
- GroupDRO、损失重训和伪实例改造会改变上游分布，不能直接复用旧Verifier；它们通过后需重新构建缓存和训练下游。
- 没有实例真值时只能称“伪实例/组件质量代理”，不能声称提升真实实例AP。

## 证据

| 来源 | 支撑内容 |
|---|---|
| MetaSeg, IJCNN 2020 | 区域熵、几何等特征可用于低容量区域质量预测 |
| SelectiveNet, ICML 2019 | 拒绝机制应以风险—覆盖率权衡评价 |
| Lovász-Softmax, CVPR 2018 | 像素CE与IoU目标存在错位 |
| IRNet / BESTIE / Panoptic-DeepLab | 边界、距离/位移、中心可用于从语义区域恢复实例结构 |
| BRIGHT项目现有实验 | fixed S2优于P3，修改簇Oracle存在空间，但旧高分辨率单证据未通过跨事件准入 |

