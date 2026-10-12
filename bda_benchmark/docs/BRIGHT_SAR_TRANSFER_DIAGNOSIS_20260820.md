# BRIGHT RGB–SAR 表征与迁移诊断（2026-08-20）

## 研究问题

1. 光学–SAR 异构变化检测如何处理 SAR、对齐和融合？
2. 当前 H6-A2 的主要瓶颈究竟是输入、SAR 编码器、融合，还是数据本身？
3. CROMA、SARMAE、SARDet、HeteCD、DSTCD 的权重能否直接迁移？

## 核心结论

当前问题不是单一的“网络不够强”，而是三个约束叠加：

1. **观测上限**：BRIGHT 只有单极化、8-bit 振幅 SAR。官方论文明确指出它对局部/轻中度 Damaged 的信息不足，对大范围破坏更敏感。
2. **事件/传感器域差异**：数据来自 Capella 与 Umbra、Spotlight 与 Stripmap、VV 或 HH；官方数据分析也显示 SAR 特征具有明显事件间可分性。当前代码却只使用一组训练集全局 p01/p99 与 mean/std。
3. **迁移契约不一致**：当前 A2 是在 BRIGHT 上从随机初始化学习的一通道 ConvNeXt 风格金字塔；外部模型通常要求不同极化、频段、动态范围、空间尺度和预训练目标。直接替换权重会同时破坏输入和融合特征坐标系。

因此，当前最值得验证的不是第十五个融合模块，而是：**候选 SAR 表征在不接入融合器时，能否跨事件线性地区分 Intact/Affected 与 Damaged。**

## 本地实现证据

- 数据读取将 SAR 强制为单标量通道；非重复 RGB 通道会直接报错。
- h1–h4 使用全训练集统一 p01/p99 裁剪和统一 mean/std，而非传感器/事件自适应标定。
- A2 的 SAR 主干是随机初始化的一通道 Conv/GN + ConvNeXt 风格网络；RGB 则使用 ImageNet Swin-T。
- H6-A2 冻结 SAR stem、stage1、stage2，只训练较深阶段与融合。早期 SAR 表征一旦学到事件/传感器风格，后续很难纠正。
- stage3/4 的 sensor prototype 假定 `sar_common-rgb_common` 可由单图平均原型校正；H8 Oracle 已显示该假设上限很低。

## 文献方法谱系与迁移判断

| 方法 | 真正有效的对象 | 对 BRIGHT 的启示 | 可否直接换 SAR 权重 |
|---|---|---|---|
| HeteCD (ISPRS JPRS 2025) | 非共享双编码器 + 特征一致性 + 监督差异挖掘 | 应迁移完整成对前端/训练思想，且只在可靠不变区对齐 | 否；XiongAn 二值模型与三级损毁头不兼容 |
| DSTCD (JAG 2026) | 异构配准预任务后迁移成对表征 | 迁移对象是跨模态对应能力，不是独立 SAR backbone | 否；且本项目已有“流近零/局部匹配无收益”反证 |
| CROMA (NeurIPS 2023) | Sentinel-1 双通道与 Sentinel-2 12 通道独立编码及联合预训练 | 可作冻结候选表征探针 | 否；BRIGHT 是 VV 或 HH 单通道 VHR 8-bit |
| SARDet-100K / MSFA (NeurIPS 2024) | 输入增强、域过渡、整个检测框架的分阶段迁移 | 借鉴“分阶段迁移”，不要只换第一层 | 否；频段/极化/传感器/目标和任务头均不同 |
| SARMAE (CVPR 2026) | SAR 专属掩膜预训练与 speckle-aware 表征 | 必须先做冻结多层 probe | 否；T1/T1E 已表明直接深层残差迁移无效 |

## 为什么验证高、测试下降

BRIGHT 标准划分不是未见事件测试，而是每个事件内部按 7:1:2 划分后再合并。因而验证—测试下降不能简单归因于“新灾害事件”。更可能的机制是：

- 同一事件内仍有空间块、建筑类型、损毁比例和成像质量差异；
- 全局 SAR 标准化保留了强烈事件/传感器风格；
- Damaged 只占建筑像素约 10.7%，弱信号容易被 Intact 多数类吞没；
- A2 相比 P3 的额外测试退化主要集中在 Destroyed，说明 SAR 分支的域稳定性确有问题；但 Damaged 困难同时也是数据观测上限和类别边界问题。

## 下一步：两阶段证伪，而不是直接替换主干

### E0：输入契约与域漂移审计（已提供脚本）

运行 `audit_sar_transfer_contract.py`，确认通道、dtype、值域、逐事件统计，以及 train/val 分布差异。测试集默认不读取；它不能参与模型选择。

### E1：冻结表征 Probe（下一项应实现）

对 Current-A2、SARMAE、CROMA-adapter、SARDet-backbone 使用完全相同的建筑 ROI 样本和相同的小型线性头，进行按事件分组交叉验证：

- 任务 A：Intact vs Affected；
- 任务 B：Damaged vs 非 Damaged；
- 指标：event-macro AUROC/AUPRC、worst-event AUPRC、14 事件方向一致率；
- 不接入现有 fusion，不看最终 test。

只有候选在任务 B 上较 A2 的 event-macro AUPRC 提高至少 5%，且 worst-event 不下降，才进入 E2。

### E2：兼容迁移（仅对 E1 胜者）

保留 P3 RGB 定位；候选 SAR 编码器前放模态专属输入适配器，后放多层 FPN projector。先冻结候选主干，只训练 adapter/projector 和 affected head；再按层逐步解冻。严禁将外部 checkpoint 的检测头、二值 CD 头或不兼容 patch embedding 直接装入 A2。

## 停止条件

- 输入契约不匹配且没有原始双极化/物理幅度数据：CROMA 只能作为探索性 probe，不能成为主线。
- 冻结 probe 不能改善 Damaged 的跨事件 AUPRC：停止该外部模型，不进行 1500/5000 步主模型训练。
- probe 有效而融合无效：问题在 projector/fusion；probe 本身无效：问题在表征/数据，不应再调融合。

## 主要来源

- BRIGHT 数据与限制：https://essd.copernicus.org/articles/17/6217/2025/
- HeteCD 官方代码：https://github.com/weiAI1996/HeteCD
- DSTCD 官方代码：https://github.com/Lucky-DW/DSTCD
- CROMA 官方代码：https://github.com/antofuller/CROMA
- SARDet-100K/MSFA 官方代码：https://github.com/zcablii/SARDet_100K
- SARMAE 官方代码：https://github.com/MiliLab/SARMAE
