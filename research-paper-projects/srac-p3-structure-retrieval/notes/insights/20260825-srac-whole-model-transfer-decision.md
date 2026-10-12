# SRAC 整模型迁移与 P3 改造决策

- 用户研究方向：灾前 RGB 与灾后 SAR 的建筑损毁评估
- 用户研究问题：能否直接采用现成 RGB–SAR 建筑损毁模型，并在其基础上修改，而不是继续在 P3 后叠加零散模块
- 分析日期：2026-08-25
- 论文来源：SRAC IGARSS 2026 录用记录、实验室新闻稿、用户提供的方法总览图
- 趋势来源：BRIGHT ESSD 2025、DFC 2025 冠军官方实现、BRIGHT Challenge 2026、SSN ECCV 2018、SpixelFCN CVPR 2020

## 证据边界

SRAC 已出现在 IGARSS 2026 正式录用名单（paper 2851），但截至本次检索未找到正式全文或官方代码。图中可确认四阶段总体结构，但损失权重、VMamba 具体版本、SSN 训练标签、Memory Bank 更新规则、MMA-P Gather 细节、75.65% 指标口径及消融贡献仍不可核验。因此仓库中的 P3-SRAC 系列只能称为受控迁移，不能称为 SRAC 复现。

## 论文可迁移模块

| 模块 | 图中位置 | 可迁移方式 | 适配条件 |
|---|---|---|---|
| VMamba+UNet++ 可学习超像素 | Stage 1 | 作为独立、预训练的结构分区器整块迁移 | 需获得官方实现或依据 SSN/SpixelFCN完整重建；不能用轻量软网格替代后宣称复现 |
| 结构先验调制 | Stage 2 | 对像素表征做背景压暗、边界增强、建筑支持注入 | 建筑先验必须仅来自 RGB/P3 推理，测试时不能读取 GT |
| 空间注意力区域池化 | Stage 3 | 用可学习权重把像素特征聚合成超像素节点 | 必须保留 raw mean/std skip，避免投影压掉 SAR 强度统计 |
| 类别均衡训练记忆 | Stage 4 | 训练集节点离线建库，每类固定容量，验证/测试只读 | 禁止验证/测试写库；需要明确同事件与跨事件检索协议 |
| 元数据条件检索分类 | Stage 4 | 灾害类型、区域、几何编码与检索支持一起进入分类器 | 不能使用测试真实灾害标签；应使用可推理场景统计或公开元数据并单独报告协议 |

## 与当前代码的差异

当前 `DamageFormer_P3_SRAC_Controlled.py` 已明确声明不是 SRAC 复现。SR1/SR1F/SR1P/SR1PAI主要替换或投影 P3 最终特征，复用了 P3/Swin 特征和轻量局部 SSN 式分配；它们没有复现图中的完整 VMamba 双像素表征、结构调制、MMA-P Gather 与元数据注入。SR2虽实现类别均衡 FIFO memory 和 cross-attention，但建立在尚未通过的轻量区域表征之上。故这些负结果只否定当前替代实现，不能否定完整 SRAC。

## 匹配与可行性评估

- 收录信息：SRAC 已被 IGARSS 2026 接收；目前仅有录用记录、新闻稿和结构图。
- 有无代码：SRAC 官方代码未证实；SSN、SpixelFCN、DFC 2025 冠军与 BRIGHT Challenge 2026方案有公开实现或官方入口。
- 任务匹配程度：SRAC与用户任务直接匹配；DFC 2025冠军也是直接匹配且代码成熟；HeteCD等二值异构变化检测仅为邻近任务。
- 创新度：直接复刻 SRAC 只能作为强基线；在正式复现上增加事件安全记忆、层级损毁分类或原生 SAR 表征才可能构成新方法。
- 迁移难度：完整 SRAC 高；DFC 2025冠军中等；基于 P3 的轻量模块低但已多次表现为近零收益。
- 主要风险：无全文导致错误复刻；Memory 数据泄漏；元数据形成事件捷径；像素级语义标签不足以训练单栋建筑一致性；轻量超像素与完整 VMambaSSN能力不等价。

## 落地建议

### 路线 1：优先整模型基线

先原样跑通 DFC 2025 冠军的纯监督版本，保持其 RGB 建筑提取、跨模态变化检测和建筑引导流程；禁用测试伪标签，建立 clean baseline。这样可以判断“成熟任务模型”是否优于 P3，而不再通过局部模块猜测。

### 路线 2：等待/获取 SRAC 全文与代码后整块迁移

若获得官方代码，先不与 P3混合，严格复现 SRAC 的 Stage 1–4、输入归一化、标签转换、Memory 构建和官方划分。复现达到论文报告附近后，再做单模块替换。

### 路线 3：无官方代码时的诚实重建

采用公开 SSN/SpixelFCN 作为分区器，独立训练并用 ASA、Boundary Recall/Precision、Compactness 和损毁标签纯度验收；随后实现结构调制与区域嵌入；最后才加入 train-only memory。每阶段必须有单独基线，不复用当前高熵软网格结果。

## 最小实验与关键对照

1. External-S0：DFC 2025冠军纯监督、无测试伪标签。
2. SRAC-R0：SSN/SpixelFCN官方分区器 + 原始像素分类，无检索。
3. SRAC-R1：R0 + 结构调制 + 空间注意力区域嵌入。
4. SRAC-R2：R1 + 类别均衡 train-only memory。
5. SRAC-R3：R2 + 事件留出 memory，检查跨事件稳定性。

## 建议题目方案

1. **Clean Reproduction and Event-Safe Adaptation of Retrieval-Augmented Optical–SAR Building Damage Assessment**
   - 任务：BRIGHT RGB–SAR建筑损毁。
   - 方法：完整区域检索基线 + 严格train-only、event-held-out memory。
   - 贡献：区分结构区域、类别记忆和事件泄漏各自作用。
2. **Hierarchical Superpixel Memory for Optical-Anchored Building Damage Occurrence and Severity**
   - 任务：Intact/Damaged/Destroyed层级判断。
   - 方法：RGB负责建筑与发生边界，SAR区域记忆只负责条件严重度。
   - 贡献：避免SAR破坏建筑定位，同时利用稀有损毁支持样本。
3. **Native-SAR Region Retrieval with Cross-Event Calibration for All-Weather Building Damage Mapping**
   - 任务：跨事件BRIGHT评估。
   - 方法：原生SAR编码器、结构区域嵌入、事件安全检索和场景校准。
   - 贡献：同时处理SAR输入域、长尾类别与事件偏移。

## 证据

| 类型 | 来源 | 支撑内容 |
|---|---|---|
| 录用记录 | IGARSS 2026 accepted papers | SRAC标题与paper 2851录用事实 |
| 摘要级信息 | 江海实验室新闻稿 | 研究动机、四类困难和75.65%新闻稿结果 |
| 用户图 | SRAC总览图 | Stage 1–4及模块连接关系 |
| 正式论文 | BRIGHT, ESSD 2025 | RGB–SAR任务、跨事件困难和模态差距 |
| 官方实现 | DFC 2025冠军代码 | 建筑提取与损毁变化检测两阶段成熟基线 |
| 正式论文/代码 | SSN ECCV 2018；SpixelFCN CVPR 2020 | 可微超像素的标准实现与质量评估 |

