---
paper_id: P001
title: "HeteCD: Feature Consistency Alignment and Difference Mining for Heterogeneous Remote Sensing Image Change Detection"
year: 2025
authors: ["Wei Jing", "Haichen Bai", "Binbin Song", "Weiping Ni", "Junzheng Wu", "Qi Wang"]
venue: ISPRS Journal of Photogrammetry and Remote Sensing, Vol. 223, pp. 317-327
doi: 10.1016/j.isprsjprs.2025.03.008
url: https://www.sciencedirect.com/science/article/pii/S0924271625001066
pdf: "D:/software/文献/Hetecd Feature consistency alignment and difference mining for heterogeneous remote sensing image change detection.pdf"
code: https://github.com/weiAI1996/HeteCD
status: 已精读（全文 PDF + 官方代码）
---

# P001 HeteCD 中文精读笔记

## 一句话结论

HeteCD 的核心不是把 SAR 变成光学，而是让光学与 SAR 使用**架构相同、权重不共享**的 Transformer 分别编码；再用 FCA 在高层拉近两种模态的特征分布，最后用 3D 时空注意模块从两路特征中学习二值建筑变化。它解决的是“异构二值变化检测”，不是灾害严重程度分类。（PDF p.1 摘要；p.4 Sec.4.1、Fig.2）

## 1. 研究问题与贡献

- 光学受云、光照与获取时机限制，SAR 可全天时获取；但二者成像机理、噪声分布和同类地物表达差异很大，直接差分会把模态差异当成变化。（PDF pp.1-3 Sec.1、Sec.2.2）
- 论文贡献包含：XiongAn 光学-SAR建筑变化数据集；非共享孪生 Transformer；Feature Consistency Alignment（FCA）；3D Spatio-temporal Attention Difference（3D-STA）。（PDF p.2 Sec.1 Contributions）

## 2. 输入与数据处理

### XiongAn

- 数据覆盖 2018-2023 年雄安新区，共 22 对光学-SAR大图；灾前/前时相是 4 m GaoFen-2 多光谱，灾后/后时相是 8 m GaoFen-3 QPSI 四极化 SAR。（PDF p.3 Sec.3.1；p.4 Table 1）
- SAR 预处理依次包含 multilooking、Refined Lee 去斑、地理编码与辐射定标，并通过人工同名点完成光学-SAR配准。（PDF p.3 Sec.3.1）
- 网络实际使用光学 RGB 三通道和 SAR 的 HH、VV、HV 三个极化通道；所有输入除以 255。它不是“单通道 SAR 复制三次”。（PDF p.6 Sec.5.1）
- 以 512×512 窗口、256 步长切块，得到 1941 个训练块和 419 个测试块，其中有变化的分别是 1164 和 275。（PDF p.6 Sec.5.1）
- 正文写“22 对中 18 对训练、剩余 3 对测试”，数量相加为 21；这是正文内部不一致，复现时必须以官方划分文件为准。（PDF p.6 Sec.5.1）

### 标注

- 因 SAR 难以直接目视解译，作者借助同时间光学和历史光学进行标注；仅标建筑建造、拆除、重建，不标植被季节变化。（PDF pp.3-4 Sec.3.2、Fig.1）
- 输出是二值建筑变化区域；变化边界取双时相建筑的最外边界，允许误差不超过 3 像素，并由多名专家交叉校验。（PDF p.4 Sec.3.2）

### HTCD 外部验证

- HTCD 是 2008-2020 卫星-UAV异构变化数据，覆盖约 36 km²；卫星分辨率 0.5971 m、UAV 7.465 cm，切成 256×256，训练 3080 对、测试 1321 对。（PDF p.6 Sec.5.1）

## 3. 模型结构

```text
光学 RGB ── Transformer-O（独立参数）── {O1,O2,O3,O4}
                                                │
                                                ├─ FCA：约束 O3/O4 与 S3/S4
                                                │
SAR HH/VV/HV ─ Transformer-S（独立参数）── {S1,S2,S3,S4}
                                                │
              每尺度 3D-STA(Oi,Si) → 自顶向下金字塔融合
                                                ↓
                                      二值建筑变化图
```

整体结构见 PDF p.5 Fig.2；数学总览见 PDF p.4 Sec.4.1、Eq.(1)-(3)。

### 3.1 非共享双编码器

- 两个编码器结构相同但参数独立，因此每个模态可学习自己的浅层纹理与深层语义分布；“non-shared”指**不共享权重**，不是两套完全不同的网络架构。（PDF p.1 摘要；p.4 Sec.4.1）
- 官方代码中每个分支都是四阶段层次 Transformer，通道 `[64,128,320,512]`，深度 `[3,4,6,3]`，由 overlap patch embedding、空间压缩自注意力和带 DWConv 的 MLP 构成。（官方代码 `models/hetecd.py::Backbone`、`hetecd`）
- 官方模型默认两路 `input_nc=3`。这与论文使用 RGB 和三种 SAR 极化一致，但不能直接证明原生 1 通道 SAR 输入头有效。（PDF p.6 Sec.5.1；官方代码 `models/hetecd.py::hetecd`）

### 3.2 FCA：高层特征一致性对齐

论文在高层特征上同时约束空间维分布一致性和通道维分布一致性。两者均通过带温度平滑的 KL divergence 实现，使同类语义在光学与 SAR 高维空间中更接近。（PDF p.6 Sec.4.3、Fig.4、Eq.(8)）

总损失为：

```text
L = Lseg + (2 / epoch) × Lfca
Lseg = CrossEntropy + Dice
```

即 FCA 前期较强、随训练衰减，后期让变化分割目标主导。（PDF p.6 Sec.4.3、Eq.(7)-(9)）

官方训练代码返回两路 Stage3/4 特征，执行 `CE + Dice + 2/(epoch+1) * AlignmentLoss`。代码没有用标签生成“未变化掩膜”，因此 FCA 实际是**全图高层对齐**，真实变化区域也会被约束；这是迁移到灾害损毁任务时必须重新验证的风险。（官方代码 `utils/utils_fit.py::train`、`utils/loss.py::AlignmentLoss`）

### 3.3 3D Spatio-temporal Attention Difference

该模块不是简单做 `|O-S|`，具体流程为：

1. 从两路特征交替取通道：光学放偶数位置，SAR 放奇数位置；
2. 将相邻 3 个通道视作一个“伪时间序列”；
3. 并行使用 `3×3×3`、`3×5×5`、`3×7×7` 三种 3D 卷积提取多尺度差异；
4. 用 Conv1D + Softmax 生成 temporal attention；
5. 用 `7×7` Conv2D + Sigmoid 生成 spatial attention；
6. 融合为该尺度的变化特征。（PDF p.5 Sec.4.2、Fig.3、Eq.(4)-(6)）

它的目的不是先把图像几何配准，而是在局部空间上下文和“模态/时相”轴上共同学习差异，避免 SAR 高亮散射直接触发误报。（PDF pp.5、8-9 Sec.4.2、Sec.5.3、Fig.6）

### 3.4 多尺度解码

- 四个尺度都运行 3D-STA；深层差异逐级上采样并加到浅层，最终把四尺度差异统一到浅层分辨率后拼接，通过 `1×1 Conv + residual blocks + prediction head` 输出二值图。（PDF p.4 Sec.4.1、Fig.2；官方代码 `CD_3D_Decoder.forward`）

## 4. 训练设置与损失

| 项目 | 设置 | 证据 |
|---|---|---|
| 优化器 | AdamW | PDF p.6 Sec.5.1 |
| 初始学习率 | `5e-4` | PDF p.6 Sec.5.1 |
| Epoch | 200 | PDF p.6 Sec.5.1 |
| 预训练 | 两路 backbone 均加载 ImageNet 权重 | PDF p.6 Sec.5.1 |
| 增强 | 随机裁剪、翻转、颜色变换、旋转 | PDF p.6 Sec.5.1 |
| 监督 | Cross Entropy + Dice + 衰减 FCA | PDF p.6 Sec.4.3、Eq.(7)-(9) |

## 5. 结果与消融

### 主结果

- XiongAn：Precision 81.18、Recall 80.02、F1 80.60、IoU 67.50；参数 50.19M、FLOPs 58.23G、推理 156.44 ms。IoU 比第二名 BAN 的 66.19 高 1.31 pp。（PDF p.7 Table 2）
- HTCD：F1 94.34、IoU 89.30，是表中最高值。（PDF p.7 Table 3）

### 消融（XiongAn）

| Non-shared | 3D-STA | FCA | F1 | IoU | 解读 |
|---|---|---|---:|---:|---|
| 否 | 否 | 否 | 77.28 | 62.97 | 基础对照 |
| 是 | 否 | 否 | 77.58 | 63.29 | 仅不共享权重，小幅提升 |
| 是 | 是 | 否 | 78.88 | 65.11 | 3D-STA 带来主要增益 |
| 是 | 否 | 是 | 78.53 | 64.48 | FCA 单独有效 |
| 是 | 是 | 是 | 80.60 | 67.50 | 二者互补 |

表格来自 PDF p.8 Table 4；正文明确指出 3D-STA 将 IoU 从 63.29 提到 65.11，FCA 加入完整模型后达到 67.50。（PDF pp.8-9 Sec.5.4、Fig.6-7）

## 6. 论文明确提出的局限

- 计算效率不是最优：50.19M 参数、58.23G FLOPs、156.44 ms，速度明显慢于 Changer；作者提出未来需优化效率以适配资源受限环境。（PDF pp.7-8 Sec.5.3、Table 2）
- 小而不显著的变化仍容易漏检，作者把“细微异构变化”列为未来重点。（PDF p.10 Sec.6）
- 任务只输出建筑是否变化，不区分 intact / damaged / destroyed；这是由其数据标注和二分类输出定义决定的，不是论文实验已解决的问题。（PDF pp.3-4 Sec.3.2；p.4 Sec.4.1）
- 论文没有展示跨灾害事件留一验证；XiongAn 主要是城市建设/拆除，不能直接等价于灾后结构损毁。（PDF pp.2-4 Sec.1、Sec.3）

## 7. 论文与官方代码的关键对应

| 论文模块 | 官方实现位置 | 复现时要注意 |
|---|---|---|
| 非共享 Transformer | `models/hetecd.py::hetecd.backbone1/backbone2` | 两路同构、权重独立，默认都为 3 通道 |
| FCA | `utils/loss.py::AlignmentLoss` | 代码是 Stage3/4 全图对齐，不是只对齐未变化区域 |
| 3D-STA | `models/hetecd.py::cross_fuse_3d_atten` | 三组 3D kernel + temporal/spatial attention |
| 金字塔差异解码 | `models/hetecd.py::CD_3D_Decoder` | 四尺度均做差异挖掘，计算量不低 |
| 总损失 | `utils/utils_fit.py::train` | `CE + Dice + 2/(epoch+1)*AlignmentLoss` |

## 8. 对 BRIGHT 课题的准确启示（解释性结论）

- 可以迁移的是：**非共享编码**和“高层再交互”的原则；它直接反驳“只给共享 RGB 主干换一个 1ch 输入头”的做法。
- 不能照搬的是：三极化 SAR 输入、全图 FCA、二值变化头以及建设/拆除数据协议。BRIGHT 是单通道灾后 SAR + 多级损毁，尤其 Damaged 的微弱变化正好落在论文承认的困难区。
- 因此 HeteCD 更适合作为“异构编码与差异挖掘基线”，不应被直接当作完整建筑损毁模型。

## 9. 原文证据索引

| 证据位置 | 内容 |
|---|---|
| PDF p.1 Abstract | 研究问题、HeteCD、FCA、3D-STA、IoU 67.50 |
| PDF pp.3-4 Sec.3、Fig.1、Table 1 | XiongAn 数据、传感器、预处理、标注 |
| PDF pp.4-5 Sec.4.1-4.2、Fig.2-3 | 整体结构与 3D-STA |
| PDF p.6 Sec.4.3、Fig.4、Eq.(7)-(9) | CE、Dice、FCA 与权重衰减 |
| PDF pp.6-7 Sec.5.1-5.3、Table 2-3 | 训练设置与主结果 |
| PDF pp.8-9 Sec.5.4、Table 4、Fig.6-7 | 模块消融与可视化 |
| PDF p.10 Sec.6 | 结论与细微变化局限 |
