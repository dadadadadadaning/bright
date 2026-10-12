# 伪实例建筑损毁语义分割：2024–2026模块迁移审计

- 用户研究方向：灾前RGB与灾后SAR建筑损毁语义分割；利用无官方实例监督的伪建筑结构改善P3像素预测。
- 用户研究问题：三个“即插即用”代码包中哪些模块真正适合P3、LocCC、RGID与Selective Context主线，应该放在哪里，如何通过受控实验验证。
- 分析日期：2026-09-04
- 本地证据：`③ 2024年.zip`、`② 2025年(全).zip`、`2026年最新！即插即用代码（VIP）.zip`及当前BRIGHT源码。
- 当前锁定证据：P3=60.5020%；LocCC+Hard A6=61.5399%；LocCC+Soft Context=61.5716%；RGID尚未超过LocCC。

## 1. 源码包审计结论

| 包 | Python文件 | PDF | 结论 |
|---|---:|---:|---|
| 2024全年 | 67 | 135 | 包含RCM、CAA、FreqFusion、FADC、MDAF等候选 |
| 2025全年 | 118 | 239 | 包含LCA、ConDSeg聚合、Strip、ARConv等候选，也有大量重型架构 |
| 2026 VIP | 30 | 68 | 主要是2024/2025模块子集，加少量2026片段；不是30个全新模块 |

三包共有215份Python文件，但按SHA256只有192份不同内容。VIP中的Strip、RCM、LCA、ARConv、DSA、FAM、TSSA、HPA等与全年包完全重复。所有源码均通过语法编译，只能证明语法成立，不能证明：

1. 与原论文官方实现一致；
2. 能在当前PyTorch/CUDA环境导入；
3. 前向、反向、AMP与非方形输入正确；
4. 对RGB–SAR损毁任务有效；
5. 具备可用于论文的开源许可。

发现的代表性风险：

- `MaskAttention`在前向中随机生成掩码，并把`[B,HW,C]`直接view回`[B,C,H,W]`，没有先permute，不能使用。
- `EdgeGaussianAggregation`首先做固定高斯低通，代码没有显式高通差分，因此名称中的“Edge”不能按边缘模块解释。
- `DeformableInteractiveAttention`没有offset/grid/deformable sampling，只是池化、卷积和上采样门控；名称大于实现。
- `UAFF`用第三路`x1`计算门权重，但输出只融合`x0`与`outx`，并不是对三个输入对称融合。
- 压缩包版`FreqFusion`强依赖`mmcv.ops.carafe`；官方仓库后来提供了无mmcv版本，应以官方版本为准。
- 大量模块依赖`timm/einops/natten/mamba_ssm/mmcv/torch_dct/pytorch_wavelets/triton`，会增加4090环境风险。

## 2. 当前应用真正缺少什么

当前并不缺少普通注意力。现有仓库已经包含：

- P3的RGB/SAR编码、跨模态融合、建筑ROI权重与像素损毁头；
- MICB-DSMIL Bag头，但锁定P3的`mil_residual_scale=0`，不会参与像素决策；
- LocCC、A6门控、Hard/Soft/Uncertain Interior反馈；
- DCMF双向窗口交叉注意力与公共/私有解耦；
- SAR可靠性门控、SAR残差适配、流式/局部可变形对齐；
- SRAC区域分类与区域记忆；
- RGID Center/Boundary、Watershed、Gate与Rollback。

实验暴露的真正缺口是：

1. **区域建议如何安全作用于像素**：硬广播有效但有事件下降；固定Soft只比Hard高0.0317 pp。
2. **区域内部与周围背景如何形成对比证据**：现在主要使用区域均值/多数，缺少core–ring差异。
3. **边界如何保护**：热点经常沿建筑边界，边界不能被当作损毁证据直接强化。
4. **RGB与SAR可靠性如何按区域选择**：RGB屋顶纹理和SAR强散射都可能产生伪证据。
5. **RGID为何值得拆分**：当前多拆857个区域，但语义净修正少于LocCC，说明“切得更细”不等于“语义更好”。

因此模块选择必须服务这五个缺口，而不是堆叠注意力。

## 3. 模块迁移优先级

### A级：值得进入受控实验

| 模块/思想 | 原任务 | 建议位置 | 迁移方式 | 难度 | 主要风险 |
|---|---|---|---|---|---|
| ConDSeg前景/背景/不确定区域解耦 | 医学分割 | PI-SRA区域Token | 不复制重型CDFA；改成伪区域core、外环ring和不确定像素三路Token | 中 | 医学域迁移；必须有Local对照 |
| RCM/RCA | 高效语义分割 | 区域残差Adapter空间混合器 | 只在LocCC bbox/低分辨率ROI内运行，输出零初始化残差 | 低–中 | 横纵轴偏置，旋转建筑可能受限 |
| CAA | 遥感旋转目标检测 | RCM的轻量替代对照 | 大水平/垂直深度卷积生成区域注意力 | 低 | 仍是轴对齐；可能吸收背景 |
| FreqFusion | 稠密预测多尺度融合 | 后期边界恢复支线 | 仅在可取得相邻尺度特征时替换一次上采样 | 高 | 需改P3特征接口；SAR高频可能是斑点 |
| StripModule | 遥感目标定位 | RGID Center/Boundary Head | 只替换Head空间块，损失和解码全部锁定 | 低 | 只改善几何；不能自动证明语义收益 |

### B级：在主线通过后再测试

| 模块 | 判断 |
|---|---|
| MDAF | 来自遥感空间–频率融合，双输入接口有价值；但当前仓库已有更贴合RGB/SAR的DCMF，应作为交叉模态消融，不应再成为主模块。 |
| Lightweight Cross Attention | 可用于区域级RGB/SAR交互，但原任务是低光双分支增强；且仓库已有双向窗口交叉注意力。优先复用现有DCMF。 |
| ContrastDrivenFeatureAggregation原块 | 思想高度匹配core/ring对比，但原代码需要现成fg/bg特征图且局部注意力较重。建议迁移思想而非整块复制。 |
| FADC/ARConv | 可能改善尺度和不规则边缘，但依赖自定义算子或动态采样，因果链和复现成本高。仅当边界审计表明尺度错位是主瓶颈时使用。 |
| MHCB/PKI/LSK | 可做多尺度空间块备选，但与RCM/CAA回答相同问题，只留一个参数匹配对照。 |

### C级：当前拒绝

`MaskAttention`、`EdgeGaussianAggregation`原代码、`DeformableInteractiveAttention`原代码、STAtten、F_Block、整套Mamba/OverLoCK/LSNet、随机频域模块、StarReLU等。原因分别是实现错误/名实不符、任务不匹配、依赖重、改变整套骨干、或无法回答当前科学问题。

## 4. 推荐主模型：Core–Ring PI-SRA

不要把若干模块串联成“模块拼盘”。建议形成一个统一机制：**伪实例条件的核心–外环对比区域残差适配器**。

```text
灾前RGB ── P3 RGB Encoder ── F_rgb ── Localization ── LocCC区域 R_j
    │                                      │
    │                                      ├─ core: 腐蚀后的稳定建筑内部
    │                                      ├─ boundary: 建筑边界带
灾后SAR ── P3 SAR Encoder ── F_sar         └─ ring: 区域外邻域
    │
    └──── P3 fusion ── F_roi ── P3 logits Z0
                              │
                       Core/Ring Token
                              │
                   RCM（CAA为匹配对照）
                              │
                 像素—区域关系残差 ΔZ
                              │
     熵/纯度/间隔/边界距离/定位/RGB-SAR分歧 ── Risk Gate g
                              │
                       Z = Z0 + g·ΔZ
```

### 4.1 区域划分

对LocCC区域`R_j`：

\[
R_j^{core}=\operatorname{erode}(R_j,r_c)
\]

\[
R_j^{ring}=\operatorname{dilate}(R_j,r_o)\setminus R_j
\]

边界带为：

\[
R_j^{bd}=R_j\setminus R_j^{core}
\]

### 4.2 模态专用核心与外环Token

对模态`m∈{rgb,sar,fused}`：

\[
v_{j,core}^{m}=\frac{\sum_{x\in R_j^{core}}w_j(x)F_m(x)}
{\sum_{x\in R_j^{core}}w_j(x)+\epsilon}
\]

\[
v_{j,ring}^{m}=\frac{\sum_{x\in R_j^{ring}}F_m(x)}{|R_j^{ring}|+\epsilon}
\]

核心–背景对比：

\[
d_j^m=v_{j,core}^m-v_{j,ring}^m
\]

这比区域均值更适合当前问题：它要求证据相对于建筑周围环境发生变化，而不是仅凭屋顶纹理或SAR亮点。

### 4.3 区域残差

对区域内部像素：

\[
h_j(x)=[F_{roi}(x),v_{j,core},F_{roi}(x)-v_{j,core},
F_{roi}(x)\odot v_{j,core},d_j]
\]

\[
\Delta Z_j(x)=f_{adapter}(h_j(x))
\]

`f_adapter`第一版使用参数量匹配的3×3深度可分卷积；第二版只把空间混合器换为RCM。输出最后一层零初始化，使初始模型严格等于P3。

### 4.4 风险门控

\[
g_j(x)=\mathbf1[x\in R_j]\cdot\sigma(f_g(z_j(x)))
\]

\[
z_j(x)=[H(p_0(x)),\rho_j,m_j,d(x,\partial R_j),P_B(x),
\operatorname{JS}(p_{rgb},p_{sar}),A_j]
\]

最终：

\[
Z(x)=Z_0(x)+g_j(x)\Delta Z_j(x)
\]

区域结构提供上下文和路由，不再向整栋建筑强制广播一个标签。

### 4.5 安全损失

\[
L=L_{semantic}+\lambda_{anchor}L_{anchor}+\lambda_{gate}L_{gate}
\]

在原P3高置信且正确、区域不可靠或边界位置使用锚定：

\[
L_{anchor}=\frac1{|\Omega_{safe}|}\sum_{x\in\Omega_{safe}}
KL(p_0(x)\Vert p(x))
\]

第一阶段冻结P3，只训练Adapter和Gate；不要重新做T1组件等权，也不要恢复MICB硬广播。

## 5. 受控实验矩阵

| 编号 | 唯一变化 | 回答的问题 |
|---|---|---|
| M0 | Frozen P3 | 锁定60.5020% |
| M1 | 相同步数普通微调 | 排除额外训练本身 |
| M2 | 参数匹配Local 3×3 Adapter | 排除参数量与局部平滑 |
| M3 | LocCC Token替代局部窗口 | 伪区域上下文是否有效 |
| M4 | M3 + core/ring contrast | 建筑—周围对比是否有效 |
| M5 | M4中3×3换RCM | RCM结构混合是否带来额外收益 |
| M5b | M4中3×3换CAA | 遥感条带上下文是否优于RCM |
| M6 | M5 + learned risk gate | 是否减少correct→wrong |
| M7 | M6 + 现有DCMF区域模态路由 | RGB/SAR可靠性是否有效 |
| M8 | M7中LocCC仅换RGID | 精细拆分是否提供额外价值 |

第一轮只跑M1–M4，各500 optimizer steps、seed=3407。M4通过后再测试RCM与CAA二选一；M5通过后才做M6。不要同时接RCM、FreqFusion、LCA和Strip。

建议预注册：

- `M3-M2`与`M4-M3`至少`+0.20 pp`；
- 最终相对P3至少`+0.50 pp`；
- Damaged F1相对P3至少`+1.0 pp`；
- 任一前景IoU下降不超过`0.50 pp`；
- 事件分层配对Bootstrap下界大于0；
- 至少10/14事件不下降；
- 三seed均值为正后才访问Test。

## 6. RGID几何支线如何使用Strip

RGID当前的监督矛盾是：Binary LocCC粘连parent通常只产生一个伪中心和外轮廓边界，但推理目标却希望从中拆出多个建筑。单纯换更强Head不能创造监督中不存在的内部中心，因此Strip只能作为次级实验：

| 几何实验 | 唯一变化 |
|---|---|
| G0 | LocCC |
| G1 | 当前RGID |
| G2 | Center Head加入Strip |
| G3 | Boundary Head加入Strip |

固定Center目标、Boundary目标、训练步数、阈值、Watershed、Gate、Rollback和A6。先看Center匹配、重复峰、Merge/Split、matched IoU，再看：

\[
\Delta_{geometry}=mIoU(G_k+A6)-mIoU(LocCC+A6)
\]

如果实例几何改善但语义不提高，只能声称“几何改善”，不能声称“损毁分割改善”。解决监督矛盾应优先于继续增加Head容量。

## 7. FreqFusion的正确位置

FreqFusion确实针对稠密预测中的类内不一致和边界位移，理论上贴合当前边界损失。但P3当前公开给主线的`rgb_feature/sar_feature/fused_roi`已经是融合后的同尺度特征；直接接入FreqFusion需要暴露相邻层级特征并改变`_encode_and_fuse`，会扩大因果范围。

因此：

1. 不进入第一轮PI-SRA；
2. 先做边界宽度分桶审计，确认错误集中于上采样边界；
3. 以官方无mmcv实现为基础，只替换一次上采样；
4. 用普通bilinear+3×3卷积作参数匹配对照；
5. 对SAR分支单独报告斑点区域的corrected/harmed，防止高频增强扩大伪散射。

## 8. 代码隔离建议

```text
bda_benchmark/experiments/spi_core_ring_adapter_v1/
├── README_ZH.md
├── experiment_contract.json
├── run.sh
├── model.py
├── regions.py
├── core_ring_tokens.py
├── risk_gate.py
├── modules/
│   ├── local_control.py
│   ├── rcm_adapter.py
│   └── caa_adapter.py
├── train.py
├── evaluate.py
└── tests/
    ├── test_p3_identity.py
    ├── test_no_gt_leakage.py
    ├── test_region_locality.py
    ├── test_core_ring.py
    ├── test_zero_init.py
    ├── test_amp_backward.py
    └── test_parameter_matching.py
```

不要把压缩包文件原样复制进`model/`。只迁移最小数学操作，补充原论文与官方仓库出处，移除公众号测试代码和不必要依赖。P3、LocCC和RGID保持单一事实来源。

## 9. 建议论文题目

1. **Pseudo-Instance Core–Ring Adaptation for RGB–SAR Building Damage Segmentation**
   - 核心：LocCC伪区域、core/ring对比Token、风险门控像素残差。
   - 最小证据：M3-M2、M4-M3、M6-M5。
2. **When Do Pseudo Buildings Help? Risk-Aware Structural Feedback for Multimodal Damage Mapping**
   - 核心：强调区域反馈有效条件与错误传播控制。
   - 最小证据：区域纯度/面积/边界/事件分层审计和Selective Context对照。
3. **Structure Before Instances: Label-Free Building-Region Context for Semantic Damage Assessment**
   - 核心：即使RGID不胜LocCC，也证明粗结构上下文已有效，精细实例不是必要前提。
   - 最小证据：LocCC对P3与局部窗口的结构特异性优势。
4. **Reliability-Routed Pseudo-Region Fusion for Pre-Event Optical and Post-Event SAR Damage Segmentation**
   - 核心：在Core–Ring Adapter成立后加入区域级RGB/SAR可靠性路由。
   - 最小证据：M7-M6及屋顶纹理/SAR强散射分组审计。

## 10. 原文与趋势证据

| 类型 | 来源 | 支撑内容 |
|---|---|---|
| 原文/官方 | PKINet, CVPR 2024, https://openaccess.thecvf.com/content/CVPR2024/html/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.html | CAA以多尺度卷积和长程上下文处理遥感尺度变化，但原任务是检测 |
| 官方代码 | CGRSeg, https://github.com/nizhenliang/CGRSeg | RCM来自高效语义分割，适合作为区域空间混合器候选 |
| 官方代码 | FreqFusion, https://github.com/Linwei-Chen/FreqFusion | 通过自适应低通、偏移重采样与高通处理类内不一致和边界位移；官方已提供无mmcv实现 |
| 官方代码 | ConDSeg, https://github.com/Mengqi-Lei/ConDSeg | 前景、背景、不确定区域解耦和对比聚合启发core/ring设计；原任务是医学分割 |
| 官方代码 | SFFNet, https://github.com/yysdck/SFFNet | MDAF用于遥感空间–频率表征对齐，但不是RGB–SAR模态对齐的直接证据 |
| 官方代码 | HVI-CIDNet, https://github.com/Fediory/HVI-CIDNet | LCA来自低光双分支增强；可迁移交互形式但存在显著任务差异 |

## 最终建议

主线只保留一个新的统一模型：`LocCC + Core/Ring Contrast + RCM/CAA二选一 + Risk Gate`。现有DCMF承担后续跨模态扩展，Strip只进入RGID支线，FreqFusion只在边界诊断成立后进入。其余模块暂不运行。

