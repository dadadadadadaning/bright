# SP-E0：固定 SLIC 与建筑内跨模态图关系差异设计

- 用户研究方向：灾前 RGB、灾后 SAR 的建筑损毁分级（Intact / Damaged / Destroyed）
- 用户研究问题：固定超像素与模态内结构关系差异，能否补充 P3/H6 对 Damaged 的漏检，而不破坏现有定位和严重度边界？
- 分析日期：2026-09-04
- 本地证据：P3/MICB-DSMIL、H6-A2、H13-SPER、BIOS-E0、IM4CD 受控原型
- 外部证据：CGSL（ISPRS JPRS 2025）、SSN、SpixelFCN 等区域/图结构方法

## 1. 结论

SP-E0 不应直接设计成一个新的大型 GNN，也不应把 SLIC、局部匹配、Memory 和 MoE 同时加入。最小、可归因的研究问题是：

> 在冻结 P3 与 H6-A2 的条件下，用灾前 RGB 生成固定共享超像素，并分别在 RGB 与 SAR 特征中计算同一张区域图上的模态内关系；这些关系的变化能否识别 P3 漏判的 Damaged 建筑？

本仓库已经存在两个相邻但不相同的原型：

1. `H13-SPER`：RGB 引导软超像素，区域内聚合的是 RGB、SAR、绝对差、乘积和局部统计，仍包含直接跨模态特征比较。
2. `BIOS-E0`：在 7×7 建筑 ROI 中比较 RGB/SAR 各自的自相似关系，但使用规则 token 邻域，而不是真正的 SLIC 区域图。

因此真正有新增价值的 SP-E0 是两者的交集：**固定 SLIC 分区 + 共享拓扑 + 模态内关系差异**。由于仓库已有审计认为 H12/H13/H14 的统计区域化尚不能稳定预测跨事件纠错，SP-E0 必须先做无参数证据审计，不能直接进入完整训练。

## 2. 模块边界

### 保持不变

- P3 的 RGB 主干、建筑定位、像素级损毁解码、MICB/DSMIL。
- H6-A2 的 RGB/SAR 非共享编码及现有 checkpoint。
- CMR Memory；SP-E0 第一阶段不连接 Memory。
- Damaged 与 Destroyed 的已有严重度边界。

### SP-E0 只负责

- 在 P3 建筑 ROI 内形成固定区域单元；
- 描述 RGB 与 SAR 各自的区域关系；
- 输出一个 `Affected occurrence evidence`，即只回答 Intact 与 Affected，不直接回答 Damaged 与 Destroyed。

### SP-E0 不负责

- SAR 配准或图像 warping；
- 全图 RGB/SAR 特征对齐；
- 重新学习超像素边界；
- 直接修改建筑定位；
- 用验证/测试标签生成推理 ROI。

## 3. 总体数据流

```text
Pre RGB ── P3 localization ── detached building proposals ───────┐
   │                                                            │
   ├── fixed RGB-SLIC map ── crop/intersect with each proposal ─┤
   │                                                            │
   └── H6 RGB Stage4 ── projection ── region pooling ── RGB graph

Post SAR ── H6 SAR Stage4 ── projection ── same region pooling ── SAR graph

RGB graph relation A_R ─┐
                        ├── relation discrepancy E ── building readout
SAR graph relation A_S ─┘                              │
                                                      ▼
                                             occurrence residual δ
                                                      │
P3 damage logits ─────────────────────────── safe hierarchical fusion
```

训练时可以用 GT 建筑实例/连通区域形成监督 bag，但推理时只能使用冻结 P3 proposal。GT ROI 只用于信号上限与训练监督，不得进入部署路径。

## 4. 固定 SLIC 与共享分区

### 4.1 为什么只在灾前 RGB 上做 SLIC

灾前 RGB 边界清晰，且不存在灾后破坏造成的结构缺失；SAR 上直接分割容易受散斑、阴影和叠掩影响。如果 RGB 与 SAR 各自独立 SLIC，节点没有一一对应关系，后续关系差异会混入分区差异。因此采用一个 RGB 生成的分区，同时池化两种模态。

### 4.2 推荐实现

在数据加载阶段或离线缓存阶段，对原始/增强同步后的灾前 RGB 计算 SLIC：

```text
颜色空间       Lab（只用于 SLIC）
连通性         enforce_connectivity=True
起始标签       0
输入           灾前 RGB
共享对象       RGB 与 SAR 特征共用同一 labels map
梯度           无；labels 完全 detach
```

第一轮不要只固定一个超像素数量后直接宣称成功。应预注册一个主尺度和两个诊断尺度，例如以近似步长而不是固定绝对数量表达：`8 / 12 / 16` 输入像素。主尺度在查看验证标签前确定，其余只用于稳定性诊断。

每次几何增强必须同步作用于 RGB、SAR、标签和 SLIC map。颜色增强发生在 SLIC 计算之后或只作用于模型 RGB，避免同一图像因颜色抖动产生不同图结构。缓存清单记录图像 ID、SLIC 参数、实现版本和哈希。

### 4.3 ROI 与超像素相交

全图超像素与 P3 建筑 proposal 相交，保留建筑覆盖率达到阈值的节点。边界节点不应简单删除，而是携带 `building_fraction` 作为可靠性：

\[
q_k=\frac{|S_k\cap B|}{|S_k|},\qquad
w_k=\operatorname{clip}\left(\frac{q_k-q_{min}}{1-q_{min}},0,1\right).
\]

区域池化只使用交集 `S_k ∩ B`，从而避免背景主导小建筑；`q_k` 则进入节点可靠性和边权。少于 4 个有效节点的建筑不走 SP-E0，严格回退 P3。

## 5. Stage4 特征与区域节点

### 5.1 特征来源

主实验读取冻结 H6-A2 的：

- `stage4_rgb_common`；
- `stage4_sar_common`。

但 Stage4 语义强、空间分辨率低，可能吞掉局部 Damaged。因此应把 Stage3 设为预注册对照而不是事后补救。已有 BIOS-E0 实际把 Stage3 local relation 设为主假设，Stage4 仅作次要诊断，这一点必须尊重。

### 5.2 投影不能强迫跨模态数值相等

分别使用轻量投影：

\[
R=\operatorname{norm}(P_R(F_R)),\qquad
S=\operatorname{norm}(P_S(F_S)).
\]

`P_R` 与 `P_S` 不共享参数。SP-E0 不添加 `R≈S` 的全图对齐损失。无参数 E0 甚至可以跳过可学习投影，直接对冻结 common features 做通道归一化，以免把读出器能力误当成关系证据。

对超像素节点 k，利用面积权重池化均值和离散度：

\[
r_k=\frac{\sum_p Q_{pk}R_p}{\sum_p Q_{pk}},\qquad
s_k=\frac{\sum_p Q_{pk}S_p}{\sum_p Q_{pk}}.
\]

节点附加属性只保留与可靠性有关的量：区域面积、建筑覆盖率、RGB 边界强度、SAR 内部方差。P3 类别概率在 E0 中不得进入关系分数，只用于定义 gate-miss 子集；否则会产生循环证据。

## 6. 图构建

### 6.1 节点与边

- 节点：一个建筑 ROI 内与建筑相交的 SLIC 区域。
- 近边：空间接触的一阶邻接边。
- 远边：可选的二跳边或同建筑内距离最远的少量节点，用于表达 RIEM/GCLN 式近远关系。
- 第一轮最多使用一阶邻接；远边只能作为独立消融，不能一开始混入。

两种模态使用完全相同的节点集合和边集合，拓扑由 RGB-SLIC 决定。这样比较的是边关系值变化，而不是两张不同图的节点匹配问题。

### 6.2 模态内关系

对每条共享边 `(i,j)`：

\[
a^R_{ij}=\cos(r_i,r_j),\qquad
a^S_{ij}=\cos(s_i,s_j).
\]

核心关系差异：

\[
e_{ij}=|a^R_{ij}-a^S_{ij}|.
\]

推荐同时保留符号量但不作为主分数：

\[
\tilde e_{ij}=a^R_{ij}-a^S_{ij}.
\]

绝对量回答“关系改变多少”，符号量回答“灾后结构变得更一致还是更破碎”。主 E0 只使用绝对量，避免增加自由度。

边可靠性：

\[
\rho_{ij}=\sqrt{w_iw_j}\cdot
\exp(-\beta b^{RGB}_{ij})\cdot c^{match}_{ij}.
\]

其中 `b_RGB` 是灾前强边界强度；`c_match` 在纯 SP-E0 中固定为 1，只有将来与 IM4CD 合并时才由匹配置信度提供。不要在 SP-E0 中提前引入 IM4CD。

### 6.3 节点异常与建筑描述子

\[
d_i=\frac{\sum_{j\in\mathcal N(i)}\rho_{ij}e_{ij}}
{\sum_{j\in\mathcal N(i)}\rho_{ij}+\epsilon}.
\]

建筑级最小描述子建议固定为：

```text
mean(edge discrepancy)
std(edge discrepancy)
q90(edge discrepancy)
mean(node discrepancy)
max(node discrepancy)
changed-edge ratio at a threshold learned only from training Intact
valid-node count / valid-edge count
```

E0 主分数仍应只有一个：可靠性加权 edge mean。其余均为次要诊断，避免验证集挑统计量。

## 7. 两阶段训练与准入

### Phase A：SP-E0 无参数证据审计

冻结全部模型，不改变 logits，不访问测试集。分别在 GT ROI 和 P3 ROI 报告：

- Intact vs Affected AUROC/AUPRC；
- P3 预测 Background/Intact 子集中的真实 Damaged AUROC/AUPRC；
- Damaged vs Destroyed（只作反证，预计不应是主优势）；
- 每事件 AUROC、事件宏平均、最差事件和方向一致事件数；
- SLIC 图与等节点数规则网格图的对照；
- 同位直接特征差的对照；
- SAR 特征平移 ±1/±2 token、空间打乱压力测试。

建议沿用 BIOS-E0 的硬门槛：P3 gate-miss AUROC ≥ 0.60、事件宏 AUROC ≥ 0.58、至少 10/14 事件方向正确、关系分数在 0/±1/±2 平移下优于直接差异、GT/P3 ROI AUROC 差不超过 0.10。额外加入：SLIC 必须稳定优于等预算规则网格；否则贡献只能写成“关系差异”，不能写“超像素图”。

### Phase B：SP-E1 可学习建筑读出

只有 Phase A 通过才训练。输入为固定图统计或节点/边集合，读出器使用小型 DeepSets/attention pooling：

\[
h_i=\phi([d_i,r_i^{stat},s_i^{stat},q_i]),\qquad
h_B=\sum_i \alpha_i h_i,\qquad
\delta_B=g(h_B).
\]

不建议第一轮使用多层 GAT。最多一层消息传播，并与“无消息传播的 DeepSets”直接对照；若 GNN 不优于集合读出，就不应声称图传播有效。

### Phase C：安全注入 P3

SP-E1 只修正 Affected occurrence：

\[
z'_{dam}=z_{dam}+\alpha M_B\delta_B,
\qquad
z'_{des}=z_{des}+\alpha M_B\delta_B.
\]

Background 与 Intact logits 不直接加残差；`M_B` 是 detached P3 建筑支持。因为 Damaged 与 Destroyed 同加相同残差：

\[
z'_{des}-z'_{dam}=z_{des}-z_{dam},
\]

所以原有严重度边界严格保持。`α=tanh(γ)α_max`，`γ=0` 初始化，使新模型训练前与 P3/H6 基线完全等价。置信度低、节点不足或图异常时令 `α=0` 回退基线。

## 8. 损失函数

第一轮只保留三个目标：

\[
L=L_{base}+\lambda_{bag}L_{affected}+\lambda_{rank}L_{rank}.
\]

- `L_base`：原 P3/H6 像素损失；若主体冻结，可只用于一致性核验。
- `L_affected`：建筑级 Intact vs Affected 的 class-balanced BCE/focal loss。Damaged 与 Destroyed 都映射为 Affected。
- `L_rank`：小权重排序约束，让 Affected 建筑的图异常高于 Intact；只在训练事件内采样，不跨测试事件。

不在首轮加入重建损失、KL commonality、跨模态对齐、边级伪标签或 Memory loss。它们会让增益无法归因，也可能抹掉真实损毁差异。

## 9. 受控实验矩阵

| 编号 | 唯一变化 | 回答的问题 |
|---|---|---|
| B0 | 冻结 P3/H6-A2 | 锁定基线与 checkpoint |
| G0 | 等节点数规则网格 + 模态内关系差 | 收益是否只来自区域池化 |
| G1 | 固定 RGB-SLIC + 直接跨模态节点差 | SLIC 本身是否有用 |
| G2 | 固定 RGB-SLIC + 模态内关系差 | 核心 SP-E0 假设 |
| G3 | G2 + DeepSets 建筑读出 | 关系证据能否转成预测 |
| G4 | G3 + 一层邻接消息传播 | 图传播是否额外有效 |
| G5 | G3 + Stage3/Stage4 双尺度 | 深浅结合是否必要 |
| G6 | G3 + IM4CD 匹配置信度边权 | 仅在 G3 已跨事件通过后运行 |

关键归因：`G2 > G0` 才能说明 SLIC 优于规则区域；`G2 > G1` 才能说明关系比较优于直接特征差；`G4 > G3` 才能说明消息传播而非集合池化有效。

## 10. 评估协议

模型选择不能只依赖当前同事件随机 validation。至少同时报告：

1. 标准 validation：与历史实验可比。
2. 4-fold event-held-out：核心选择协议。
3. 若资源允许，leave-one-event-out 或 leave-one-disaster-type-out。

主指标：Damaged IoU、Affected recall、Damaged gate-miss、四类 mIoU。稳定性指标：event mean、event worst、14 个事件非负数、两粒种子方向一致性。建议晋级门槛：event-held-out Damaged IoU `+0.5 pp`、gate-miss `-0.5 pp`、event worst 不下降、至少 10/14 事件非负。

Test 只在结构、尺度、阈值、损失权重与 checkpoint 选择全部锁定后访问一次。

## 11. 主要风险与停止条件

1. **Stage4 太粗**：小建筑少于 4 个有效节点时图无意义。若大比例建筑触发回退，主版本改用 Stage3，而不是增加复杂 GNN。
2. **SLIC 与建筑边界不一致**：若 SLIC 不优于等预算网格，停止“超像素贡献”叙事，保留纯关系差异。
3. **关系差异仍然事件相关**：若标准验证有效但 event-held-out 失败，停止训练版 SP-E1，不用 Memory 掩盖失败。
4. **只在 GT ROI 有效**：说明瓶颈是 proposal/实例分离；应转向建筑 query/ROI，而不是继续调图参数。
5. **只提高 Destroyed**：与当前缺口不匹配；SP-E0 的主目标是 Damaged gate-miss。
6. **图读出关闭自身**：若零初始化 transition 收敛到近零，说明图证据没有独立贡献，不能靠增加层数挽救。
7. **触碰 severity 边界**：首轮禁止分别修正 Damaged/Destroyed；否则 occurrence 与 severity 两个问题混在一起。

## 12. 与现有代码的最小接入点

建议新增独立实验族，不修改已有 H13/BIOS 文件：

```text
bda_benchmark/model/DamageFormer_RGBSAR_SP_E0.py
bda_benchmark/model/sp_e0/slic_regions.py
bda_benchmark/model/sp_e0/relation_graph.py
bda_benchmark/model/sp_e0/building_readout.py
bda_benchmark/script/standard_ML/audit_SP_E0_relation_graph.py
bda_benchmark/script/standard_ML/train_SP_E1_readout.py
bda_benchmark/configs/sp_e0_preregistered.json
bda_benchmark/tests/test_sp_e0.py
```

可复用：

- P3 的 `BuildingBagSpec`、训练/推理 bag 构建和 MICB/DSMIL 建筑级接口；
- H6-A2 的 `stage3/4_rgb_common` 与 `stage3/4_sar_common` 输出；
- BIOS-E0 的 AUROC、事件统计、平移/打乱压力测试；
- H13 的区域池化/回填思想，但不要复用其“直接 RGB-SAR difference”作为核心关系；
- IM4CD 的 `matching_confidence` 仅用于最后的 G6。

首批单元测试至少覆盖：共享分区、边集合一致、模态相同时关系差为零、SAR 节点置换使异常升高、少节点回退、零初始化输出严格等价基线、Damaged/Destroyed logit 差严格不变。

## 13. 创新与迁移评估

| 维度 | 判断 |
|---|---|
| 任务匹配 | 高：直接面向 RGB-SAR 建筑损毁 |
| 数据匹配 | 高于一般 MCD：使用 BRIGHT 的建筑 ROI 和三级标签 |
| 可迁移模块 | SLIC 区域、共享图拓扑、模态内关系差 |
| 创新度 | 中等偏高，前提是证明跨事件 Damaged 改善；仅在普通 val 提升则偏低 |
| 实现难度 | E0 中等，E1 中等，IM4CD 联合版中高 |
| 最大风险 | Stage4 空间分辨率、P3 ROI 误差、事件相关性、与 H13 贡献重叠 |

## 14. 建议题目方案

1. **Optical-Anchored Superpixel Relation Graphs for Cross-Modal Building Damage Assessment**
   - 任务：BRIGHT RGB-SAR 三级建筑损毁。
   - 方法：灾前 RGB 固定 SLIC、共享图拓扑、模态内关系差、层级 occurrence 注入。
   - 最小实验：G0-G3 + event-held-out。
   - 预期贡献：证明结构关系比同位跨模态特征差更适合识别中度损毁。

2. **Building-Internal Organization Shift for Reliable Optical-SAR Damage Recognition**
   - 任务：P3 漏判 Damaged 的建筑级纠错。
   - 方法：建筑 ROI 内组织关系变化，不做跨模态特征强对齐。
   - 最小实验：BIOS-E0/SP-E0 审计 + 保守 occurrence readout。
   - 预期贡献：把图关系定位为选择性补充证据，而非全图融合器。

3. **Event-Robust Region Relation Learning for All-Weather Building Damage Mapping**
   - 任务：未知事件/灾种上的 Damaged 泛化。
   - 方法：SP-E1 与 event-held-out episodic training。
   - 最小实验：4-fold event split、event worst、跨种子方向一致性。
   - 预期贡献：从同事件精度转向跨事件关系稳定性。

4. **Confidence-Weighted Local Matching and Region Graph Reasoning for Optical-SAR Damage Assessment**
   - 任务：处理局部错位下的区域关系比较。
   - 方法：先验证 SP-E1，再用 IM4CD 式响应峰置信度调节图边权。
   - 最小实验：G3 vs G6、平移压力测试。
   - 预期贡献：把局部匹配用于“判断边是否可信”，而非直接把 offset 当损毁。

## 15. 证据边界

- CGSL 支持“超像素图、共享结构空间和结构差异可用于异构变化检测”，但其任务是无监督二值变化，不能直接证明 BRIGHT 的 Damaged 会提升。
- H13 的负面或弱增益结果说明“区域化本身”不是充分条件；SP-E0 必须通过 `SLIC relation > grid relation > direct feature difference` 的因果链。
- BIOS-E0 的主假设目前是 Stage3 规则 token 局部关系，而非 Stage4 SLIC；若其完整验证未通过，应优先停止或重新定位关系路线，而不是绕过门槛训练 SP-E1。
- IM4CD 风格匹配只有在 SP-E1 独立有效后才有资格作为可靠性增强，不能与 SP-E0 同时首发。

