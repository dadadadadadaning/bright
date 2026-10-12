# BRIGHT 伪建筑结构主线：下一步创新与修正路线

- 研究方向：灾前 RGB + 灾后 SAR 的建筑损毁语义分割与伪建筑区域纠错
- 当前基线：P3 `60.5020%` Valid mIoU
- 当前最好后处理：LocCC + Selective Context S2 `61.5716%`，相对 P3 `+1.0696 pp`
- 分析日期：2026-09-04
- 目标：把已经有效的建筑区域上下文转化为真正可学习、可部署、跨事件验证的创新方法

## 1. 总判断

当前最值得继续的不是 RGID、可微超像素、Memory 或更多 SAR 对齐模块，而是：

> **将完整分辨率 LocCC-S2 作为训练期结构教师，训练一个推理期不再读取 LocCC/RegionMaps 的软连接学生，并用“纠错是否真的有益”的方向感知效用路由器控制残差，最后在事件互斥协议上训练和选择。**

可暂定方法名：

> **Utility-Guided Optical-Anchored Connectivity Distillation（UOCD）**  
> 效用引导的光学锚定软连接蒸馏

它由三个必要模块组成：

1. `Structure Teacher`：冻结的完整分辨率 S2，只在训练时产生结构软目标；
2. `Soft Connectivity Student`：由 P3 软建筑概率和 RGB 特征形成连续连接关系，推理时不做 CPU 连通域；
3. `Direction-aware Utility Router`：预测某一类具体修改是否有益，而不是笼统预测“改/不改”。

事件稳健训练是实验协议与训练策略，不再额外包装成第四个复杂网络模块。

## 2. 现有证据怎样重新解读

| 结果 | 数值/现象 | 可以说明 | 不能说明 |
|---|---:|---|---|
| P3 | 60.5020 | 稳定语义基线 | 已利用好建筑级结构 |
| S1 Hard A6 | 61.5399，+1.0379 pp | 可靠区域意见有效 | 硬覆盖一定优于软融合 |
| S2 Soft Context | 61.5716，+1.0696 pp | 区域软上下文有效 | Soft 显著优于 Hard |
| S2-S1 | 仅 +0.0317 pp | 两者接近 | 软融合是核心创新 |
| Local Control | 60.5080，+0.0060 pp | 普通局部平滑几乎不能解释 S2 | 已证明所有事件都稳定 |
| RGID | 能拆更多区域但未超过 LocCC | 几何头能工作 | 精细实例拆分有语义价值 |
| MICB 广播 | 约 57.50 | 整区强制输出有害 | 建筑级建模本身无用 |

论文当前最稳的机制不是“软融合”，而是：

> **预测建筑区域限定的长距离类别证据共享，比普通局部平滑更有效。**

S2 相对 S1 的差值太小，不能把 soft blending 单独写成主要贡献。

## 3. 当前代码中必须修正的问题

当前 `spi_s2_distillation_v1` 的 `StructuralResidualStudent.forward(...)` 接收：

```text
StructuralBatch.region_context
StructuralBatch.region_distribution
StructuralBatch.region_variance
StructuralBatch.accepted_support
```

这些输入都由 `RegionMaps/LocCC` 计算。因此现有 K3-K5 是“可学习的 LocCC 区域适配器”，并没有真正去掉连通域。若论文声称推理无连通域、端到端或更快，代码会直接反证该主张。

有两种诚实选择：

### 选择 A：保留当前实现

将其定义为：

> Learnable LocCC-conditioned semantic adapter。

此时不能声称去除后处理或提升速度，只能研究“学习的区域残差是否优于固定 S2”。创新度较弱。

### 选择 B：推荐修正

让 LocCC 只存在于训练教师端。学生接口改为：

```python
student(
    fused_feature,
    base_logits,
    building_probability,
    rgb_feature,
    sar_feature,
)
```

推理时不得传入：

```text
region labels
connected components
region mean/distribution
accepted region mask
S2 thresholds
```

训练时 `StructuralBatch` 只用于构造 teacher probability、teacher change mask 和 utility supervision，不能作为学生前向输入。

## 4. 修正后的方法结构

```text
                         训练期
P3 loc ── full-res LocCC ── S2 teacher probability p_T
  │                                 │
  │                         correction utility target
  │                                 │
  └─────────────┐                   │
                ▼                   ▼
P3 soft building probability ── Soft Connectivity Student
P3 fused/RGB/SAR features  ───── Direction-aware Utility Router
P3 base logits             ───── bounded residual
                │
                ▼
          student prediction p_S

                         推理期
P3 features + soft building probability
                ↓
Soft Connectivity + Utility Router
                ↓
        prediction（无 LocCC/S2）
```

### 4.1 光学锚定软连接

在低分辨率特征网格中，对 8 邻域像素建立连续边权：

\[
a_{xy}=
\sqrt{P_B(x)P_B(y)}
\exp\left(-\frac{\|f^R_x-f^R_y\|_2^2}{\tau_R}\right)
\exp\left(-\frac{|P_B(x)-P_B(y)|}{\tau_B}\right).
\]

这里：

- `P_B` 来自 P3 的 RGB 建筑定位；
- `f^R` 是 RGB 定位/融合特征；
- 不使用硬阈值或连通域；
- 强 RGB 边界自动降低信息传播；
- SAR 不决定建筑拓扑，避免散斑切断建筑。

进行 2-3 次有界消息传播：

\[
h_x^{t+1}=h_x^t+eta_t
\sum_{y\in\mathcal N(x)}\bar a_{xy}
\bigl(V(h_y^t)-V(h_x^t)\bigr),
\]

其中 `η_t` 使用有界参数并零初始化或小值初始化。这个学生学习近似“同一建筑区域内的长距离信息共享”，但不显式生成实例。

### 4.2 候选纠错方向

学生输出有界候选残差：

\[
\Delta z_x=\delta_{max}\tanh(H([f_x,h_x,p_0(x)])).
\]

不要只输出一个标量 gate。至少显式表示四种关键方向：

```text
I → D
I → X
D → X
X → D
```

Background 由 P3 建筑概率保护，不作为自由纠错方向。当前主要目标是 `I→D`；其他方向用于避免一个标量门控把不同错误混合。

### 4.3 纠错效用监督

在训练集上，S2 对像素/区域是否有益可以直接从语义标签计算：

\[
u_x=
\operatorname{CE}(p_0(x),y_x)
-
\operatorname{CE}(p_T(x),y_x).
\]

- `u_x>0`：S2 比 P3 更接近真值；
- `u_x<0`：S2 会伤害 P3；
- `|u_x|`：纠错收益或风险强度。

效用 gate：

\[
g_x=\sigma(G(f_x,p_0(x),p_T(x)-p_0(x),d_x)),
\]

其中 `d_x` 是候选修改方向。推理时不能使用 `p_T`，所以实际 student gate 输入应换成自身预测的 region-like context disagreement；训练期可以用 teacher disagreement 蒸馏该表示，但不能直接成为部署输入。

最终：

\[
z'_x=z^0_x+g_x\odot\Delta z_x.
\]

这比历史 Risk Gate 更具体：历史 gate 主要看 entropy、边界和简单跨模态相似度；新 gate 学的是“哪一种修改在当前证据下会降低损失”。

### 4.4 损失

\[
L=
L_{sem}
+\lambda_d L_{distill}
+\lambda_u L_{utility}
+\lambda_h L_{harm}
+\lambda_a L_{anchor}.
\]

其中：

1. 语义损失：正常四类监督；
2. 蒸馏损失：只在 S2 接受区域且 teacher 与 base 有有效差异的位置蒸馏；
3. 效用损失：预测 `u_x>0` 或回归裁剪后的效用值；
4. 伤害惩罚：当学生比冻结 P3 的训练损失更差时惩罚；
5. 锚定损失：教师未修改区域、低建筑概率区域保持 P3。

建议损失形式：

\[
L_{distill}=
\frac{\sum_x w_x^+ KL(p_T(x)\Vert p_S(x))}
{\sum_xw_x^++\epsilon},
\]

\[
w_x^+=M_T(x)\max(u_x,0),
\]

即主要模仿训练标签证明“有益”的教师修改，而不是把 S2 的所有错误也蒸馏进去。

注意：效用标签只能在 Train2918 或其训练折产生；不能用 Val450 生成 gate 标签。

## 5. 下一轮严格实验顺序

### Stage 0：先锁定尚未完成的因果证据

#### E0.1：S2-S1 配对 Bootstrap

目的：判断 Soft S2 是否真的优于 Hard S1。若置信区间跨 0，正文把二者统称为 Selective Region Context，不再强调 soft 优势。

#### E0.2：低分辨率教师一致性

同时报告：

- 全图像素 agreement；
- 前景像素 agreement；
- S2 实际修改像素 agreement；
- `I→D / I→X / D→X / X→D` 分方向 agreement；
- full-S2 与 low-S2 的 mIoU 差。

建议预注册门槛：修改像素 agreement ≥ 95%，low-S2 与 full-S2 的 Valid mIoU 差绝对值 ≤ 0.10 pp。若不通过，训练教师改为离线缓存的 full-S2 soft probability；不要继续调学生。

#### E0.3：训练集纠错效用审计

统计 S2 修改中：

- helpful / harmful / neutral 比例；
- 各修改方向的 precision；
- 各事件 helpful/harmful 比；
- 一个 train-only Oracle gate 的上限。

如果 `I→D` 的 helpful 比率不高于 harmful，方向效用路由没有学习基础，应停止该方向。

### Stage 1：500-step 因果 Pilot

| 变体 | 唯一变化 | 目的 |
|---|---|---|
| K0 | 冻结 P3 | 锁定基线 |
| K1 | 固定 full-S2 | 教师上限 |
| K2 | 无结构输入的参数匹配 Local Student，只做语义训练 | 容量/训练控制 |
| K3 | K2 + S2 蒸馏，推理无 LocCC | 纯教师行为是否可被学习 |
| K4 | K3 + optical-anchored soft connectivity | 软连接是否恢复区域长程行为 |
| K5 | K4 + direction-aware utility router | 是否减少 harmful corrections |
| K6 | K5 + event-balanced/GroupDRO | 跨事件稳定性 |

第一因果比较是 `K3-K2`；结构学生比较是 `K4-K3`；路由价值比较是 `K5-K4`；稳健性比较是 `K6-K5`。

Pilot 只访问 Train2918 内部 event-stratified calibration。建议晋级条件：

- K3 至少优于 K2 `0.20 pp`；
- K4 至少不低于 K3，且修改像素 agreement 更接近 full-S2；
- K5 的 harmed pixels/regions 相对 K4 下降至少 20%，且 mIoU 不下降；
- Damaged F1 相对 P3 至少 `+0.5 pp`；
- 任一前景类下降不超过 `0.5 pp`。

这些是实验前建议锁定的决策门槛，不是已经取得的结果。

### Stage 2：3000-step 与事件互斥验证

只晋级 Pilot 中最好的一个结构版本和一个无结构控制：

1. 3000-step，seed 3407；
2. 通过后再运行 3 seeds；
3. 4-fold event-held-out；
4. 最后可选 leave-one-disaster-type-out；
5. 模型、阈值和 checkpoint 规则锁定后，只访问一次 Test。

主指标：

```text
Valid mIoU
Damaged IoU / F1
Damaged gate-miss
event macro
event worst
14事件非负数量
harmful correction rate
inference latency（含全部后处理）
```

若只在 Val450 提高、event-held-out 不提高，论文只能定位为同事件后处理，不能声称跨灾害稳健。

## 6. 2024-2026 模块应如何放置

| 模块 | 下一步角色 | 是否进入主线 | 使用条件 |
|---|---|---|---|
| MSFA 2024 | SAR 初始化/迁移对照 | 否 | F0/F1/F2 解冻消融证明 SAR 表征是瓶颈后再做 |
| HeteCD 2025 | 高可信 Intact 区的局部一致性正则 | 暂否 | 只能可靠区域对齐，禁止全图 FCA |
| M²CD 2025 | “P3 专家 vs 结构纠错专家”的效用路由思想 | 是，轻量迁移 | 不改 backbone 为大 MoE，只实现纠错 utility gate |
| LPEM/RIEM/CGSL | 模态内关系变化作为 gate 可靠性特征 | 条件进入 | BIOS-E0/SP-E0 在事件互斥审计通过后 |
| IM4CD 2026 | 局部匹配置信度/offset 作为拒绝特征 | 条件进入 | 只能帮助判断 SAR 证据是否可信，不直接当损毁分数 |
| DSTCD 2026 | 注册预训练支线 | 否 | 少样本或配准审计明确有上限时才做 |
| GCLN 2026 | 近远关系/对比约束 | 暂否 | 简单 relation feature 已有效后才增加 |
| BRDG/Laplacian SP | 可学习区域边界 | 停止 | RGID 未证明精细拆分优于 LocCC，当前数据已反对其优先级 |
| BRIGHT Challenge 2026 | 光学主导定位、模态专用编码、区域/query 决策、场景稳健性 | 是 | 作为总体设计原则和强基线来源 |

### 为什么不是把模块全部拼进去

你的历史结果已经说明：局部匹配、超像素、风险 gate、RGID、原型和扩散分别可能有合理解释，但叠加不能自动形成增益。下一篇方法最多应有三项核心贡献：

1. teacher-only pseudo-region supervision；
2. CC-free optical-anchored connectivity student；
3. direction-aware utility routing + event-held-out validation。

MSFA、HeteCD、IM4CD、CGSL 只能是条件消融或后续论文，不能同时进入首版模型。

## 7. 最近工作与差异定位

| 最近工作 | 与本项目相近处 | 本项目必须不同的轴 |
|---|---|---|
| BRIGHT Challenge 2026 outcome | 光学主导定位、RGB/SAR 专用编码、建筑级损毁 | 本项目研究无实例标注的语义模型结构教师与跨事件纠错 |
| HeteCD 2025 | 非共享编码、异构特征一致性与差异挖掘 | 不做全图对齐，重点是 predicted-building context 与 correction utility |
| M²CD 2025 | 专家路由和跨模态自蒸馏 | 路由对象不是 backbone token，而是 P3 与结构纠错候选 |
| CGSL 2025 | 超像素图公共结构与结构差异 | 图关系只作可靠性候选；核心学生学习 LocCC 教师而非无监督二值变化 |
| IM4CD 2026 | 局部模板匹配和响应峰偏移 | 匹配量只作拒绝/置信信号，不直接决定三级损毁 |

仅“把 S2 蒸馏进 CNN”与普通后处理蒸馏过近，创新不足。`teacher-only region supervision + continuous connectivity + counterfactual utility routing` 才形成明确差异。

## 8. Idea Evaluator 结论

### 8.1 第一印象

- 论文类型：Novel Method，建立在一个已测得有效的 New Setting 上。
- 一句话故事：无需实例标注，训练期从语义模型自身定位构造结构教师，推理期以光学锚定软连接学生和效用路由学习建筑内长程纠错，并在未知事件上验证其稳定性。

### 8.2 致命风险审计

| 风险 | 严重性 | 修正 |
|---|---|---|
| 当前 K3-K5 推理仍读取 LocCC，却可能被描述为无连通域蒸馏 | MAJOR | 将 `StructuralBatch` 限制为训练目标，学生 forward 禁止区域输入；否则删除速度/端到端主张 |
| “S2 蒸馏”本身与普通后处理蒸馏差异不足 | MAJOR | 加入连续连接学习与纠错效用路由，并用 K2-K6 因果链分别证明 |

以 RGID 为核心创新的版本已经被“未超过 LocCC”的数据反驳，应 `Reject and Pivot`；不再把 Center/Boundary/Offset 复杂化作为主线。

### 8.3 五维评分（修正版 UOCD）

| 维度 | 分数 | 依据 |
|---|---:|---|
| Higher | 7/10 | S2 已测得 +1.0696 pp，但学生与 utility router 尚未取得数据，且 S2-S1 仅 +0.0317 pp |
| Faster | 8/10 | 机制型评分：若学生推理完全不读取 LocCC，可删除 CPU CC/阈值/区域统计；必须实测端到端延迟 |
| Stronger | 8/10 | 机制型评分：直接以 event-held-out、GroupDRO 和 harmful correction 为目标；尚待 4-fold 验证 |
| Cheaper | 7/10 | 不依赖官方实例标注，结构教师由已有定位自动产生；仍需正常语义标签训练 utility |
| Broader | 6/10 | 可迁移到其他“语义定位 + 区域一致标签”任务，但首篇只应验证 BRIGHT |

最应强调：`Stronger + Faster`，其次是 `Higher`。不要把论文写成单纯追求更高 Val450 mIoU。

### 8.4 范式潜力

- First Principles：是。挑战“要利用实例结构就必须有实例标注或显式实例推理”的假设。
- Elephant in the Room：是。S2 有效但依赖不可学习后处理，跨事件是否稳定又长期未解决。
- Technology Cycle：部分。并非依赖全新硬件，但受益于成熟的多模态编码和知识蒸馏。
- Hamming's Rule：部分。若能稳定解决未知事件 Damaged，价值高；仅提升当前 Val450 则影响有限。

结论：具有中等范式潜力，但必须由跨事件结果兑现。

### 8.5 可行性

| 风险 | 等级 | 缓解 |
|---|---|---|
| 计算 | 中 | 一张 4090 可运行冻结 P3 + 小学生；只让 K4/K5 晋级完整训练 |
| 数据 | 低 | BRIGHT 语义标签和 P3 定位已存在，不需要新增实例标注 |
| 工程 | 中 | 现有 S2、蒸馏、event folds 已实现；主要改动是学生输入契约、soft affinity 和 utility target |
| 时间 | 中 | 建议 8-12 周完成 E0、Pilot、3 seeds、event folds；用户每周有效工时未知，需据实际资源校准 |

### 8.6 最终判定

**Accept with Revisions — 值得推进，但必须先完成 teacher-only 输入隔离和 Stage 0 三项审计。**

## 9. 三个可投稿题目方案

### 方案 A：推荐主线

**Utility-Guided Optical-Anchored Connectivity Distillation for All-Weather Building Damage Mapping**

- 任务：BRIGHT RGB-SAR 语义损毁分割；
- 方法：full-S2 teacher、CC-free soft connectivity、direction-aware utility router；
- 最小实验：K0-K6 + 4-fold event-held-out；
- 预期贡献：无实例标注、无推理 CC、降低有害区域纠错。

### 方案 B：保守论文

**Selective Pseudo-Building Context for RGB-SAR Damage Segmentation without Instance Annotations**

- 任务：冻结 P3 后处理；
- 方法：LocCC + A6/S2，不声称端到端；
- 最小实验：S1/S2/local/shuffle/bootstrap/event breakdown；
- 预期贡献：证明预测建筑结构具有独立于局部平滑的语义价值；
- 风险：创新较增量，但最容易完成。

### 方案 C：跨事件强化版

**Event-Robust Correction Utility Learning for Optical-SAR Building Damage Assessment**

- 任务：未知事件上的 Damaged gate-miss；
- 方法：train-only correction utility、event-balanced/GroupDRO、最坏事件约束；
- 最小实验：P3/S2/UOCD 在 4-fold、LOEO、灾种留一上的对比；
- 预期贡献：把重点从平均 mIoU 转向纠错可靠性与未知灾害稳定性。

### 方案 D：后续扩展，不作为首轮

**Matching-Confidence-Aware Structural Correction for Misregistered Optical-SAR Damage Mapping**

- 任务：局部错位与 SAR 伪证据；
- 方法：IM4CD 式峰值、offset、entropy 只作为 utility gate 的拒绝特征；
- 最小实验：K5 与 K5+matching confidence、±1/±2 位移压力测试；
- 使用条件：IM4CD 匹配信号必须先在 train-held-out 审计中预测 harmful correction。

## 10. 最先执行的三件事

1. 补做 S2-S1 配对 Bootstrap，并完成 full-S2/low-S2 的“修改像素与修改方向”一致性审计。
2. 修正蒸馏学生接口：LocCC/StructuralBatch 只能生成训练目标，禁止成为学生部署输入；建立 K2/K3 精确控制。
3. 在 Train2918 内建立 correction utility 标签和 K4/K5 500-step Pilot；未通过前不加 IM4CD、CGSL、MSFA、Memory 或 RGID。

## 11. 证据来源与边界

- 本地 P3/S1/S2/S4 数值来自 `spi_pseudo_instance_mainline_v1` 的锁定记录。
- 当前本地仓库只有 S2 distillation 的代码与协议，未找到其服务器完整结果，不能假设 K3-K5 已经成功或失败。
- BRIGHT Challenge 2026 outcome 显示实例级任务在未见事件上显著下降，并指出模态专用编码、晚期融合、光学主导定位和场景适应是领先方案共同特征。
- HeteCD、M²CD、CGSL、IM4CD 主要验证异构二值变化；它们不能直接证明三级损毁，迁移时必须降低主张。

