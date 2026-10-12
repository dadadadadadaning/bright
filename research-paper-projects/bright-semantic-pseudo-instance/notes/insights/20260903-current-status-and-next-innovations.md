# BRIGHT 伪建筑结构主线：当前状态、创新去重与下一步

日期：2026-09-03

## 1. 当前正在做什么

当前工作的准确名称应是：

> **Semantic-only pseudo-building structural refinement for RGB–SAR building damage mapping**：不使用官方实例标注训练主方法，从 P3 建筑定位形成伪建筑区域或 learned RGID 分区，以锁定的区域决策反向修正语义损毁图，并用真实实例仅作诊断评价。

它不是普通实例分割，也不是 Memory 检索，更不是把真实实例替换成伪实例。主问题是：免费结构是否能超过 P3 的 `60.5020%` 语义基线。

## 2. 证据链已经走到哪里

### 已成立

- P3 在 Standard-ML Val450 上严格复现 `60.5020%`。
- LocCC+Majority 为 `61.3790%`，Mean Probability 为 `61.2243%`。
- 锁定 A6 后，LocCC 为 `61.5399%`，相对 P3 `+1.0379 pp`。
- 匹配的 5×5 局部窗口约为 `60.4987%`；C2−C1 的事件分层 bootstrap 95% CI 为正，说明收益不能由普通局部平滑解释。
- Stage-0 GT geometry pooling 上限为 `62.74%`（另一 697 张公共实例验证协议），说明正确建筑归属有价值。

### 已否定或暂停

- T1 组件均衡训练：Val450 `59.7052% → 59.5607%`，不能证明伪组件训练提高 raw semantic head。
- MICB 区域广播：`57.5036%`，显著有害；它降低 D→I/X→I 的同时把错误转移到其他类别或背景。
- CMR/Memory：没有安全 operating point，事件因子化也不足；不能并入当前主方法。
- 面积上限和边界腐蚀：前序门控实验没有带来稳定收益，A6 已固定关闭它们。

### 唯一缺失的主因果实验

在完全相同 A6 下比较：

\[
\Delta_{RGID}=mIoU(\mathrm{RGID\mbox{-}G3+A6})-mIoU(\mathrm{LocCC+A6}).
\]

只有该值稳定为正，才能把论文从“建筑区域后处理”升级为“learned pseudo-instance geometry improves semantic damage mapping”。

## 3. 与外部工作相比，哪些不再算创新

- center+offset：Panoptic-DeepLab 已系统提出，本地 RGID 只是任务迁移。
- center+boundary+watershed：建筑接触边界和 watershed 已有成熟先例。
- Mask2Former/query segmentation：通用实例框架和 BRIGHT 冠军均已采用。
- 光学/SAR 双编码：BRIGHT 冠亚军和本仓库 P4/P5/H6 已覆盖。
- ordinal loss：建筑损毁与本仓库 OA-HMD 已覆盖。
- pseudo-label/self-training：DFC 2025 冠军与 BRIGHT 2026 冠军已覆盖。
- prototype/Memory/local matching：本仓库 BGMP、SIRM、CMR、SR2、IARM、CCU 等已大量验证。

所以论文不能把“我也用了这些模块”写成贡献。

## 4. 仍有价值的三个创新候选

### A. 可靠性门控的按需实例拆分（首选主创新）

观察：LocCC 在 8/14 个事件为正，但 Turkey 等密集场景下降；RGID 的 Oracle Center 又显示大量 Merge/Split 可修复。固定对所有 CC 拆分并不合理。

对 LocCC 区域 \(R_j\) 构造拆分风险：

\[
u_j=f(A_j,e_j,h_j,\bar H_j,n_j,b_j),
\]

其中面积 \(A_j\)、伸长度 \(e_j\)、孔洞数 \(h_j\)、平均语义熵 \(\bar H_j\)、稳定中心峰数 \(n_j\) 和边界支持 \(b_j\) 均可由训练期伪结构或预测得到。仅当 \(u_j>\tau_s\) 且存在至少两个稳定中心时运行 RGID；其余区域保持 LocCC。

最终分区为：

\[
\Pi(R_j)=
\begin{cases}
\mathrm{RGID}(R_j),&u_j>\tau_s,\\
\{R_j\},&\text{otherwise}.
\end{cases}
\]

创新点不在“watershed”，而在**只对可信粘连建筑拆分，避免把可靠单体建筑越切越坏**。它直接针对 Turkey 负增益和 Oracle Center 的 Merge/Split 证据。

### B. 变换等变的中心—边界自监督（首选 RGID 优化）

当前 Center 的漏检、重复和偏移是最大瓶颈；仅用静态 pseudo target 会继承同类粘连错误。对两种几何增强 \(T_1,T_2\)，要求预测回到同一坐标后保持一致：

\[
L_{eq}^{center}=\|T_1^{-1}H_c(T_1x)-T_2^{-1}H_c(T_2x)\|_1,
\]

\[
L_{eq}^{boundary}=\|T_1^{-1}H_b(T_1x)-T_2^{-1}H_b(T_2x)\|_1.
\]

中心峰只在多视图位置一致时保留。该项不需要真实实例，直接抑制偶然纹理峰与 SAR 散射峰，比继续降低 NMS 距离更符合现有诊断。

### C. 光学几何锚定的建筑 Query 严重度头（备选第二阶段）

若 C3 证明实例分区有效，可将每个建筑子区转为 query，在 RGB 中聚合几何，在 SAR/差分中聚合损毁证据：

\[
q_j^{geo}=Pool_{R_j}(F_{rgb}),\qquad
q_j^{dam}=Pool_{R_j}([F_{sar},|F_{sar}-F_{rgb}|]).
\]

使用小尺度残差更新损毁 logits：

\[
z_j=z_j^{P3}+\alpha_j g(q_j^{geo},q_j^{dam}),\quad \alpha_j\in[0,1].
\]

这与优胜方案的 optical-dominant late fusion 一致，但本项目必须强调 pseudo-only geometry、P3 safe residual 和事件留出评价。它不是现在就做的实验，因为它会同时改变几何和分类器。

## 5. 哪些只是增强，不应作为主创新

- 512 或双尺度 Center head：解决约 3.96% 小 parent 在低分辨率消失，属于必要消融。
- train-only 温度/ordinal bias：用于事件校准，仓库已有相近历史线，只作补充。
- EMA teacher 与高置信 pseudo refinement：可提高稳定性，但已有大量先例。
- Mask2Former 外部基线：必须比较，但不能称为自己的创新。
- 标注比例曲线：是论文价值证明，不是网络模块创新。

## 6. 论文能否开始写

可以写问题、相关工作、协议和已完成结果，但暂时不能写“RGID improves semantics”这一核心结论。当前可严谨声称：

> P3 定位产生的完整建筑连通结构，在不使用真实实例和不训练新分类器的条件下，为语义损毁预测带来约 `+1.04 pp` mIoU，且显著优于匹配局部平滑；收益具有明显事件异质性。

最终题目可围绕：

> **Reliability-Gated Pseudo-Building Decomposition for Semantic RGB–SAR Damage Mapping**

## 7. 最小后续实验矩阵

1. C3：RGID-G3+A6 vs LocCC+A6，完成几何因果闭环。
2. 若 C3 正：Center baseline vs `+equivariance`，再比较 always-split vs split-on-demand。
3. 三 seed：3407、一个新 seed、第三个新 seed；只在 C3 单 seed 通过后运行。
4. 事件留出：至少 4-fold group split；报告 event mean/worst。
5. 真实实例仅评价：segm AP、Merge、Split、matched IoU、same-class-touching 子集。
6. 外部基线：DFC 2025 Building-Guided、官方 Mask R-CNN/Mask2Former 类实例基线；clean 与 transductive 协议分列。
7. 标注预算：pseudo-only、1%、5%、10%、100% true instance，用于证明 annotation efficiency。

若 C3 不超过 C2，则停止 RGID 语义主张，保留 LocCC+A6 作为结构化语义校正方法，并把实例分割作为独立支线。
