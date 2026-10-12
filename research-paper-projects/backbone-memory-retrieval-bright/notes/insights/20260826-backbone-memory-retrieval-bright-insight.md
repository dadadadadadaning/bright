# 骨干保持式记忆检索对 BRIGHT RGB–SAR 建筑损毁评估的启发

- 用户研究方向：灾前光学 RGB 与灾后 SAR 的建筑级损毁评估
- 用户研究问题：怎样让外部训练记忆补回 P3 漏掉的 Damaged，同时避免大量 Intact 被误报
- 分析日期：2026-08-26
- 论文证据：RAC、Tip-Adapter、REACT、SupCon、SPKD、Balanced Meta-Softmax、Logit Adjustment、SelectiveNet、IM4CD、SRAC
- 本地证据：RM0–RM5-B 的验证结果；Test 均未访问

## 一、先解释 RM5-B 的最终含义

RM5-B 使用 `SupCon + 5 × 点对点特征保持`，训练到了 3000 步，但安全选择仍回到 Step 0。

| Step | Region Macro F1 | Event F1 | D-net | AUROC | Safety |
|---:|---:|---:|---:|---:|---|
| 0 | 73.0272 | 50.3393 | 300 | 0.6449 | 基线 |
| 500 | 71.7175 | 50.3191 | 291 | 0.6515 | 未通过 |
| 1000 | 72.1850 | 51.4266 | 240 | 0.6406 | 通过保持门，但检索可信度下降 |
| 1500 | 70.7396 | 51.5436 | 281 | 0.6695 | 未通过 |
| 3000 | 70.7969 | 52.2482 | 211 | 0.6499 | 未通过 |

结论不是 Memory 无用，而是：

1. 全局 SupCon 仍会重排 P3 邻域；
2. 点对点 L2 保持只能限制坐标漂移，不能同时保留有用邻域并释放跨事件变化；
3. 训练越久，Event F1 可以提高，但安全检索与 D-net 并不随之提高；
4. 当前 RM5-A/B 不能进入融合或 Test。

## 二、别人怎样处理“骨干＋Memory”

| 常见做法 | 代表工作 | 关键设计 | 对当前项目的启发 |
|---|---|---|---|
| 并行检索分支 | RAC | 基础编码器与外部记忆分支并行，最后组合两支 logits | Memory 不应直接替换 P3 |
| 键值 Cache 残差 | Tip-Adapter | 冻结骨干；相似度生成 Cache logits；用系数控制增量 | 需要保留 P3 先验和零增量安全状态 |
| 冻结原模型、只训新增块 | REACT | 检索相关知识，原模型权重全部冻结 | 继续解冻 P3 没有文献必要性 |
| 长尾先验校正 | Balanced Softmax / Logit Adjustment | 把类别频率显式加入 logits 或损失 | Memory 容量均衡不能替代真实类别先验 |
| 关系保持而非坐标保持 | SPKD | 保持样本两两相似矩阵，不要求逐点特征相等 | RM5-B 的 L2 应改为邻域关系保持，且只做一次受控验证 |
| 可拒绝预测 | SelectiveNet | 联合学习分类和 abstain，评估 risk–coverage | Memory 必须允许“不给建议” |
| 局部跨模态匹配 | IM4CD | 用局部匹配响应代替逐像素比较 | 可用于构造检索证据，但你现有实验证明不能直接加权进 key |
| 超像素检索 | SRAC | 以结构区域作为查询单位并引入外部知识 | 超像素只是检索单位，不自动解决先验和可信度问题 |

## 三、当前实现的根本错位

### 错位 1：把 Memory 的存储分布当作真实后验

当前 Memory 每类等容量，并按类别分别取 Top-k。这样做适合保证稀有类在库中有代表，但推理时等价于把三类先验近似设为相同。验证查询实际以 Intact 为主，于是 Damaged 被系统性过预测。

这与数据完全一致：跨事件检索救回 Damaged `+273`，却破坏 Intact `-1857`。

### 错位 2：让 P3 分类特征同时承担检索度量

P3 的区域特征为像素损毁分类训练，不保证跨事件同损毁建筑互为近邻。RAC 使用独立固定检索编码器；REACT 也把检索知识适配与原模型解耦。当前直接复用 P3 key，是一个可用基线，不应被当作最终检索表征。

### 错位 3：检索器必须三选一，没有拒绝项

每一类都强制产生 Top-k，哪怕所有 Damaged 邻居都很远，检索器仍会给出 Damaged 分数。RM4 中 retrieval damaged score 的 AUROC 只有 `0.4987`，说明该分数几乎不知道自己何时可信。

### 错位 4：RM5-B 保持了坐标，而不是邻域关系

`||z_new-z_old||²` 要求新旧特征方向接近；但检索真正关心的是近邻顺序和两两相似度。SPKD 表明可以保持 Gram/相似矩阵，让新空间在不摧毁原邻域的前提下发生整体变换。

## 四、下一步唯一推荐主线

### 总体定义

> 将 Memory 从“第二个硬分类器”改成“带训练先验、跨事件检索键和拒绝能力的条件证据分支”。

```text
冻结 P3 建筑定位与四类预测
              │
              ├── P3 posterior：稳定基础先验
              │
训练集建筑 Memory
→ 独立 Retrieval Key
→ 排除同事件 / 邻居事件多样化
→ 类别先验校正
→ 可靠则输出证据，不可靠则 abstain
              │
              └── 只校准 P3 的条件 I/D/X，不修改建筑定位
```

### RM6-A：先验校正审计，不训练

先不改 embedding。保留类别×事件均衡 Memory，但从未均衡的训练候选区域统计真实训练先验 `π_c`。将检索类别证据写成：

`s'_c = s_c + τ log π_c`

其中 `τ` 只能在训练事件内部的 leave-one-event-out 中确定，验证集只评估一次。该实验只回答：1863 个 Intact 误伤有多少来自错误先验。

准入条件：

- Intact harm 至少下降 50%；
- Damaged D-net 仍大于 0；
- overall net correction 大于 0；
- 非负事件至少 10/14。

若 RM6-A 仍无法得到 overall net > 0，则停止“纯检索分类器”，不再训练 RM5-C。

### RM6-B：独立检索键，仅在 RM6-A 显示先验问题可修时运行

用单独小投影器读取冻结 P3 区域特征和 SAR 场景相对统计，但不改变 P3：

```text
P3 region feature 256
→ LN → Linear 256→128 → GELU → Linear 128→64
SAR relative mean/std
→ Linear 2→16
concat → Linear 80→64 → L2 normalize
```

训练目标：

`L = L_cross_event_supcon + λ_rel ||G_new-G_p3||² + λ_ce L_prior_aware_CE`

这里 `G` 是批次两两余弦相似矩阵，不再使用 RM5-B 的逐点 L2。它回答一个明确问题：能否学习跨事件损毁距离，同时保留 P3 邻域关系与真实类别先验。

### RM6-C：Train-only 拒绝校准器，仅在 RM6-B 通过后运行

输入只使用：

- P3 三类条件概率、熵和 margin；
- Memory 三类先验校正证据；
- Top-k 标签一致率；
- Top1–Top2 距离差；
- 邻居事件数量；
- SAR 场景相对统计。

输出不是 I/D/X，而是 `accept_memory ∈ [0,1]`。所有训练和阈值选择都在训练事件内部完成；验证集只做一次外层评估。

### 最后才允许融合

只有 RM6-C 存在正的 risk–coverage 点，才使用：

`z_final = z_P3 + α · accept_memory · e_memory`

其中 `α` 初始化为 0，P3 的 Background/Building 概率保持不变。

## 五、想法评估

### 1. First impression

- Paper type：Cross-domain Technique Paper
- One-sentence story：在光学锚定的建筑损毁骨干上，将外部 Memory 建模为经过类别先验和跨事件可靠性校准的可拒绝 SAR 损毁证据，而非第二个硬分类器。

### 2. Fatal-flaws audit

| # | Flaw | Severity | Defense |
|---|---|---|---|
| 1 | 若只声称“超像素/建筑区域＋Memory”，会与 RAC、Tip-Adapter 和任务最接近的 SRAC 高度重叠 | MAJOR | 创新必须限定为跨事件、先验校正、risk-controlled abstention，并用 SRAC/RAC 风格基线对照 |
| 2 | 当前直接 kNN、RM5-A 与 RM5-B 已被数据否定，若继续把它们作为核心机制会形成数据反驳 | MAJOR | 正式终止硬替换和全局 metric rewrite，只允许 RM6-A 这一项统计修正决定是否继续 |

### 3. Lifecycle and capability match

| Aspect | User input | Assessment |
|---|---|---|
| Idea category | 跨域创新技术 | 以检索增强和选择性预测迁移到 RGB–SAR BDA |
| Lifecycle | 预计 2–3 个月形成完整受控实验 | 取决于 RM6-A 是否通过；失败应立即收敛论文 |
| Weekly effective hours | 未提供 | 暂定 Yellow，需要用户按实际投入校正 |
| Fit | 已有 5090、完整 P3/RM 缓存和审计代码 | Green/Yellow |

### 4. Five-dimension radar

| Dimension | Score | Evidence | Lift suggestion |
|---|---:|---|---|
| Higher | 6 | 已测得 Damaged D-net 为正，但总体净纠正为负 | RM6-A 必须先实现 overall net > 0 |
| Faster | 7 | 区域 Memory 查询比逐像素检索更可控；尚无速度实测 | 报告区域数、查询耗时和 Memory 占用 |
| Stronger | 8 | 机制直接针对跨事件偏移和拒绝风险；尚未由最终指标验证 | 采用 train-event LOEO 与 risk–coverage 曲线 |
| Cheaper | 7 | 只用已有训练标签和冻结骨干，不增加标注 | 明确 train-only Memory 可更新机制 |
| Broader | 8 | 将 RAC/Tip-Adapter/SelectiveNet 的思想迁移到异构遥感 BDA | 用 RGB–SAR 模态非对称性与事件偏移形成领域特定机制 |

Verdict：**Accept with Revisions，worth pursuing pending RM6-A**。旧的“硬 kNN 或 RM5 全局改写”版本应 Reject and Pivot；修订后的“先验校正＋可拒绝证据”值得一个严格的最小实验。

## 六、论文逻辑骨架

### Paper-type positioning

- Type：Cross-domain Technique Paper
- Rationale：任务已有明确基准，贡献来自把检索增强、长尾校准与选择性预测迁移并改造成适合 RGB–SAR BDA 的结构。

### Thinking template

| Stage | Your content |
|---|---|
| Research background | 全天候建筑损毁评估需要灾前光学与灾后 SAR；BRIGHT 暴露了弱 Damaged、异构配准和跨事件偏移问题。 |
| Limitation 1 | 像素骨干把稀有损毁知识隐式压入参数，弱 Damaged 容易被多数 Intact 吞没。 |
| Limitation 2 | 现有类别均衡检索把存储均衡错误传播为预测先验，导致大量 Intact→Damaged。 |
| Limitation 3 | 相似度同时编码事件/传感器条件，且没有可靠拒绝机制。 |
| Key Idea | 把训练建筑 Memory 作为经过类别先验与跨事件可靠性校准的条件证据，只在可信时修正光学锚定 P3。 |
| Challenge 1 | 如何构造既保留 P3 语义又适合跨事件检索的建筑键。 |
| Challenge 2 | 如何分离损毁相似度、类别先验和事件成像捷径。 |
| Challenge 3 | 如何保证 Memory 只救援而不大规模误伤 Intact。 |
| Methodology topic sentence | 提出光学锚定、先验校正且可拒绝的跨事件建筑记忆增强框架。 |
| Module A | 关系保持的独立跨事件检索键，解决 Challenge 1。 |
| Module B | 类别先验校正与事件多样化检索，解决 Challenge 2。 |
| Module C | Train-only 风险校准和 abstention，解决 Challenge 3。 |
| Contribution 1 | 提出面向异构 BDA 的条件 Memory 角色定义与框架（Section 3）。 |
| Contribution 2 | 提出关系保持、先验校正的跨事件检索机制（Section 3.2–3.3）。 |
| Contribution 3 | 建立 rescue/harm、risk–coverage 和事件留出的安全评估协议（Section 4）。 |

### Self-consistency checks

- Limitations → Key Idea：Pass
- Key Idea → Challenges：Pass
- Challenges → Methodology：Pass
- Methodology → Contributions：Pass

## 七、建议题目

1. **Risk-Calibrated Cross-Event Memory for Optical–SAR Building Damage Assessment**
2. **Optical-Anchored Selective Memory Augmentation for All-Weather Building Damage Mapping**
3. **Prior-Aware Retrieval of Rare Damage Evidence under Cross-Disaster Domain Shift**

## 八、最终执行决定

现在只做 RM6-A，不写 Cross-Attention，不继续扫 RM5-B 的 preserve weight，不访问 Test。RM6-A 是对当前最大统计错位的修正；若它仍无法让 overall net correction 转正，Memory 可以保留为论文诊断贡献，但不再作为最终提分模块。

## 证据链接

- RAC：https://openaccess.thecvf.com/content/CVPR2022/html/Long_Retrieval_Augmented_Classification_for_Long-Tail_Visual_Recognition_CVPR_2022_paper.html
- Tip-Adapter：https://arxiv.org/abs/2207.09519
- REACT：https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Learning_Customized_Visual_Models_With_Retrieval-Augmented_Knowledge_CVPR_2023_paper.html
- SupCon：https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html
- SPKD：https://arxiv.org/abs/1907.09682
- Balanced Meta-Softmax：https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html
- Logit Adjustment：https://openreview.net/forum?id=37nvvqkCo5
- SelectiveNet：https://proceedings.mlr.press/v97/geifman19a.html
- IM4CD：https://doi.org/10.1016/j.isprsjprs.2026.02.004
- SRAC：https://jianghai.ucas.ac.cn/index.php/zh/xshde/xsdte/461-igarss-2026

