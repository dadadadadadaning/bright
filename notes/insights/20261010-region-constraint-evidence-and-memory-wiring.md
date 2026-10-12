# 当前 Q 的正向证据与 Memory 的具体连接方案

日期：2026-10-10。本文依据用户新提供的五路径结果，以及当前本地分区、诊断、建库、J123 与 hardquery 代码。未取得该表所在服务器完整报告，未运行新验证或训练。数值按用户表格计算；不混用另一轮 legacy 协议的 A1 成绩。

**更新判断：现有 Q 在保留原 A1 特征与分类头的情况下，已经表现出正向作用。应优先保留 Q，检查实际区域表示、分类头及 Memory 接法为什么没有保留这部分收益。仅根据原生 region 成绩，先把问题归因于超像素不好，不够准确。**

## 1. 五条路径的实质差异

定义：F 是 A1 的 256 维像素特征；H 是原 A1 的 256→4 affine 1×1 分类头；S 是 softmax；P 是按 Q 的归一化区域均值汇总；U 是同一 Q 的区域到像素回投；R 是统一的双线性概率插值及类别和归一化。设全局 Q 的行对应像素，列对应区域，D 是各区域质量，则 P=D⁻¹Qᵀ、U=Q。空区域按实现约定处理，正质量区域精确除以其质量。

| 路径 | 计算结构 | 前景 mIoU % | 相对 pixel_grid，百分点 |
|---|---|---:|---:|
| pixel_grid | R[S(HF)] | 61.4194 | 0 |
| feature_mean | R[S(HUPF)] | 61.5373 | +0.1179 |
| region_mean | R[US(HPF)] | 61.7127 | +0.2933 |
| grid_probq | R[UPS(HF)] | 61.8381 | +0.4187 |
| region | R[US(H_new T)] | 61.4220 | +0.0026 |

四条诊断路径取自同一次前向的 F、Q、原 H，并明确禁止 F 与 Q 网格不一致时静默插值。[诊断调用](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/feature_audit.py:195)；[四路径实现](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_stage_diagnostics_v1/feature_core.py:65)。当前表的具体运行完整性仍以原 JSON 为准。

H 是线性变换加偏置，在归一化汇总下，H(PF) 与 P(HF)、H(UPF) 与 UP(HF) 在数学上可交换；代码检查相应数值误差。**softmax 不能这样交换。**

- feature_mean：先混合区域特征并回到像素，再将分类分数转为概率。
- region_mean：区域分类分数先转为概率，再混合回到像素。
- grid_probq：每个像素先独立形成概率，再按区域平均并回投。

region_mean 没有训练新的分类头，仍使用原 H；它优于 feature_mean 的 0.1754 个百分点不能解释为“新区域头训练得更好”。grid_probq 再增加 0.1254 个百分点，支持本轮先形成像素概率再作区域共享的方式，但不是任意数据集上的数学优越性定理。

UP 是由 Q 决定的加权平滑算子。这张表支持当前 Q 所定义的区域关系能够改善已有判断；还未证明它优于规则网格、SLIC 或其他分区，也不直接证明每个类别都改善。需要查看四类和逐事件结果，尤其是 Damaged 与 Intact。

原生 A1 是先插值 logits、再 softmax；pixel_grid 是先 softmax、再插值概率。二者不能混称。旧诊断入口使用严格数值设置，正式 E0/E1 使用 legacy，因此这张表与旧正式成绩不能未经核对直接相减。

## 2. “区域约束”在代码里有三个不同位置

1. **生成 Q 时的空间和建筑先验**：鼓励相近位置、相容建筑概率归到共同区域。
2. **实际区域表示中的结构注入**：把建筑概率、软边界、不确定性等加入特征。
3. **Q 汇总与回投产生的信息共享**：把同一区域的信息用于互相修正。

本表直接支持第三项在现有 Q 上有用。若要归因某一个先验或结构注入模块，还需要对应消融；不能由“按 Q 平滑有效”推出所有结构模块均有效。

## 3. 当前超像素的完整设计

### 3.1 输入与区域数量

冻结 A1 提供 RGB 多尺度特征、融合特征 F 和预测建筑概率 B。RGB 多尺度特征经轻量 UNet++ 风格解码器形成默认 64 维结构特征。Q 的生成主要使用这份结构特征，并结合 B；它不是按真实损毁标签直接切割。

分区在特征网格上铺设规则种子，每个位置仅考虑周围 3×3 个种子。默认 stride=8、迭代 3 次、temperature=0.2、spatial_weight=0.1、building_weight=0.5；实际值应读对应快照 config。stride 是特征网格单位。如果网格是 256×256，则默认约 32×32 个种子，不代表原图每 8 像素一个区域，也不代表每个最终区域都是方格。

[分区器](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_transformer_v1/model.py:74)；[前端连接](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_transformer_v1/model.py:401)。

### 3.2 每个位置怎样选择区域

分配分数可概括为：

\[
a_{ir}=\frac{\gamma\,\mathrm{cos}(q_i,k_r)}{\tau}
-\lambda_s\lVert x_i-\bar x_r\rVert^2
+\lambda_b\log[B_i\bar B_r+(1-B_i)(1-\bar B_r)].
\]

其中 gamma 是学习到的相似度缩放；q_i、k_r 是查询／中心的投影特征。候选有效性、非空质量和数值截断按实现处理。在九个局部候选间 softmax 得到 Q，然后按 Q 更新特征、位置与建筑概率中心，再迭代。

第一项看内容是否相似，第二项避免区域无限扩散，第三项鼓励建筑与建筑、背景与背景相容。B 是软预测且被 detach；这不是建筑／背景硬拆组，也不是建筑实例分割。当前不保证区域连通或同一栋建筑只对应一个区域。

最终存储稀疏 Q[B,9,h×w] 与候选全局 ID，无需显式建立像素数×全部区域数的大矩阵。画 argmax(Q) 得到硬分区图只是可视化；实际汇总仍使用软权重。

### 3.3 实际 region 的额外路径

实际区域表示 T 不等于 PF。其结构为：

```text
原 F256
  → 128 维内容映射
  → 注入建筑概率／Q 软边界／Q 熵的结构残差
  → Q × 空间注意力汇总
  → 主投影
  + 原 F 的 Q 均值／标准差旁路
  + 区域面积、中心、空间范围的几何编码
  → LayerNorm → T128
  → 新训练的区域分类头 → 区域概率 → Q 回投
```

[StructureTokenAdapter](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_transformer_v1/model.py:165)；[SuperpixelEmbedding](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_transformer_v1/model.py:271)。主实验默认不启用额外区域自注意力，应以快照配置为准。

region 相比 region_mean 低 0.2907 个百分点，相比 grid_probq 低 0.4161 个百分点。值得检查的是这整段表示／分类路径，包括训练适配，而不是已经有正向证据的 Q 单独失效。原生汇总还有注意力与质量分母截断，诊断 PF 则使用精确正质量均值，不能把所有差异只归因于 128 维压缩。

最小定位办法：固定 Q 和 F，比较 PF 与 T 的匹配小分类头；结合已有 uniform_attention、no_structure、no_stats、no_geometry 推理干预，确定应单独重训哪一项。推理时删除模块会造成分布改变，不能把敏感性直接当作训练后性能收益。

## 4. 为什么区域约束有效，Memory 仍可能失败

图内区域共享利用的是：邻近、同结构的像素应相互参考。跨图 Memory 利用的是：不同事件的区域特征在检索空间中接近时，其损毁语义也应该相容。这是两种不同关系。

分类头能利用 F 中某些方向区分标签，并不保证整体余弦距离主要由这些方向决定；背景、材质、事件外观或几何同样可能影响邻近关系。类似地，T 配合当前新头表现较差，也不能证明 T 对检索一定差，可能只是新头或训练不合适。

当前标签按 Q 的加权标注组成计算；T 的主路又包含空间注意力，二者代表范围可能不同。这一差异是待诊断的机制，不是本表已经证明的失败原因。将 PF 作为检索 key 的候选，可以让特征和标签的汇总范围更容易对应，但不能保证跨事件检索立即改善。

因此，“当前 Q 的约束有效”与“当前 T→近邻→标签证据尚不够可靠”可以同时成立。

## 5. Memory 现在实际接在哪里

### 5.1 库中存什么

当前 bank 取的是 `result['tokens']`，即 T128；不是诊断里的 PF256，也不是 grid_probq 的四类概率。另存 Train 真实软标签、事件、图像、区域 ID、几何与灾种。[建库实现](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_transformer_v1/build_memory.py:93)。

默认按事件×主类随机优先级选样，每桶至多 64 条，默认 min_purity=0，实际设置以 manifest 为准。推理查询不能使用 GT 类别或 GT 纯度选择支持。旧 metric 的高纯度训练筛选与 bank 成员准入不是同一个步骤。

### 5.2 旧 J1/J2/J3 接的是特征上下文

```text
当前区域 T128 → Reader → 增强区域特征 → 同 Q 回投 → context128
                                                           ↓
原像素 F256 → 可训练像素头 ─────────────── 与 context 分类结果相加
                                                           ↓
                                             插值 logits → softmax
```

- J1：Reader 读取学习到的无历史支持槽。
- J2：按 T 余弦检索 Train Top-8，读取支持。
- J3：按投影 T 的余弦检索同一个库，读取支持；投影主要改变选中 ID，Reader 拿到的仍是原 T 载荷及标签等信息。

Reader 的值包含支持特征、软标签嵌入和事件／灾种／几何嵌入，输出增强特征。缺失支持被学习槽补齐；并没有因为支持不可靠就严格回到一个固定概率基线。[Reader](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_joint_context_v1/model.py:94)。

像素头在第 0 步复制 A1，context 头零初始化，此后像素头和 Reader 一起训练。因此第 0 步恒等不等于训练后保持原 A1。J2、J3 的解码结构一致，但各自训练的权重不一定一致，不能把两者最终差异解释成“只在同一 Reader 上换一次近邻”的推理干预。

**这条路径没有把 grid_probq 的概率结果作为不可变基线。** 所以不能说“先获得 61.8381%，再加 Memory 得到旧 J3”；实际上是另外一套预测路径。[JointContextDecoder](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_joint_context_v1/model.py:215)。

### 5.3 新 hardquery 接的是固定 J1 的 logit 修正

新版本已经有：固定 J1、标签证据及质量统计、有界修正、门控、缺失支持拒绝。形式为 `J1_low_logits + U(g×delta)`，再插值 logits 和 softmax。[修正头](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_hardquery_memory_v1/reader.py:165)；[接入处](/D:/software/dowmload/BRIGHT-master/bda_benchmark/experiments/damageformer_srac_hardquery_memory_v1/runtime.py:37)。

这是比旧联合头更明确的回退设计，但基线仍是 J1，key 仍来自 T128。不能认为它已经自动继承本表 grid_probq 的收益；当前五路径表也没有测试新 hardquery 训练后的效果。

## 6. 建议新增的连接：保留 grid_probq，Memory 修正区域概率

此方案是新增匹配实验，保留原 J1 正式对照及其已有实验，不覆盖原产物。先在统一的正式数值协议确认 grid_probq 的四类与逐事件表现，再冻结为新分支基线。

### 6.1 保留已经有效的路径

\[
p_i=S(HF_i),\qquad \pi_r=(Pp)_r,\qquad B=R[U\pi].
\]

p 是原分类头的像素概率；π 是区域内平均的像素判断；B 正是 grid_probq。这三者都是模型预测，不是 GT 标签。

### 6.2 历史证据与原判断分别保存

用当前 T，或者作为新对照的 PF，生成查询 key。排除同事件、同图像后检索 Top-8。取回 Train 软标签 y_j，并形成标签证据 e_r，同时保留绝对相似度、标签冲突、支持质量及来源多样性。

Reader 输入当前区域表示、π、e 和质量信息，学习输出候选四类分布 ρ 与门控 g。ρ 应是综合原判断与参考后得到的候选，不能默认等于裸标签投票 e。

### 6.3 接在同 Q 回投之前

\[
\pi^{new}_r=(1-g_r)\pi_r+g_r\rho_r,
\qquad p_{new}=R[U\pi^{new}],\quad 0\leq g_r\leq1.
\]

这样保留有效的“像素判断→区域概率共享”路径。g=0 保留原区域判断，g>0 只接受相应程度的修正；全部拒绝时应回到 B。ρ 非负且归一化，门控位于 [0,1]，区域概率仍是合法分布。

若以 logit 偏移构造候选，可采用 ρ∝π×exp(有界偏移) 的形式；这只是一个可测试参数化，不是要求照搬旧 J1 修正头的权重。零修正初始化、自动门控、强制关闭的数值恒等性需分别验证。

### 6.4 必须避免的接线错误

1. **不重复平滑**：π 直接从原 p 汇总取得。把已经 Q 平滑并放大的 B 再降采样、再次 Q 汇总，会增加平滑；软分配的 UP 一般不是幂等算子。不能把 R_Q² 当成 R_Q。
2. **不改变输出顺序**：沿用 grid_probq 的概率回投、概率插值、归一化。旧 `upsample_probability` 实际接收 logits 后做 softmax，不能直接拿来处理新概率路径。
3. **不对有符号差值做概率归一化**：ρ−π 可能含负值且类别和为 0，不能直接交给 `resize_probability`。优先先形成合法 πnew，再走相同输出链。全局 force_off 可直接返回缓存 B。
4. **不把支持资格变成基线资格**：原库的质量筛选可以拒绝某条支持，但不能删掉基线中 0<mass<1e-6 的区域，改变原诊断的精确正质量均值。
5. **拒绝语义明确**：空区域没有回投贡献；无支持时 g=0。单一区域 g=0 不保证其覆盖像素完全不变，因为 Q 有重叠；全局修正关闭才保证整幅图回到基线。sigmoid 的软门控本身也不等于学会精确硬拒绝。
6. **重新训练匹配新基线**：旧 Reader 的困难查询、置信度和监督都基于 J1，迁移到 B 后需重新生成相应 Train 查询状态并训练；不能只替换最终输出变量。

## 7. 针对这组结果，怎样建立新 key 对照

不要同时重选 bank、改变 Q 和训练新 Reader。第一轮保留同一批 `(event,image,region)` 成员、同一软标签和 Top-8，另外保存这些区域的 PF256。

| 方案 | 查找使用什么 | Reader 载荷 |
|---|---|---|
| 原键对照 | T128 的余弦 | 原 T128、标签、质量 |
| 均值键对照 | PF256 的余弦 | 保持原载荷不变 |

这能回答“哪种键表示更适合找参考”，而不是同时改变查询表示与 Reader 表达能力。区域均值分类较好，是提出这个对照的理由，不是检索一定更好的证明。

当前逐图缓存主要保存 T128，没有完整 PF256；需用同一冻结前端重新提取对应区域，不能从 T128 无损恢复 PF256。Q 不改，因此保持成员 ID、标签与配额有明确对应关系。数据处理、量化和协议要记录，避免把 FP16／FP32差异混入表示比较。

随后如训练投影，两方案需要相同查询采样、监督、排除规则和预算，报告输入维度与参数量。旧 encode_keys 及 select_ids 有维度契约，256→128 不能直接插进旧实现。需要独立新接口，保留旧哈希产物；也不能把不同参数量的投影对照宣称为严格只改变距离函数。

支持标签仍来自 Train 的 Q 加权 GT 分布。可同时保存模型预测 π，供 Reader 判断历史模型状态，但 π 与真实支持标签不能混为一列，更不能用 Val GT 筛选查询或支持。

## 8. 建议实验顺序

| 阶段 | 内容 | 回答的问题 |
|---|---|---|
| P0 | 同协议复现五条路径，补四类与逐事件 | grid_probq 是否值得作为正式新增基线 |
| P1 | 固定 Q/F，定位 T 与新区域头 | 实际区域输出的损失来自哪里 |
| M0 | 固定 B，不加修正 | 保留区域约束本身的收益 |
| M1 | 同结构 query-only 修正头，去除历史证据 | 多一个可训练头本身有多少收益 |
| M2 | 同 B、同容量头，加入原库与原键证据 | 历史证据是否超过无历史信息的适配 |
| M3 | 同成员与载荷，换 PF 均值检索 key | 键表示变化能否改善跨事件取证 |
| M4 | 在有效方案上训练困难与风险查询、比较接受机制 | 是否降低 Intact 误改同时保留 Damaged 救回 |

有无门控的对照应在相同证据和相同基线下训练；不要拿新基线上的 gated 与旧 J1 上的 ungated 比。无 Memory 头的支持可用性控制保持一致，去除历史标签、相似度和质量内容，防止增加参数产生的收益被归因于检索。

保留已有六组 hardquery 对照作为原 J1 协议的实验，不自动替换。新增 B 分支用新的清单、缓存和输出目录。首先只读检查 P0/P1 所需已有报告，不需要仅因本次解释就重跑所有训练。

关注指标：四类 IoU、前景 mIoU、逐事件表现、相对共同基线的救回与伤害、真实 Intact 新增 Damaged 假阳性、原本正确 Intact→Damaged 的精确转移、接受率、支持质量分布、训练与检索成本。单个 mIoU 均值上升不能代替这些检查。

## 9. 结论的证据等级

- **当前数据支持**：按现有 Q 汇总原 F／原概率有效；概率先形成再汇总在这五条路径中表现最好；实际 region 未保留这些收益。
- **代码确认**：现有 bank 使用 T128；旧 J123 使用特征上下文融合；新版 hardquery 使用固定 J1 的 logit 修正；二者都不是 grid_probq 的直接扩展。
- **值得验证**：PF 检索 key 是否更可靠；以 grid_probq 为固定基线的选择性概率修正是否带来独立 Memory 收益。
- **尚不能确定**：哪个结构模块导致 region 降分、Damaged 是否同样受益、当前 Q 是否优于其他分区、Memory 是否稳定超过强无 Memory 基线。

目前最有依据的方向是：**保留已经有收益的 Q 和原像素判断链，单独验证区域表示，再让 Memory 在同一强基线上提供可拒绝的增量。**
