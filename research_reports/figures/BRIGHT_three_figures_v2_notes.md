# 三张方法图：设计定义与图注

这是一套推理检索版候选模型设计图，不是已核验代码的完整实现图，也不代表新颖性或性能已经成立。未更改任何训练或模型代码。使用figure-designer技能，输出矢量SVG与PNG预览；延续白底、细线、低饱和蓝橙及Arial英文排版。

## 本次为画图明确的候选约定

1. 区域模块采用预测建筑约束的软关联，不混用固定SLIC。P生成预测分组；R和局部中心生成Q。预测分组可能黏连或漏检，不等同真实实例。
2. F为N×C，Q为N×M；有效建筑像素的关联行归一化，建筑外行可为零。T=D⁻¹QᵀF，其中D=diag(Qᵀ1)；实际实现需屏蔽空区域或加数值稳定项。
3. 本图为符号一致先采用区域维度C。G(T)输出增量，T_A=T+G(T)。若改用内部维度d，必须增加输入/输出投影并同步更改尺寸标注。
4. 区域Attention采用同预测建筑掩码作为候选配置；图未宣称固定近邻稀疏拓扑已经实现。F_A=F+QW_local(ΔT_local)。
5. 建筑Key由T_A按建筑汇总、投影并归一化得到。支持V_j应由相同版本的特征流程生成。所有影响Key和Value的参数发生变化时，都需明确缓存重建或同步策略。
6. 每栋支持分别参与共享参数Cross-Attention。这里“shared attention weights”指共享可学习投影参数，不是各栋共享同一注意力概率矩阵。注意力输出U_b,j的区域数量属于查询建筑，不属于支持建筑。
7. 支持标签只在建筑级编码；跨支持聚合读取结果U，不直接聚合原始V替代读取。掩码贯穿计算；无合法支持为零增量。图中3栋为示意，不固定K=3。
8. ΔF_mem=QW_mem(ΔT_mem)，最终h(F_A+ΔF_mem)。线性投影建议无偏置或显式掩蔽无支持位置，确保零支持真的产生零更新。
9. 当前查询损毁GT不参与分区或推理检索；训练标签用于监督与支持记录标注。图1的“Training images + annotations”表示建库数据来源，不表示标签输入特征编码器。训练查询排除自身与协议禁止来源；推理来源规则按具体设置定义。Val/Test不入库。

## Figure 1 caption

**Overall architecture of the candidate retrieval-augmented model.** The online path extracts pixel, structural and building-probability features, constructs regional context, and reads local representations from retrieved support buildings before pixel classification. A direct pixel pathway carries F_A to the final residual combination. The offline path constructs building records from training cases using a matched feature-extraction pipeline; the resulting memory remains active at inference. Solid arrows indicate online computation and dashed arrows indicate offline construction.

## Figure 2 caption

**Building-constrained soft region generation and pixel-preserving interaction.** (a) Predicted building groups restrict local region candidates; structural features and local centres determine soft assignments Q. The illustrated connections show permitted within-building and excluded cross-building associations, not true damage boundaries. (b) Both pixel content F and assignments Q enter weighted pooling, followed by masked regional interaction producing updated tokens T_A and the increment ΔT_local. (c) Projection and unpooling return this increment to the pixel space while preserving the original feature pathway.

## Figure 3 caption

**Building-level retrieval followed by per-building local support reading.** (a) A pooled query key selects valid support records under provenance, deduplication and masking rules. (b) Query regions read each retrieved building's regional set through cross-attention with shared parameters, yielding separate readouts U_b,j; three supports are illustrated for clarity. (c) Support-level aggregation combines these readouts with building-label embeddings and masks, and the projected memory increment is unpooled to pixel space. With no valid support the memory update is zero; support building labels are not local damage annotations.

## 画图方案与质量检查

- 类型：三张Solution overview/detail；图1双层在线/离线，图2顺序分面，图3检索/并行读取/聚合分面。避免将全部节点挤进一张图。
- 画布宽1200px，高分别690/870/960px；主文字20px，辅助19px。建议180mm全宽排版时最小字约8.1pt；更窄时需重新排版，不直接压缩成单栏。
- 配色：灰色基础组件，蓝色当前区域，橙色历史支持；附文字标签，不仅靠颜色区分。无装饰性立方体、伪造注意力热图或实测结果。
- 工具：沿用可编辑SVG，浏览器渲染PNG并逐张检查。SVG为正式编辑源，PNG仅预览。后续可在矢量编辑软件调整；目前未提供原生Draw.io文件。
- 图1修正Memory读线穿过分类框；图2拆分过长标签。无关键可见裁切；最终期刊字体、版心、真实实现以及训练/缓存更新协议仍需投稿前复核。
- 图中超参数和候选操作是设计约定，而非本轮对代码逐一核验的结论。不得将三个分图自动计作三项创新。

文件：BRIGHT_fig1_overview_v2、BRIGHT_fig2_regions_v2、BRIGHT_fig3_memory_v2；每张均有SVG及PNG。
