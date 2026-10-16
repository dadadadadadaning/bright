# 建筑损毁论文：参考WANet的图表重排与正文修订

[修改后论文Word](paper_wanet_style_revision.docx) · [论文PDF](paper_wanet_style_revision.pdf)

[逐图解读与写作说明Word](figure_explanations_and_writing_guide.docx) · [说明PDF](figure_explanations_and_writing_guide.pdf)

[可编辑图件PPT](all_figures_editable.pptx) · [PPT预览](all_figures_editable.pdf)

[空白结果表与逐图材料清单](paper_result_tables_blank.xlsx) · [原数值核对表（非正式成绩）](reported_numbers_audit_only.xlsx)

[完整18例补充图集](supplementary_qualitative_atlas.pdf)

## 本次完成

- 以用户最新《论文20261016（1）.docx》为底稿，保留89处原公式，重新组织实验与结论。
- 主文安排8组图、6组主表；表4含信息来源与固定Reader两部分。总体、结构、内容、支持、检索和成本分别回答不同问题。
- 重绘3张方法图，分开Train-only离线建库、在线支持选择、原始支持读取与像素联合分类；修正原PPT的Frozen A1标签。
- 从用户PPT原始图片对象重新排定性图，真实图片字节不改；原18例全部保留为补充图S1–S3，主文选6例提高可读性。
- MPF是旧像素前端，Ours只指完整方法；没有完整方法输出的格子留白，没有把MPF改个标题冒充Ours。
- B、Q、b(Q)、H(Q)、支持ID/案例及相似度尚缺实测时留白；没有生成伪热图、伪邻居、曲线、混淆矩阵或性能。
- 原稿重新填入的数值完整存入核对表，不直接删除；正式结果单元格继续保持真正空白。
- 正文压缩工作批注，另给11页逐图说明：每图怎么读、写什么解释、需哪些实测证据与材料。

## 已纠正的具体问题

1. 低清三张预测拼图被放在“各类IoU与总体mIoU”图题下，后面还重复旧图。现已拆清内容与图题，只保留规范布局。
2. 表3的90.29、72.51、96.13等是旧F1/OA串进mIoU和边缘错误率列，不能作为结构消融值。
3. 旧base/region/roi/memory不能按递进顺序直接赋给新的MPF-HA、Parametric Support和监督检索；尤其旧ROI不等于参数化支持。
4. Ours行四个IoU的算术平均69.64与原写69.62不一致；SiamCRNN也有差异。这些已保留原值和核算，需回原始计数或指标口径确认，不能擅自选一个替换。
5. PPT第11页A/B/C/D的数值缺来源绑定。它们独立保存，不能搬到本稿的余弦/监督检索行，更不能预写soft与Memory协同。

## 为什么没有“预测大概成绩”

结构与流程无法推导可信的mIoU、F1、命中率或延迟。按用户“数值有不确定的空着”的要求，未确认的正式数值一律留空。说明书另列可检验的方向性假设，不把假设写成已经观察到的涨幅。

## WANet借鉴范围

依据用户提供全文，借鉴三线表、图题在下/表题在上、总体比较→模块消融→内部机制→失败边界的论证层次。不迁移其沟渠二分类指标、小波组件、数据集或成绩，也不照搬17列过密定性网格。

Word为A4单栏审阅稿，适合继续填数和修改；宽图按约17cm通栏设计，方法图提供SVG/PDF，可移入目标期刊双栏模板。它不是最终投稿模板。参考论文的详细逐图/逐表分析在`wanet_reference_design_review.md`。

## 下一步最小闭环

1. 先锁定MPF与完整方法的同协议模型、数据名单与解码，收集完整评估的原始4×4混淆矩阵和逐事件计数。
2. 补图5/8同一裁窗的真实Ours输出；已有预测可直接复用，不为排版重新训练。
3. 做表2四组结构消融，再做表3固定Q条件，输出图6真实响应。
4. 固定Train库，做表4信息来源和共同Reader只换ID，输出表5及图7实际支持。
5. 对核心配置做多种子、成本与逐事件分析；额外SAR、双槽、筛库作为单独实验，不混填表格。

当前环境没有可调用的论文写作skill；已完成检索确认。本次直接按原稿、原PPT和参考论文全文执行结构审查、改写、排版与文件验证，没有声称调用不存在的skill。

本次没有训练、GPU推理、新Test访问、修改原稿或覆盖原PPT。所有缺失证据仍待真实结果补充，空白不代表零分或已完成实验。

## 文件说明

- `paper_wanet_style_revision.docx/.pdf`：18页论文正文修订。
- `figure_explanations_and_writing_guide.docx/.pdf/.md`：11页整体思路、8图解读、6表逻辑、数值政策与后续材料。
- `all_figures_editable.pptx/.pdf`：12页可编辑图件，含主图及补充图；文字、箭头、边框及各真实裁块可独立编辑。
- `paper_result_tables_blank.xlsx`：与正文一致的空表、逐图解读、原图映射及方向性假设。
- `reported_numbers_audit_only.xlsx`：用户上传原值、指标错位及算术核对，仅为追溯，不是新验证成绩。
- `supplementary_qualitative_atlas.docx/.pdf`：3页全部18例外部模型与MPF图集。
- `figures/`：各图PNG、单页PDF、方法SVG及来源保真记录。原始162张裁块保存在PPT内，记录用于追溯。
- `table_figure_manifest.json`、`validation.json`、`SHA256SUMS`：图表配置、验证与文件校验。
