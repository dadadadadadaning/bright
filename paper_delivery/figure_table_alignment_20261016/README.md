# 图表与模型身份统一修订稿

旧图可以继续使用。需要纠正的是图中模型的身份和论证范围，而不是因更换论文设计就丢弃全部预测图。

[下载修订Word](paper_figure_table_aligned.docx) · [查看PDF](paper_figure_table_aligned.pdf) · [图表对应与空结果表](figure_table_alignment_and_blank_results.xlsx)

## 这次实际改了什么

1. 以用户上传的《论文20261016.docx》为底稿修改，保留其已写为“Frozen MPF”的图1，不退回旧图；原有4张方法/数据图、89处公式完整保留。
2. 将原稿图5–7的18个多模型样例恢复为正文图11–13，采用横向整页，预测图片字节未改动；图注明确“Ours (A1)”对应像素前端MPF。
3. 总体表2只放外部模型、MPF和完整方法Ours，补齐U-Net、DeepLabv3+、SiamAttnUNet、SiamCRNN等旧图实际列。内部读出配置单独列于表5。
4. 用功能明确的标签替代A1/J0/Jcur/J1/J2/J3；旧编号只在附录用于查找历史代码。把“空支持”改为“参数化支持”，说明其包含可学习先验。
5. 表格和图注同步修改，新增模型名称、图列和检查点的对应表。“同预算训练”放入协议说明，不再当作方法名或来源。
6. 清掉摘要中遗漏的旧分数，与正文空结果表保持一致。所有正式结果数值继续留空，没有从局部图猜测全测试集成绩。
7. 三张内嵌局部分数的旧放大图未丢弃，另存为`legacy_zoom_figures_review_only.docx/.pdf`，原图PNG也完整附在`legacy_figures/`。其数字不是本次新验证的正式成绩。

## 为什么之前图和表对不上

旧多模型图的实际预测列是：UNet、DeepLabV3+、SiamAttnUNet、SiamCRNN、DamageFormer、Ours (A1)。

旧总体表却没有UNet与实际A1前端行，末行是完整检索模型；新版总表又只保留部分外部模型并混入内部消融。因此，读者很容易把旧图的“Ours (A1)”误认成完整方法。

表格可以有ChangeOS、ChangeMamba，而定性图不展示它们。图是模型或样例的子集，本身没有问题。真正必须一致的是：同一模型名称是否对应同一权重、同一输入/解码、同一数据划分；同一图片行是否使用同一裁窗。

## 正式命名

| 原编号 | 本稿短标签 | 含义 |
|---|---|---|
| A1 | MPF | 多模态像素前端，直接评价既有分类器 |
| J0 | MPF-HA | 冻结同一前端，仅适配像素分类头；HA指Head Adaptation |
| Jcur | Pixel–Region | 当前区域特征与像素特征融合，不使用支持读取器 |
| J1 | Parametric Support | 使用可学习支持向量，没有真实Train案例 |
| J2 | Cosine Retrieval | 同一Train库，使用原始余弦选择真实支持 |
| J3 | Supervised Retrieval；总体表写Ours | 跨事件类别监督投影选择支持的完整配置 |

MPF是基础网络身份；其余是实验配置名称，不是六个新模型品牌。两种真实检索都排除同事件，区别在于排序是否经过跨事件类别监督，不能仅把后者写成“跨事件”、让人误以为前者使用同事件案例。

## 已找到的旧图来源证据

仓库已有`a1_models.json`、`test_comparison_ppt/build/data.json`、`test_evidence_details/build/analysis.json`和拼图记录。

- 旧图末列`Ours_A1`绑定`DamageFormer_S8_MICB_DSMIL`与`best/best_val_miou.pth`，保存的SHA256为`cf834b5929c96f84642b8b856c985dc818eecac66a1f01ca3803e1b1dc200add`。
- 记录参数包括`roi_alpha=0.1`、`detach_building_prior=true`、`mil_residual_scale=0.0`。这进一步区分了旧像素前端图与新的外部支持检索框架。
- 其余五个模型也记录了构造参数、检查点和SHA256。
- 18个样例的影像ID及裁窗坐标完整。保存的六组裁窗混淆矩阵，其GT行和一致，按矩阵重算的IoU与原分析记录一致。

上述是代码、保存元数据和计数的核对。当前工作区没有原始预测标签PNG与对应GT，尚未重新从数组执行验证，也没有重新读取服务器上的原权重。详细记录见`legacy_traceability_audit.json`及Excel的“每个模型的结果绑定”“18个旧图样例”。

## 下一步最省时间的顺序

1. **找回原预测，不重训。** 优先查看服务器`/root/shared-nvme/BRIGHT-test-sixmodels-v2/predictions/`或本地备份`BRIGHT_test_sixmodels_images.zip`。复核需要各模型的`*_labels.png`、对应`*_building_damage.tif`和已记录的裁窗坐标；彩色展示PNG不能自动替代原始类别标签。
2. **总体表按全量记录回填。** 已有完整Test原始混淆矩阵且来源匹配时直接复用；只有18窗矩阵时只能报告选定样例，不能填全Test878表。没有全量预测/计数时才补评估，不因缺表格就重训。
3. **补完整框架图14。** 保留同样裁窗时，RGB、SAR、GT和来源匹配的外部模型预测可继续用。新增真实MPF-HA、区域、参数化支持、余弦检索及完整方法预测；已有输出即可读取，不必再次训练。
4. **局部放大图按原窗口复核。** 局部分数只代表该窗口；保留改善、持平、下降案例，避免把挑选的最大提升当作普遍收益。
5. **从同一份结果生成所有图表。** 摘要、表2、表9、曲线和图注用相同的模型标识、划分和原始计数回填。不能拿公开论文数字配本地不同权重的预测，也不能仅换列名把MPF图变成完整方法图。

本次未启动训练、GPU评估、Test推理或重新建库；未修改用户原稿和项目实验源码。

## 文件

- `paper_figure_table_aligned.docx/.pdf`：当前稿的直接修订版。
- `figure_table_alignment_and_blank_results.xlsx`：11张与Word一致的空结果表，以及命名、图表、18窗、模型权重绑定与后续步骤。
- `legacy_zoom_figures_review_only.docx/.pdf`：3张旧局部放大图的独立归档。
- `legacy_figures/`：旧6张结果图PNG，保持原始字节。
- `legacy_traceability_audit.json`：历史生成记录与计数核对，范围仅18个选定裁窗。
- `model_and_figure_map.json`、`validation.json`、`SHA256SUMS`：对应关系及交付检查。

本文件是审稿前工作版本。结果留空表示等待同源记录，不代表零值，也不代表所规划的实验均已完成。
