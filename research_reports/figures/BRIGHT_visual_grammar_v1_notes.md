# 图形化架构图v1：视觉符号与论文图注

任务：基于已有候选推理Memory架构丰富图形表达。使用figure-designer技能，保留a总览、b区域、c记忆读取三分面。没有新增方法、验证性能或修改模型代码。

## 视觉语法

| 图形 | 含义 | 限制 |
|---|---|---|
| 影像缩略图 | 输入或输出的二维空间对象 | 本版为示意，不能冒充真实RGB、SAR及结果 |
| 分块建筑 | 建筑内局部区域组织 | 只是软关联示意，不代表真实损毁边界 |
| 网格矩阵 | 像素特征或关联矩阵 | 必须标F/Q等，颜色不是实测数值 |
| 条带集合 | 区域token集合 | 不是一个条带对应一种损毁类别 |
| 矩形 | 有名称的计算，如Pool、Attention、Project | 不用圆圈表示任意神经网络模块 |
| 带+圆圈 | 同形状张量相加 | 不表示拼接、注意力或任意融合 |
| 小节点圆点 | 若出现，只代表图节点，并以图例区分 | 本版未用普通圆点承载运算 |
| 案例库图标 | 历史训练特征库 | 不表示推理样本入库 |
| 实线箭头 | 前向数据依赖 | 上下双向需确有双向计算，不能装饰性添加 |
| 虚线箭头 | 离线建库 | 浅色模块外框不具有箭头线型语义 |

主线方向左到右；当前Key向下进入检索，历史支持向上进入读取。放大模块内部允许折返，但每条线只有一个终点箭头。移开了F输入池化与池化输出的共线段。

公式只放语义接口：F_A=F+ΔF_local、Y=h(F_A+ΔF_mem)。注意力softmax、投影和池化归一化完整公式留在正文。ΔF_local、ΔF_mem均已处于像素特征空间；不能将原始区域token直接与像素F相加。

## 实现边界

- 主干输出F、结构R与建筑P；总览只显式写F。Building prior图标概括P产生的预测分组，不是以GT轮廓作为输入。
- Association包括同建筑候选约束、局部中心和结构特征；Q是软关联。实际选择可学习软区域还是固定SLIC须与实现一致。
- F和Q共同输入池化。Attention区域结果还生成T_A；该残差关系在总览与符号中表达，本版细节只画像素增量路径，完整T_A=T+ΔT_local见正文。
- 每栋支持单独Cross-Attention，权重参数共享；读出U对应当前查询区域。三栋只是示意，不固定K。
- 聚合中使用支持建筑标签及有效掩码，投影与Q回填后才形成ΔF_mem。无合法支持时必须为零。标签不得当作每个局部区域的损毁真值。
- Matched encoder包含相同版本特征、区域和Key生成流程。训练样本建库；推理仍使用检索。

## English caption

**Visual overview of the candidate building-region retrieval-augmented model.** (a) Pre-event RGB and post-event SAR inputs provide current-image features and building-constrained regional representations. A building-level query retrieves training supports, whose local evidence is read and combined with the retained pixel pathway before classification. (b) Pixel features and soft associations jointly form regional tokens; the resulting regional increment is projected and unpooled to pixel space. (c) Shared-parameter cross-attention reads each support building separately, followed by support aggregation, projection and unpooling. All icons, matrix colours and connections are schematic rather than experimental observations.

## 导出与排版检查

1500×990，Arial，最小24px；180mm宽排版约8.16pt。灰色复用编码、蓝色当前区域、橙色历史支持，同时使用文字和形状区分。SVG为编辑源，PNG为预览。无伪造数值、性能声明或热力图。遵照已有矢量资产工作流，不使用生成式图片制造不可编辑文字。

已渲染检查；修复F进池化和池化出Attention的共线方向问题。最终期刊版心和真实实现仍需复核。所有视觉改动不证明创新点成立。
