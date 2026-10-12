# 图标化主路线图

两版对应相同候选架构：BRIGHT_icon_overview_en_v1（英文论文版）和BRIGHT_icon_overview_zh_v1（中文理解版）；各有SVG及PNG。不是训练期Memory替代方案，Memory仍在推理前向中使用。

## English caption

**Overview of the candidate building-region retrieval-augmented network.** Pre-event optical and post-event SAR images are encoded by DamageFormer. Building-constrained soft regions pool and exchange pixel information, producing regional queries and a pixel-preserving feature pathway. A building-level query key retrieves training support cases, whose local regional sets are read by cross-attention; the readouts are aggregated, projected and mapped back to pixel space before classification. Memory records are constructed offline with a matched feature pipeline and remain available at inference. Image tiles, region divisions, token links and damage maps are schematic illustrations, not observed data, measured attention or experimental predictions.

## 中文图注

该候选网络利用历史建筑的局部支持增强当前影像的像素预测。灾前光学与灾后SAR影像经DamageFormer提取特征，建筑约束区域聚合和交互形成当前区域表示，并保留像素特征直通路径。当前建筑Key检索训练集案例库，交叉注意力读取支持建筑的局部表示，读取结果聚合、投影和回填后参与最终分类。案例库离线构建、推理时读取；图中影像、建筑分区、连线和输出均为机制示意，不表示实测结果。

## 设计及使用边界

- figure-designer指导下采用“在线主路径＋离线建库”双层架构，不将完整公式堆入总览。
- 灰蓝表示当前影像与区域，橙色表示历史案例；实体图标替代长文字框，配文字防止只靠颜色理解。
- 1400×810矢量画布，最小22px；180mm全宽放置时约8pt。最终仍需按投稿版心核对，避免直接缩为单栏。
- SVG是可编辑正式源，PNG仅预览。已检查渲染，修正英文标题编号重叠、图例间距和案例卡片溢出。
- 图中区域分块仅是软关联的示意；节点连线不指定已实现的精确邻接规则。
- Matched encoder包含与查询一致的骨干、区域与Key生成流程；不是任意固定编码器。更新相关参数时需同步案例库。
- Query key线表示建筑汇总与投影后的检索查询；其完整定义见已有Memory细节图。
- ΔF_mem包含支持级聚合、通道投影与按Q回填，不直接把原始区域向量加到像素特征上。无合法支持时应为零；来源掩码和去重在细节图/正文说明。
- 三个支持卡片不固定K=3；支持标签是建筑级，不是每个局部区域的真实损毁标注。
- 仅改善展示，不证明候选结构有效或创新。本轮没有修改模型、读取新的实验结果或启动训练。
