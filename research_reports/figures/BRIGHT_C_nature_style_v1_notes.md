# 模型图说明

按用户本次附图重绘推理检索版，未改成训练期Memory。使用figure-designer技能：白底、细线、灰/蓝/赭色低饱和配色、Arial英文标签、a–c分面、减少解释性文字；“Nature类风格”仅指视觉设计，不表示已满足某刊投稿规范。

## 图注（英文）

Building-level retrieval supplies local support representations to a pixel-preserving prediction pathway in this candidate architecture. **a**, DamageFormer features are enriched with regional context and retrieved support features before pixel-wise classification; the memory contains training cases only and remains active at inference. **b**, building-constrained soft assignments pool pixel features into regional tokens, and regional updates are mapped back through Q while retaining the original pixel pathway. **c**, building keys select valid support buildings, whose regional sets are read by cross-attention and aggregated with support-level label information. Connections and regional shapes are schematic, not measured attention or ground-truth damage locations; the architecture and its benefits remain to be validated.

F: fused pixel features; R: structural features; P: building probability; Q: pixel-to-region assignments; T_A: updated query-region representations; V_j: local representations of support building j. The four output classes are Background, Intact, Damaged and Destroyed. Invalid or absent support produces zero memory update; source exclusions and masks must be maintained in implementation.

## 图形设计与检查

- 类型：Solution overview，整体路径+局部展开；纯流程图不足以展示两个细节分面，部署架构图会引入无关工程内容。
- 1200×790矢量画布，主体19px、辅助18px、分面26px；建议先以190mm宽排版，最小文字约8.1pt。若目标版心更窄，需增大字体或拆图，不能直接缩为单栏。
- 灰色为基础组件，蓝色为区域处理，赭色为检索支持；均配文字，颜色不代表已成立的新贡献。
- 实线是在线前向，虚线是离线建库；推理读取箭头保留。未引入训练期队列或额外损失。
- SVG为可编辑矢量源，PNG仅作预览；使用已有矢量工作流，避免栅格生成文字错误。已查看渲染图，无明显裁切和标签重叠。
- 图中不设置数值坐标轴，无虚构结果、渐变和阴影。中文长说明移至图注，保留候选方法状态说明。
- 当前预览无关键可见排版问题；投稿前需要用户核实最终版心字体大小、期刊要求，以及各模块是否与实际实现一致。

文件：BRIGHT_C_nature_style_v1.svg / BRIGHT_C_nature_style_v1.png。
