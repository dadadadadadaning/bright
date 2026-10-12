# 参考图2布局草稿

使用figure-designer技能生成Solution overview。上方约50%为主线和训练案例支持来源；下方左44%为区域模块、右52%为Memory读取细节，中间保留间隔。采用参考图2的当前/支持关系组织，借鉴参考图3的双模态入口；不复制原论文方法模块。没有新增SAM2、LoRA、原型或双检测头。

SVG为可编辑矢量源，PNG为预览。1320×840约1.57:1，最小字号21px，以180mm全宽放置约8.1pt；单栏需重新排版。蓝色当前区域、橙色历史支持、灰色复用骨干，文字同时编码含义。图标和节点均为示意，不能作为实测结果或新颖性依据。

实现约定：DamageFormer内部仅概括RGB/SAR路径及既有融合，不意味着两路权重共享、主干冻结或每层交互。主干同时提供F、结构R和建筑概率P；总览为简洁未逐条画出R/P。T_A=T+ΔT_local；区域输出投影与按Q回填保留F直通。Memory三路Attention共享参数，输出U分别进入支持聚合；其中标签编码、有效mask、投影和无支持置零等留在详细方法中。总览的ΔF_mem是像素空间增量。只用训练集建库，并与查询特征版本匹配；推理时读取。

图注草稿：The candidate framework combines current-image regional context with retrieved building-level support. (a) RGB and SAR features are processed by DamageFormer and a building-constrained regional module. A current-building key retrieves training support records, whose local representations augment the pixel pathway before classification. (b) Pixel features and soft assignments jointly produce regional tokens, whose updates are projected and unpooled through a residual pixel pathway. (c) Each support building is read separately by shared-parameter cross-attention before support aggregation and pixel-space mapping. All icons and links are schematic, and efficacy remains to be validated.

质量检查：已渲染检查，修复支持卡片标签与token重叠、读取结果与聚合线的连接。无数据轴、伪造性能或装饰渐变。正式投稿前仍需核对实际模块与最终字号。未修改模型代码或启动实验。
