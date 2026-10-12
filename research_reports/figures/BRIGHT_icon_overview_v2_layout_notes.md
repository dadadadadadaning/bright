# 图标主路线图排版修订v2

保留v1及所有细节图，新建中英文v2的SVG/PNG。仅修改展示，不改模型代码或实验。

使用figure-designer技能；Solution overview，采用五栏对齐在线主线＋横向离线/检索带。模块同顶同底、统一字级；预测图及类别图例合并一栏。像素直通线走上方专用通道，建筑查询线与支持回读线分开；Memory支路不穿过标题和公式。灰/蓝表示当前样本，橙表示历史支持，配合线型和标签，不仅靠颜色理解。

输出：1440×750；正文23px、辅助22px。建议以185mm或更宽版心使用（最小字号约8pt），更小版心须增大字号或拆分图，而非直接缩为单栏。SVG为矢量源，PNG为预览。Nature类简洁视觉风格不等同已符合某刊投稿规范。

图形均为示意，没有真实实验预测或注意力。建筑软区域、节点连接、三栋支持不分别主张真实边界、固定拓扑或K=3。区域块输出F_A与T_A；ΔF_mem包含逐栋读取后的聚合、投影和按Q回填；没有有效支持则增量应为零。Matched encoder包括同版本区域/Key特征流程，推理仍需Memory。

Caption: The candidate network combines building-constrained regional context with inference-time reading of local support representations. The upper path extracts features from pre-event RGB and post-event SAR imagery, forms regional queries, reads retrieved supports and combines the resulting pixel-space increment with the retained current-image pathway before classification. The lower band constructs memory from training cases with a matched feature pipeline and retrieves building-level records using the current query key. All icons and token connections are schematic; they do not represent measured images, attention or predictions.

已渲染检查中英文排版；英文较长标签缩短，修复文字越框和支持回读线的连接。最终实现与投稿版心仍需核对；本次未做新颖性或性能验证。
