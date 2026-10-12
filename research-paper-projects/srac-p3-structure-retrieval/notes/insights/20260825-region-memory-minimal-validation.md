# BRIGHT 区域记忆检索最小验证方案

- 用户研究方向：灾前RGB–灾后SAR建筑损毁评估
- 用户研究问题：如何实现SAR raw mean/std、类别均衡Memory Bank、场景元数据和Cross-Attention检索；超像素是否主要用于加速
- 分析日期：2026-08-25
- 论文来源：SRAC结构图；SSN；SpixelFCN；IEM；RCA；RAC；Pin the Memory；Memory Attention
- 趋势来源：CVF正式论文和论文官方代码

## 核心判断

超像素既降低检索规模，也把决策单位从噪声像素改成结构区域；对BRIGHT而言，后者比纯速度更重要。但检索机制不应继续等待一个尚未通过的可学习超像素。第一轮应直接使用训练集官方建筑实例或高纯度建筑区域构造离线Memory，以无训练kNN验证“区域记忆是否真的补充Damaged”。只有该信号成立，才值得训练Cross-Attention和重新研究超像素分区。

## 可迁移模块

| 模块 | 公开依据 | 最小实现 | 关键约束 |
|---|---|---|---|
| 区域统计旁路 | SRAC图Stage 3 | 每区域计算SAR mean/std并经MLP投影，与主嵌入相加 | 统计量只使用训练集均值方差标准化 |
| 类别均衡Memory | IEM、RAC、RCA | Intact/Damaged/Destroyed各自固定容量；L2归一化key | 验证/测试只读，按类Top-k防止Intact淹没尾类 |
| 元数据Embedding | SRAC图Stage 4 | 灾害类型Embedding + 连续几何MLP | Region ID容易成为事件捷径，主实验先不用 |
| 检索注意力 | Memory Attention、RAC | 当前region为Query，Top-R邻居为Key/Value，MHA+FFN | 必须先以无训练kNN证明检索邻居有标签纯度 |
| 跨事件记忆 | Pin the Memory | Leave-one-event-out建库与查询 | 同事件检索与跨事件检索分别报告 |

## 复杂度

像素直接检索的暴力复杂度为 `O(N_pixel × N_memory × d)`，存储也随所有有效像素增长。区域化后为 `O(N_region × N_memory_region × d)`；再使用近邻索引后，完整Memory搜索被替换为Top-R候选，Cross-Attention只计算 `O(N_region × R × d)`。例如一个1024×1024影像有约一百万像素，而数百到数千个结构区域即可覆盖同一场景，查询和存储通常下降数百到数千倍。实际倍数取决于特征分辨率与区域数，不能从结构图得到固定值。

## 最小验证链

1. RM0：冻结P3/H6/Q0，从训练集高纯度建筑区域离线导出region key、label、event、geometry、raw SAR mean/std。
2. RM1：不训练Transformer，仅余弦kNN；按类Top-k计算三个类别分数。
3. RM2：加入raw mean/std旁路，比较RM1。
4. RM3：只在RM2通过后训练Cross-Attention；Memory保持冻结。
5. RM4：只在RM3通过后加入灾害类型/几何Embedding，并做元数据消融。
6. RM5：leave-one-event-out Memory，检查收益是否来自同事件捷径。

## 必须记录

- region Macro-F1、Damaged Recall、Destroyed Recall；
- pixel valid mIoU、Damaged IoU、Event mean/worst；
- Recall@R、同标签邻居比例、同事件邻居比例；
- P3错/检索对、P3对/检索错及净纠正；
- Memory每类数量、查询耗时、显存/内存；
- full-memory、event-excluded、label-shuffled三个对照。

## 准入门槛

RM1或RM2应至少满足：Damaged检索Top-R同标签率显著高于类别先验；P3错而kNN对多于P3对而kNN错；event-excluded仍保持正净互补；至少10/14事件方向非负。若无训练kNN不满足这些条件，Cross-Attention无法凭空创造可检索语义，应停止Memory路线并回到SAR表示学习。

## 建议题目方案

1. **Event-Safe Region Memory for Long-Tailed Optical–SAR Building Damage Assessment**
   - 任务：BRIGHT跨事件损毁识别。
   - 模块：类别均衡区域记忆、事件排除检索。
   - 最小实验：full/event-excluded/shuffled memory。
   - 预期贡献：证明Memory改善尾类而非记住事件。
2. **Raw-Statistics Preserving Region Retrieval for Native-SAR Damage Severity Recognition**
   - 任务：Damaged/Destroyed条件严重度。
   - 模块：深层区域嵌入+SAR mean/std旁路+Top-R注意力。
   - 最小实验：无raw、mean、mean+std消融。
   - 预期贡献：保留深层网络易丢失的SAR散射统计。
3. **Optical-Anchored Building Memory with Metadata-Calibrated Retrieval**
   - 任务：RGB定位、SAR损毁分类。
   - 模块：RGB建筑区域、SAR损毁key、灾害类型与几何条件化。
   - 最小实验：无元数据、几何、灾害类型、全部元数据。
   - 预期贡献：区分物理场景条件与事件ID捷径。

## 风险

- 使用训练/验证GT生成查询区域会高估部署效果；
- Region ID、事件名和地点可能导致记忆事件而非损毁；
- 低质量区域key会导致检索近邻标签纯度低；
- 全训练集动态Memory会增加训练不稳定，第一版应离线冻结；
- 仅报告平均mIoU会掩盖Damaged和最差事件下降。

