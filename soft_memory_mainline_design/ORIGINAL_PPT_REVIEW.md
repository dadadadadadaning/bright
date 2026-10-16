# 新图.pptx — 逐页结构复核（不采用原数值）

原文件12页，大小17.5×9.1667英寸；原件未修改。第1–3页主体为可编辑PowerPoint形状，第11–12页是表格；第12页没有图片。

| 页 | 原图用途 | 改为新主线时的动作 |
|---|---|---|
| 1 | DamageFormer pixel–region–memory总图；Train-only建库、Frozen A1、RGB guided Q、单区域摘要、Top8、线性像素区域融合 | 重画为“冻结前端/固定Q→父区域+双槽→父key选Top64→双槽案例Reader→像素联合头”；画出P0=grid_probq直达最终logits的分支与logP0残差公式。Train库使用相同前端和Q，但GT仅用于Train支持标签。 |
| 2 | 旧Structure-guided soft region encoding：≤9局部候选、三轮Q、结构adapter、注意力均值、方差与几何T128 | 保留为父T128来源的小插图/附录，主图改成固定Q后的双槽定义：w0=Q(1−B)、w1=QB。两个槽分别汇总F256、Train软标签、占比，保留父T128。不得将标题中的soft assignment当成当前soft routing。 |
| 3 | 旧Memory reader：Top8、T128 K/V、元数据label embedding、PreLN+FFN、learned slots fallback、同Q回投、WpF+WcC | 改Top64父案例→最多128子槽；K来自子槽F256+父T128+role+fraction，V另加四类标签embedding。当前是多头cross-attn+context_projection，无图示旧FFN/残差层。空支持context=0且有mask。父上下文按Q回投、子槽和Memory按对应w回投，非全部按Q。最终是logP0+上采样的非线性joint head。 |
| 4 | 灾害地理分布与示例 | 保留数据介绍图；增加真实Train/Val/Test划分与Train-only支持规则的小示意。不能把同事件支持排除说成数据划分本身是event-disjoint。 |
| 5–7 | 18个旧Test例子，列为RGB/SAR/GT/UNet/DeepLab等/Ours(A1) | 作为历史前端比较附录保留。新主图需真实重导出grid_probq/A/B/C/D，绝不能将Ours(A1)改名D。每页减少行列、同尺度放大。 |
| 8 | noto案例：最大整体mIoU增益 | 替换为soft减少混合的机制例图，含Q、B、w0/w1、A/B/C/D与rescue/harm图；避免以“最大收益”选取作为代表性证据。 |
| 9 | turkey案例：Damaged提升 | 替换为Memory成功/失败成对案例：查询位置+真实检索Train案例+标签分布+rank+cross-attn可视；并说明注意力权重不是贡献证明。 |
| 10 | myanmar局部Damaged恢复 | 改为边界、小建筑、低B、类别混合中至少一个失败案例；增加500与5000步一致窗口的变化，不能把旧A1恢复当成新Memory恢复。 |
| 11 | 主结果表+旧逐项累加消融+Test500文本 | 清空旧数值，分为同协议主结果表与2×2交互消融表；明确foreground mIoU(3类)和mIoU(4类)；数值待填。 |
| 12 | 逐事件结果表（无图像） | 主文优先逐事件配对差值点图/热图，保留表作附录；应以统一split全局confusion指标为主，event均值另标。 |

## 已核对当前fix2代码的具体事实

- `slots.py:route_from_q`：slot0=Q×(1−B)，slot1=Q×B；capacity_control为feature grid上4格checkerboard。两槽权重相加回原Q。
- 槽位角色不是GT类，背景槽仍可能含Damaged证据；固定Q与soft routing不保证连通、不跨建筑或实例内部。
- A/B使用Q centroid空间近邻（包括自己），支持标签来自native A1预测分布；C/D使用Train支持软标签，排除同图、同事件。
- 两个query槽都可读同一64父案例下全部合法子槽，不是建筑槽只能读取建筑槽。
- 一个支持value由child F256、parent T128、child fraction、role及label构成；事件/disaster元数据并非该Reader的embedding输入。
- `model.py`父分支`region_encoder([F256_mean,T128])`，槽分支`_encode_slots`，Memory分支`_read`；三者与pixel projection、P0拼接，经过非线性1×1头。
- 最终 `logits=log(clamp(P0))+upsample(delta_grid)`；delta末层zero-init使初始输出为P0，非拒绝门控。
- 原ppt Linear WpF+WcC、Top8、learned missing-support slots、T128-only K/V、PreLN+FFN链都不对应此实现。

## 新可视证据的最小集合

1. 总体同源预测：RGB、SAR、GT、P0、B、C、D，统一裁剪，来自至少多个事件，同时包含失败。
2. soft机制：Q hard-argmax轮廓（明确仅展示）、B连续热图、两槽权重/类别混合度与对应B−A、D−C变化。
3. Memory机制：查询区域、实际选中Train支持、ID/event/rank、soft标签和attention；配套随机/打乱干预，图本身不能证明因果。
4. 训练退化：同窗500/1500/5000预测，附固定Train/Val loss和类混淆趋势。
5. 局部风险：低B真实建筑、相邻不同损毁、细小建筑/边界各至少一例；提供原正确→错(harm)及错→对(rescue)。

案例选择用预定分层规则和固定列表，记录image_id、split、crop坐标、checkpoint SHA、源图像大小、真实区域ID和支持ID；不得用Val/Test标签选择邻居。可视机制在Val开发、最终Test只展示冻结方案的事后案例，并标明选择规则。
