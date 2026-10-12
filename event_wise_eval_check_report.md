# BRIGHT Event-wise 评估与 Destroyed 错误诊断检查报告

检查日期：2026-06-22  
检查范围：`full_evaluate_unified.py`、`scripts/eval/`、standard ML 数据列表及现有评估输出。  
研究目标：找出拖累整体建筑损毁性能的灾害事件，并解释 `destroyed_iou` 下降的来源。

类别定义：

```text
0 background
1 intact
2 damaged
3 destroyed
```

## 1. 当前是否支持 event-wise

### 1.1 已有能力

`full_evaluate_unified.py --eval_protocol official` 已支持基础 event-wise 评估：

- `full_evaluate_unified.py:984` 为每个实际出现的 event 创建四类 confusion。
- `full_evaluate_unified.py:1014-1017` 按样本提取 event 并累计全像素 damage confusion。
- `full_evaluate_unified.py:1021` 输出 `event_metrics_official.csv`。

当前 CSV 字段只有：

```text
event
background_iou
intact_iou
damaged_iou
destroyed_iou
official_style_4class_miou
target_3class_miou
```

因此当前状态是：

```text
支持基础 event-wise IoU
不支持完整的 event 拖累分析
不支持完整 destroyed 错误归因
不支持 event 级 loc 指标
不支持多模型 event 结果自动合并
```

当前工作区中没有找到已生成的 `event_metrics_official.csv`，说明现有代码具备输出能力，但当前本地结果目录尚不能直接用于六模型 event 对比。若服务器结果目录也没有这些文件，需要按 official 协议重新评估或从已保存的 per-event confusion 恢复。

### 1.2 research 协议

`research` 协议也会输出 `event_metrics.csv`，并包含 event 级 loc 和部分误分率；但其 damage confusion 只统计 GT building 区域，且空类记 0，和 official 全像素协议不同。

论文分析不应把 research event CSV 与 official event CSV 混用。

建议以 official 全像素 confusion 为唯一 damage 数据源，同时为每个 event 额外保存定位 confusion 和计数器。

## 2. Event ID 来源与提取规则

### 2.1 当前数据格式

`standard_ML/test_set.txt` 每行是样本 ID，例如：

```text
bata-explosion_00000007
la_palma-volcano_00000123
turkey-earthquake_00001082
ukraine-conflict_00000468
```

数据文件沿用同一 ID：

```text
pre-event/{sample_id}_pre_disaster.tif
post-event/{sample_id}_post_disaster.tif
target/{sample_id}_building_damage.tif
```

因此 event 信息已经直接编码在 list 中的样本 ID 和文件名里。

项目范围内没有找到比样本 ID 更权威、完整的 event metadata 文件。

### 2.2 当前提取逻辑

`event_from_name()` 位于 `full_evaluate_unified.py:376-396`，优先：

1. 在 dataset 返回的 name 和 list entry 中匹配硬编码 `EVENT_NAMES`。
2. 若指定 `--event_index`，从 list 路径字段取某一级目录。
3. 否则尝试选择包含 `-` 的路径字段。
4. 最后使用 basename 的 `_` 前部分。

对当前已知 14 个 standard event，该逻辑能够正确工作。

### 2.3 当前 test set 实测

878 个 test 样本被分为 14 个事件：

| Event | Images |
|---|---:|
| bata-explosion | 23 |
| beirut-explosion | 27 |
| congo-volcano | 27 |
| haiti-earthquake | 15 |
| hawaii-wildfire | 14 |
| la_palma-volcano | 189 |
| libya-flood | 26 |
| marshall-wildfire | 18 |
| mexico-hurricane | 47 |
| morocco-earthquake | 116 |
| myanmar-hurricane | 28 |
| noto-earthquake | 18 |
| turkey-earthquake | 224 |
| ukraine-conflict | 106 |

### 2.4 推荐的 canonical event 提取规则

建议以 list entry 为权威来源，规则如下：

1. 读取 list 行并去掉目录和扩展名。
2. 去掉 `_pre_disaster`、`_post_disaster`、`_building_damage` 后缀。
3. 用正则去掉最后的图像编号：

```python
event = re.sub(r"_\d+$", "", sample_id)
```

例如：

```text
la_palma-volcano_00000123 -> la_palma-volcano
turkey-earthquake_00001082 -> turkey-earthquake
```

4. 将结果与允许的 event 列表校验。
5. 无法解析时直接报错或记录 `unknown`，不要静默把每张图像 ID 当成独立 event。

推荐优先级：

```text
list txt 的 sample ID
> dataset 返回的 sample name
> 文件 basename
> 显式 metadata event_id
> 父级目录名
```

当前数据的父级目录只有 `pre-event/post-event/target`，不能用于区分具体灾害事件。

### 2.5 当前 fallback 风险

如果未来新增 event 没有加入硬编码 `EVENT_NAMES`，当前 `event_from_name()` 的包含连字符 fallback 可能返回完整 sample ID，例如：

```text
new-event_00000001
```

这会导致同一 event 的不同图像被拆成多个 event。

因此应把“去掉末尾数字 ID”的规则放在硬编码匹配之前或作为可靠 fallback。

## 3. 建议的 event-wise 指标

每个 event 建议保存以下基础量：

```text
event
num_images
num_pixels
num_valid_pixels
num_building_pixels
num_background_pixels
num_intact_pixels
num_damaged_pixels
num_destroyed_pixels
```

其中：

```python
num_valid_pixels = count(gt != 255)
num_building_pixels = count(gt in {1,2,3})
num_intact_pixels = count(gt == 1)
num_damaged_pixels = count(gt == 2)
num_destroyed_pixels = count(gt == 3)
```

建议的性能指标：

```text
official_style_4class_miou
target_3class_miou
background_iou
intact_iou
damaged_iou
destroyed_iou
damaged_precision
damaged_recall
destroyed_precision
destroyed_recall
loc_iou
loc_f1
```

建议的错误流向指标：

```text
damaged_to_background
damaged_to_intact
damaged_to_destroyed
destroyed_to_background
destroyed_to_intact
destroyed_to_damaged
```

建议同时输出 event confusion 的 16 个原始计数，或保存独立文件：

```text
event_confusion_4class.csv
```

这样后续可以在不重新推理的情况下调整诊断公式。

## 4. 当前指标是否足够

当前 `event_metrics_official.csv` 不足以完成研究问题，缺少：

```text
num_images
各类 GT support
damaged/destroyed precision 和 recall
六个误分流向
event 级 loc_iou / loc_f1
destroyed FP 来源
impact score
```

当前 `event_mean_official_style_4class_miou` 是各 event mIoU 的等权宏平均，不按图像数、建筑像素数或 destroyed support 加权。它可以表达“平均事件表现”，但不能表示某个 event 对整体像素级结果的实际拖累。

`summarize_official_conf()` 已经能够从一个 event confusion 计算大部分 per-class 和误分率，但 `write_official_event_csv()` 没有把这些字段写出。

event 级 loc 尚未在 official 评估中累计。当前 official 只有一个 overall `loc_conf`，`event_confs` 只保存 damage confusion。

建议 event state 至少为：

```python
event_stats[event] = {
    "damage_conf": np.zeros((4, 4)),
    "loc_head_conf": np.zeros((2, 2)),
    "damage_building_conf": np.zeros((2, 2)),
    "num_images": 0,
    "num_pixels": 0,
    "num_valid_pixels": 0,
}
```

## 5. 空类别处理

### 5.1 当前 official 行为

当前 `official_per_class_from_conf()` 在：

```text
TP + FP + FN == 0
```

时将 IoU 记为 NaN。

因此：

- event 没有 destroyed，且模型也没有预测 destroyed：`destroyed_iou=NaN`。
- event 没有 destroyed，但模型预测了 destroyed：`destroyed_iou=0`。

这符合 union-based IoU，但不完全符合“只要 GT destroyed support=0 就缺失”的 event 分析需求。

### 5.2 推荐规则

event-wise 类别指标应以 GT support 为准：

```python
if num_destroyed_pixels == 0:
    destroyed_iou = NaN
    destroyed_recall = NaN
    destroyed_to_intact = NaN
    destroyed_to_damaged = NaN
    destroyed_to_background = NaN
```

destroyed precision 仍可在模型预测 destroyed 时计算，因为它反映假阳性；但表格中应明确：

```text
GT-destroyed metrics: support=0 -> NaN
prediction-conditioned precision/FP metrics: 可保留
```

### 5.3 当前 standard test

当前 14 个 test event 都有 destroyed 像素，但数量差异极大：

```text
haiti-earthquake: 2,815
morocco-earthquake: 5,535
myanmar-hurricane: 10,505
la_palma-volcano: 4,450,669
```

因此即使没有空 destroyed event，小 support event 的 destroyed IoU 方差也会很大。报告必须同时展示 `num_destroyed_pixels`，不能只按 IoU 排名。

## 6. Destroyed 错误诊断指标

### 6.1 已有指标

official 已有：

```python
destroyed_to_background = conf[3,0] / conf[3,:].sum()
destroyed_to_intact     = conf[3,1] / conf[3,:].sum()
destroyed_to_damaged    = conf[3,2] / conf[3,:].sum()
destroyed_recall        = conf[3,3] / conf[3,:].sum()
destroyed_precision     = conf[3,3] / conf[:,3].sum()
```

它们在 overall summary 中已计算，但目前没有全部写入 event CSV。

对 GT destroyed 而言，若无 ignore 和非法预测：

```text
destroyed_recall
+ destroyed_to_background
+ destroyed_to_intact
+ destroyed_to_damaged
= 1
```

这是很有用的完整漏检分解。

### 6.2 Destroyed FP 来源

建议新增原始计数：

```python
destroyed_fp_from_background_count = conf[0,3]
destroyed_fp_from_intact_count     = conf[1,3]
destroyed_fp_from_damaged_count    = conf[2,3]
```

并输出两种比例，避免名称含糊。

第一种：占所有 destroyed false positives 的来源比例：

```python
destroyed_fp_total = conf[0,3] + conf[1,3] + conf[2,3]

destroyed_fp_from_intact_share
= conf[1,3] / destroyed_fp_total

destroyed_fp_from_damaged_share
= conf[2,3] / destroyed_fp_total
```

它回答：

```text
模型预测出来的错误 destroyed，主要来自哪一类？
```

第二种：源类别被误判为 destroyed 的概率：

```python
intact_to_destroyed = conf[1,3] / conf[1,:].sum()
damaged_to_destroyed = conf[2,3] / conf[2,:].sum()
```

它回答：

```text
GT intact/damaged 有多大比例被升级成 destroyed？
```

建议不要只使用 `destroyed_fp_from_intact` 这一模糊名称，应显式区分：

```text
*_count
*_share
intact_to_destroyed / damaged_to_destroyed
```

### 6.3 Destroyed IoU 的完整解释

```text
destroyed_iou = TP_destroyed /
                (TP_destroyed + FN_destroyed + FP_destroyed)
```

下降可能来自两部分：

```text
FN 增加：
destroyed -> background/intact/damaged

FP 增加：
background/intact/damaged -> destroyed
```

因此只分析 `destroyed_recall` 不足以解释 destroyed IoU；必须同时分析 destroyed precision 和 FP 来源。

## 7. Impact Score 设计

### 7.1 整体建筑损毁拖累

建议保留用户提出的：

```python
event_building_pixel_ratio
= event_num_building_pixels / overall_num_building_pixels

impact_score
= event_building_pixel_ratio * max(
    0,
    overall_target_3class_miou - event_target_3class_miou,
)
```

它适合回答：

```text
哪个 event 对整体三类建筑损毁性能拖累最大？
```

优点：

- 同时考虑事件规模和性能差距。
- 不会让极小 event 仅因低 IoU 自动排到第一。
- 与论文主指标 `target_3class_miou` 对齐。

### 7.2 Destroyed 专项拖累

建议同时增加：

```python
event_destroyed_pixel_ratio
= event_num_destroyed_pixels / overall_num_destroyed_pixels

destroyed_impact_score
= event_destroyed_pixel_ratio * max(
    0,
    overall_destroyed_iou - event_destroyed_iou,
)
```

若 `num_destroyed_pixels=0`：

```text
destroyed_impact_score = 0
destroyed_iou = NaN
```

它更适合回答：

```text
哪个 event 真正在拖累 destroyed 表现？
```

### 7.3 推荐选择

不建议二选一。应使用两个互补排名：

```text
impact_score：整体建筑三类拖累，作为主排名
destroyed_impact_score：destroyed 专项拖累，作为诊断排名
```

本任务重点分析 destroyed，因此 destroyed diagnosis 文件应默认按 `destroyed_impact_score` 降序；worst-events 总表按 `impact_score` 降序。

注意：IoU 是非线性指标，这两个 score 是解释性启发式分数，不是删除该 event 后总体 IoU 的精确变化。

如果需要精确贡献，可额外计算 leave-one-event-out：

```text
overall metric - metric computed from (overall_conf - event_conf)
```

这比加权 gap 更严格，但解释和实现稍复杂。建议后续同时输出：

```text
impact_score
leave_one_event_out_delta
```

## 8. 自动诊断规则

不建议用固定绝对阈值定义“高/低”，因为不同模型尺度不同。推荐使用：

```text
高：高于 overall 值至少 0.05，或位于所有 event 的上四分位
低：低于 overall 值至少 0.05，或位于所有 event 的下四分位
loc 高：不低于 overall loc_iou - 0.02
loc 低：低于 overall loc_iou - 0.05
```

阈值应写入输出文件，保证诊断可复现。

### 8.1 Destroyed 被低估为 intact

条件：

```text
destroyed_to_intact 高
destroyed_recall 低
loc_iou 高
```

诊断：

```text
建筑基本找到了，但损毁头把 destroyed 判为 intact。
主要是 damage existence / 完好-损毁分离问题。
```

优先模型方向：

```text
A4-DG-beta
DamageExistHead
A4-HDC
```

### 8.2 Destroyed 被漏成 background

条件：

```text
destroyed_to_background 高
loc_iou 低
```

诊断：

```text
主要是建筑定位或 ROI 覆盖失败。
只增强 damage/severity head 不足。
```

优先方向：

```text
ROI 策略
loc head
building-aware fusion
边界/小目标定位
```

### 8.3 Destroyed 与 damaged 严重程度混淆

条件：

```text
destroyed_to_damaged 高
destroyed_to_background 低
destroyed_to_intact 低或中等
loc_iou 不低
```

`destroyed_recall` 不可能在 destroyed_to_damaged 很高时完全不受影响，因此“recall 不低”应解释为相对其他低性能 event 仍可接受，而不是绝对很高。

诊断：

```text
建筑和损毁存在性识别正常，主要是 damaged/destroyed severity 边界不清。
```

优先方向：

```text
Severity Head
HDC
large-scale MDDM
ordinal/severity loss
```

### 8.4 Destroyed precision 低、recall 尚可

条件：

```text
destroyed_precision 低
destroyed_recall 不低
destroyed_fp_from_intact_share 高
```

诊断：

```text
模型过度预测 destroyed，尤其把 intact 升级成 destroyed。
```

建议：

```text
校准 destroyed decision boundary
检查 class weight / focal loss
加强 intact consistency
```

若：

```text
destroyed_fp_from_damaged_share 高
```

则主要是 damaged 被过度升级成 destroyed，仍属于 severity 校准问题。

### 8.5 小样本不稳定

条件：

```text
num_destroyed_pixels 很低
destroyed_iou 极低或极高
```

诊断标签：

```text
LOW_SUPPORT_UNSTABLE
```

建议至少输出 support rank，并在低于阈值时不做强模型结论。阈值可先设：

```text
num_destroyed_pixels < 10,000
```

当前 haiti-earthquake 和 morocco-earthquake 属于这一风险区间。

## 9. Event 级 Loc 指标建议

当前 official event state 没有定位 confusion，因此 event 级 `loc_iou/loc_f1` 尚不支持。

建议同时输出两组：

```text
loc_head_iou / loc_head_f1
damage_building_iou / damage_building_f1
```

其中：

```text
loc_head：来自 output_loc
damage_building：来自 damage_gt>0 与 damage_pred>0
```

destroyed 错误诊断更推荐使用 `damage_building_iou`，因为 destroyed_to_background 来自最终四分类输出；它和最终 damage confusion 在同一预测空间内。

所有指标计算前都应过滤 `GT==255`。

## 10. 多模型比较能力

### 10.1 当前可批量评估的模型

`scripts/eval/eval_all_official_checkpoints.sh` 已能批量运行：

```text
A4
A4-D
A4-DG
A4 + DamageExistHead
```

M07 和 M11 有独立 official 脚本：

```text
eval_M07_A4DG_b03_official.sh
eval_M11_A4HDC_a03_official.sh
```

### 10.2 当前不足

1. 通用脚本查找的 A4/A4-D 目录名称与当前已有：

```text
M03_A4_MDDMbase_ROId5_C640_I200k_BS2
M05_A4D_MDDMlarge_ROId5_C640_I120k_BS2
```

并不一致，自动发现可能找不到 M03/M05。

2. M07、M11 不在同一个批处理脚本中。
3. 当前 summarizer 只读取 `summary_official.json`，不读取 `event_metrics_official.csv`。
4. 没有生成 `model × event` 的长表或透视比较表。
5. 每个 checkpoint 都输出 event CSV，但没有明确规定六个模型各选哪个 checkpoint；直接混入四种 best checkpoint 会造成比较混乱。

### 10.3 推荐比较集合

建议先固定每个模型一个 checkpoint 选择规则，例如：

```text
主比较：best_valid_miou 或 best_balanced_score
专项附录：best_destroyed_iou
```

至少优先分析：

```text
M03_A4
M05_A4D
A4-DG
A4 + DamageExistHead
M07_A4DG_b03
M11_A4HDC_a03
```

推荐顺序：

1. M03_A4：基础参照。
2. M05_A4D：检查 large-scale 对 destroyed 的影响。
3. A4-DG：检查 dual-scale gate。
4. M07_A4DG_b03：检查 beta=0.3 是否减少 destroyed→intact。
5. A4 + DamageExistHead：检查 damage existence 分离。
6. M11_A4HDC_a03：检查分层 severity 建模。

## 11. 建议输出文件

### 11.1 `eval_analysis/event_wise_summary.csv`

一行一个 `model/checkpoint/event`，包含用户要求的全部字段：

```text
model
checkpoint
event
num_images
num_pixels
num_valid_pixels
num_building_pixels
num_intact_pixels
num_damaged_pixels
num_destroyed_pixels
official_style_4class_miou
target_3class_miou
background_iou
intact_iou
damaged_iou
destroyed_iou
damaged_precision
damaged_recall
destroyed_precision
destroyed_recall
damaged_to_intact
damaged_to_destroyed
damaged_to_background
destroyed_to_intact
destroyed_to_damaged
destroyed_to_background
loc_head_iou
loc_head_f1
damage_building_iou
damage_building_f1
impact_score
destroyed_impact_score
```

### 11.2 `eval_analysis/event_wise_destroyed_diagnosis.csv`

建议字段：

```text
model
checkpoint
event
num_destroyed_pixels
destroyed_iou
destroyed_precision
destroyed_recall
destroyed_to_background
destroyed_to_intact
destroyed_to_damaged
destroyed_fp_from_background_count
destroyed_fp_from_intact_count
destroyed_fp_from_damaged_count
destroyed_fp_from_background_share
destroyed_fp_from_intact_share
destroyed_fp_from_damaged_share
intact_to_destroyed
damaged_to_destroyed
damage_building_iou
destroyed_impact_score
diagnosis_code
diagnosis_text
recommended_direction
support_warning
```

### 11.3 `eval_analysis/worst_events_top10.csv`

建议每个模型输出两套排名，并在同一文件中增加 `ranking_type`：

```text
overall_impact
destroyed_impact
lowest_target_3class_miou
lowest_destroyed_iou
```

不应只按最低 IoU 排名，否则小 support event 会占据榜首。

### 11.4 `eval_analysis/model_event_comparison.csv`

建议使用长表：

```text
event
model
checkpoint
metric
value
delta_vs_M03_A4
rank_within_event
```

也可额外输出便于阅读的宽表：

```text
event × model 的 target_3class_miou
event × model 的 destroyed_iou
event × model 的 destroyed_to_intact
```

### 11.5 建议额外文件

```text
eval_analysis/event_confusion_4class.csv
eval_analysis/event_support.csv
eval_analysis/analysis_config.json
```

`analysis_config.json` 应保存诊断阈值、模型别名、checkpoint 选择和数据列表 hash。

## 12. 建议实现结构

后续修改时建议分两层：

### 12.1 评估阶段

每个模型只负责输出可复算的原始统计：

```text
overall confusion
per-event damage confusion
per-event loc confusion
per-event counters
```

### 12.2 分析阶段

新增独立分析脚本，从各模型评估目录读取上述统计并生成：

```text
指标
impact score
diagnosis
top10
跨模型比较
```

这样修改诊断阈值或 impact 公式时无需重新运行模型推理。

## 13. 修改优先级

### P0

1. official event state 增加 `num_images`、像素 support、event loc confusion。
2. event CSV 输出 Recall、Precision 和六个误分流向。
3. GT support=0 时 destroyed GT 条件指标设为 NaN。
4. 保存每个 event 的原始四类 confusion。
5. 增加 `impact_score` 和 `destroyed_impact_score`。

### P1

1. 增加 destroyed FP 来源 count/share。
2. 增加自动 diagnosis code 和 support warning。
3. 新增跨模型 event summarizer。
4. 把 M03、M05、M07、M11 纳入同一模型配置清单。

### P2

1. 增加 leave-one-event-out 精确拖累量。
2. 输出 disaster-type 聚合和 event 内 bootstrap 置信区间。
3. 生成 heatmap/散点图：

```text
event × model destroyed_iou heatmap
destroyed_recall vs destroyed_precision
destroyed_to_intact vs damage_building_iou
support vs destroyed_iou
```

## 14. 最终建议

当前代码已经具备 event-wise damage confusion 的正确基础，但输出层过于精简，无法直接回答 destroyed 下降原因。

推荐最终分析框架：

```text
overall impact_score
    -> 找出拖累整体三类建筑性能的 event

destroyed_impact_score
    -> 找出真正拖累 destroyed 的 event

destroyed FN 分解
    -> background / intact / damaged

destroyed FP 来源
    -> background / intact / damaged

damage_building_iou
    -> 判断是定位失败还是损毁分类失败

support warning
    -> 避免对小 destroyed event 过度解释
```

优先对 M03_A4、M05_A4D、A4-DG、A4+DamageExistHead、M07_A4DG_b03 和 M11_A4HDC_a03 使用同一 test list、同一 checkpoint 规则和同一 event analysis 配置进行比较。
