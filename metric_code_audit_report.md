# BRIGHT 指标计算代码审计报告

审计日期：2026-06-22  
审计范围：`full_evaluate_unified.py`、`scripts/eval/`、`bda_benchmark/` 中与 IoU、Precision、Recall、F1、confusion matrix、定位及类别误分率有关的代码。  
类别约定：

```text
0 background
1 intact
2 damaged
3 destroyed
```

## 1. 审计结论摘要

核心结论如下：

1. `full_evaluate_unified.py --eval_protocol official` 中四类 IoU、Precision、Recall、F1 的 TP/FP/FN 方向和公式正确。
2. official confusion matrix 明确是 `confusion[gt, pred]`，没有发现正式评估入口使用 `confusion[pred, gt]`。
3. `damaged_to_intact`、`destroyed_to_intact`、`destroyed_to_damaged`、`destroyed_to_background` 均已实现，索引和 GT 分母正确。
4. `target_3class_miou` 正确排除 background；`official_style_4class_miou` 正确包含四类。
5. 当前 `loc_iou / loc_f1` 使用独立定位头 `output_loc`，不是由最终四分类 damage prediction 的 `pred > 0` 计算。这是指标语义不一致，不是 TP/FP/FN 公式错误。
6. 存在明确的 ignore-label bug：定位二值化在过滤 255 之前执行 `loc_label > 0`，因此 255 会被误计为 building。
7. `make_data_loader_roi_mask.py` 中由 `(clf_label > 0)` 生成 building mask，也会把 255 当作建筑。
8. 旧 `bda_benchmark/util_func/metrics.py` 的 `Pixel_F1_score()` 在 precision=recall=0 时可能产生 `0/0 -> NaN`。
9. 训练验证和旧 Evaluator 通过给分母加 epsilon 将空类 IoU/F1 记为 0；公式主体正确，但空类语义与 official 协议不统一。
10. 对当前本地 `standard_ML/test_set.txt` 的 878 张 target 标签做了只读扫描：实际像素只包含 0/1/2/3，未发现 255 或其他异常值。因此 255 bug 没有影响现有 standard official test 数值，但代码仍应修复以保证其他划分、padding 数据和未来数据安全。

## 2. Confusion matrix 定义

### 2.1 统一评估入口

位置：`full_evaluate_unified.py:294-299`

当前代码等价于：

```python
encoded = num_classes * target + pred
conf += bincount(encoded).reshape(num_classes, num_classes)
```

因此：

```text
confusion[row, column] = confusion[GT, Prediction]
```

例如：

```text
confusion[2, 1] = GT damaged、预测 intact
confusion[3, 1] = GT destroyed、预测 intact
confusion[3, 2] = GT destroyed、预测 damaged
confusion[3, 0] = GT destroyed、预测 background
```

结论：正确。

### 2.2 旧 Evaluator

位置：`bda_benchmark/util_func/metrics.py:91-100`

代码：

```python
label = num_class * gt_image + pre_image
```

调用形式普遍为：

```python
evaluator.add_batch(labels_clf, output_clf)
```

因此同样是：

```text
行 = GT
列 = Prediction
```

结论：正确。

### 2.3 BGCA 训练验证

位置：

- `bda_benchmark/script/standard_ML/train_DamageFormer_BGCA_Method05_06.py:484-505`
- `bda_benchmark/script/standard_ML/train_DamageFormer_BGCA_final.py:240-261`

虽然函数参数顺序写成：

```python
fast_hist(pred, label, ...)
```

内部编码为：

```python
num_classes * label + pred
```

因此矩阵仍为 `confusion[GT, Prediction]`，没有反转。

### 2.4 AdaptSeg 工具函数

位置：`bda_benchmark/model/AdaptSeg/utils/func.py:55-57`

```python
fast_hist(a, b, n)
return bincount(n * a + b).reshape(n, n)
```

该函数的参数名不明确，但项目中没有找到实际调用。按常见约定，`a` 应为 GT、`b` 应为 prediction。它目前不会影响现有评估结果，但建议改名为 `fast_hist(gt, pred, n)`，避免未来误用。

### 2.5 是否发现 `confusion[pred, gt]`

在当前扫描范围内，没有发现正在使用的评估代码构造 `confusion[pred, gt]`。

因此不存在因矩阵转置导致以下指标互换的问题：

```text
Recall 与 Precision 被交换
damaged_to_intact 被误算成 intact_to_damaged
destroyed_to_intact 被误算成 intact_to_destroyed
```

## 3. 每类 IoU

位置：

- `full_evaluate_unified.py:306-316`
- `full_evaluate_unified.py:446-467`

当前代码：

```python
tp = confusion[c, c]
fp = confusion[:, c].sum() - tp
fn = confusion[c, :].sum() - tp
iou = tp / (tp + fp + fn)
```

这与要求完全一致：

```text
TP = gt == c and pred == c
FP = gt != c and pred == c
FN = gt == c and pred != c
IoU = TP / (TP + FP + FN)
```

| 指标 | 当前来源 | 是否正确 |
|---|---|---|
| `background_iou` | `per_class[0]["iou"]` | 正确 |
| `intact_iou` | `per_class[1]["iou"]` | 正确 |
| `damaged_iou` | `per_class[2]["iou"]` | 正确 |
| `destroyed_iou` | `per_class[3]["iou"]` | 正确 |

official 协议在 `TP+FP+FN=0` 时返回 NaN；research 和旧 Evaluator 则返回 0 或近似 0。这是空类规则差异，不是 TP/FP/FN 方向错误。

## 4. Recall

位置：`full_evaluate_unified.py:449-455`

当前代码：

```python
support = confusion[c, :].sum()
recall = tp / support
```

由于行是 GT：

```text
support = TP + FN = count(gt == c)
Recall = TP / (TP + FN)
```

所以：

```text
damaged_recall
= confusion[2,2] / confusion[2,:].sum()

destroyed_recall
= confusion[3,3] / confusion[3,:].sum()
```

结论：正确。

注意：当 GT support 为 0 时，当前 `safe_div()` 返回 0，而不是 NaN。整体 test set 通常不受影响；event-wise 表格中建议改为 NaN。

## 5. Precision

位置：`full_evaluate_unified.py:449-453`

当前代码：

```python
fp = confusion[:, c].sum() - tp
precision = tp / (tp + fp)
```

由于列是 Prediction：

```text
TP + FP = count(pred == c)
Precision = TP / (TP + FP)
```

结论：正确。

当模型从不预测某类时，当前返回 0。

## 6. F1 与 mean F1

### 6.1 每类 F1

位置：

- `full_evaluate_unified.py:312-315`
- `full_evaluate_unified.py:453-455`

当前公式：

```python
f1 = 2 * precision * recall / (precision + recall)
```

结论：正确。

若 precision 和 recall 同时为 0，统一入口通过 `safe_div` 返回 0。

### 6.2 research 的 mean F1

位置：`full_evaluate_unified.py:332-337`

当前代码：

```python
valid = [1, 2, 3]
valid_mean_f1 = mean([
    intact_f1,
    damaged_f1,
    destroyed_f1,
])
```

结论：

- 公式是三类 F1 的算术平均。
- 不包含 background。
- 空类在 research 协议中按 0 进入平均。

### 6.3 official 是否输出 mean F1

official summary 输出四类各自的 F1，但没有单独输出：

```text
official_mean_f1
target_3class_mean_f1
```

若论文需要 mean F1，建议明确新增两个名称：

```python
official_4class_mean_f1 = mean(
    background_f1, intact_f1, damaged_f1, destroyed_f1
)

target_3class_mean_f1 = mean(
    intact_f1, damaged_f1, destroyed_f1
)
```

并采用与 IoU 相同的空类规则。

### 6.4 旧 DamageFormer 的 “clf f1”

位置：`bda_benchmark/script/standard_ML/train_DamageFormer.py:184-186` 和 `230-232`

代码不是算术 mean F1，而是：

```python
harmonic_mean_f1 = N / sum(1 / class_f1)
```

它是 intact/damaged/destroyed 三类 F1 的调和平均，不包含 background。

因此该旧日志中的 `clf f1` 不能与 `valid_mean_f1` 直接比较。

### 6.5 旧 Pixel_F1_score 的除零 bug

位置：`bda_benchmark/util_func/metrics.py:29-34`

```python
F1 = 2 * Rec * Pre / (Rec + Pre)
```

若 `Rec=0` 且 `Pre=0`，分母为 0，会产生 NaN。

建议修改为安全除法：

```python
den = Rec + Pre
F1 = 0.0 if den == 0 else 2 * Rec * Pre / den
```

该 bug 只影响使用旧 `Evaluator.Pixel_F1_score()` 的训练/验证脚本；统一 official evaluator 不受影响。

## 7. damaged_to_intact

位置：

- official：`full_evaluate_unified.py:504-508`
- research：`full_evaluate_unified.py:341-359`

当前公式：

```python
gt_damaged = confusion[2, :].sum()
damaged_to_intact = confusion[2, 1] / gt_damaged
```

等价于：

```text
count(gt=damaged, pred=intact) / count(gt=damaged)
```

结论：正确。

## 8. destroyed_to_intact

位置：

- official：`full_evaluate_unified.py:504-511`
- research：`full_evaluate_unified.py:342-361`

当前公式：

```python
gt_destroyed = confusion[3, :].sum()
destroyed_to_intact = confusion[3, 1] / gt_destroyed
```

结论：正确。

## 9. destroyed_to_damaged

位置：

- official：`full_evaluate_unified.py:512`
- research：`full_evaluate_unified.py:362`

当前公式：

```python
destroyed_to_damaged = confusion[3, 2] / confusion[3, :].sum()
```

结论：已经实现且正确，无需新增。

## 10. destroyed_to_background

位置：`full_evaluate_unified.py:510`

当前公式：

```python
destroyed_to_background = confusion[3, 0] / confusion[3, :].sum()
```

结论：official 协议已经实现且正确。

research 协议没有单独输出该名称，但已有：

```text
pred_background_on_gt_building_rate
```

后者是所有 GT 建筑类的合并比例，不等于单独的 `destroyed_to_background`。若继续维护 research 输出，建议也增加单类指标。

## 11. target_3class_miou

位置：`full_evaluate_unified.py:472-481`

当前代码：

```python
ious = [
    background_iou,
    intact_iou,
    damaged_iou,
    destroyed_iou,
]
target_3class_miou = nanmean(ious[1:4])
```

即：

```text
mean(intact_iou, damaged_iou, destroyed_iou)
```

不包含 background，结论正确。

空类使用 `nanmean`：只有 `TP+FP+FN=0` 的类会被跳过。

## 12. official_style_4class_miou

位置：`full_evaluate_unified.py:472-480`

当前代码：

```python
official_style_4class_miou = nanmean([
    background_iou,
    intact_iou,
    damaged_iou,
    destroyed_iou,
])
```

整体 test set 四类 union 均非零时，就是严格四类平均，结论正确。

event-wise 若某类 union 为 0，该类会被跳过，因此此时不是固定除以 4。建议在论文中明确 event-wise 的 present-class 规则。

## 13. loc_iou

### 13.1 当前实际计算

位置：

- `full_evaluate_unified.py:320-328`
- `full_evaluate_unified.py:1000-1008`

当前流程：

```python
loc_pred = argmax(output_loc)
gt_building = loc_label > 0
pred_building = loc_pred > 0
loc_iou = TP_building / (TP_building + FP_building + FN_building)
```

二分类 IoU 公式本身正确。

### 13.2 与要求的差异

用户要求的是：

```python
gt_building = damage_gt > 0
pred_building = damage_pred > 0
```

当前 prediction 来自独立定位头：

```text
argmax(output_loc)
```

而不是最终四分类头：

```text
argmax(output_dam)
```

因此当前 `loc_iou` 应准确命名为：

```text
loc_head_iou
```

若要衡量最终四分类输出的建筑定位，应新增：

```python
valid = damage_gt != 255
gt_building = (damage_gt > 0) & valid
pred_building = damage_pred > 0
damage_building_iou = binary_iou(gt_building, pred_building, valid)
```

## 14. loc_f1

当前定位 confusion 的 class 1 指标为：

```text
loc_precision = TP_building / (TP_building + FP_building)
loc_recall    = TP_building / (TP_building + FN_building)
loc_f1        = 2PR / (P+R)
```

位置：`full_evaluate_unified.py:306-328`

公式正确，但 prediction 同样来自独立 `output_loc`，不是最终四分类结果。

结论：

- 数学公式正确。
- 指标来源与用户要求不一致。
- 建议区分 `loc_head_f1` 与 `damage_building_f1`。

## 15. 255 / ignore_index 处理

### 15.0 当前 official test 标签实测

对：

```text
bda_benchmark/dataset/splitname/standard_ML/test_set.txt
```

对应的 878 张 `target/*_building_damage.tif` 做了只读唯一值统计：

```text
0: 790001718 pixels
1: 108561582 pixels
2:  13541338 pixels
3:   8545090 pixels
255: 0 pixels
其他值: 0 pixels
缺失文件: 0
```

因此：

- 当前 standard official test 的已有 loc/damage 数值没有被 255 污染。
- 以下 255 问题属于确定的代码 bug，但在当前这套 test 标签上没有触发。
- train crop padding、其他数据划分或未来带 ignore 区域的数据仍可能触发。

### 15.1 damage confusion：正确过滤

位置：`full_evaluate_unified.py:294-299`

```python
mask = (
    (target >= 0) &
    (target < num_classes) &
    (pred >= 0) &
    (pred < num_classes)
)
```

因此 damage target=255 会被过滤，不进入：

```text
IoU
Precision
Recall
F1
damaged/destroyed 误分率
```

结论：正确。

### 15.2 loc 指标：存在 bug

位置：

- `full_evaluate_unified.py:837-840`
- `full_evaluate_unified.py:1005-1007`
- `scripts/eval/eval_object_voting.py:188-190`

当前代码先执行：

```python
loc_gt_bin = (loc_np > 0).astype(np.int64)
```

由于：

```text
255 > 0
```

ignore 像素会变成 building=1，随后传入二分类 confusion 时已经无法识别它原本是 255。

建议修改逻辑为：

```python
valid = loc_np != 255
loc_gt_bin = (loc_np > 0).astype(np.int64)
loc_pr_bin = (loc_pred_np > 0).astype(np.int64)

update_confusion(
    loc_conf,
    loc_gt_bin[valid],
    loc_pr_bin[valid],
    2,
)
```

若使用 damage label 派生定位，则应使用：

```python
valid = dam_np != 255
gt_building = dam_np > 0
pred_building = dam_pred_np > 0
```

### 15.3 research 的 GT building mask：存在同类 bug

位置：`full_evaluate_unified.py:841-842`

```python
building_mask = dam_np > 0
```

target=255 同样满足 `>0`。虽然后续 `update_confusion()` 会因为 target 不在 0..3 而过滤这些位置，所以当前 damage confusion 数值最终不会纳入 255，但该 mask 的语义仍不严谨。

建议写成：

```python
building_mask = (dam_np > 0) & (dam_np != 255)
```

### 15.4 ROI mask 数据加载器：存在 bug

位置：

- `bda_benchmark/dataset/make_data_loader_roi_mask.py:210`
- `bda_benchmark/dataset/make_data_loader_roi_mask.py:166`

当前：

```python
bmask = (clf_label > 0).astype(np.float32)
```

会把 255 变成 building mask=1。

建议：

```python
bmask = ((clf_label > 0) & (clf_label != 255)).astype(np.float32)
```

注意：该文件的 `loc_label` 在 `:234` 保留了 255，这是正确的；问题只在 bmask 二值化。

### 15.5 旧 Evaluator

`bda_benchmark/util_func/metrics.py:92` 使用：

```python
(gt_image >= 0) & (gt_image < num_class)
```

因此 GT=255 会被过滤，处理正确。

BGCA 训练验证的 `fast_hist()` 也显式执行：

```python
mask = label != ignore_index
```

处理正确。

## 16. 发现的问题及严重程度

| 问题 | 位置 | 严重程度 | 是否影响核心 official damage 指标 |
|---|---|---:|---|
| 255 在 loc 二值化时被当成 building | `full_evaluate_unified.py:837-840,1005-1007` | 高 | 不影响 damage IoU；影响 loc 指标 |
| object-voting 的 loc/GT building mask 把 255 当建筑 | `eval_object_voting.py:188-198` | 高 | 可能影响 voting 结果与 loc 指标 |
| ROI bmask 把 255 当建筑 | `make_data_loader_roi_mask.py:166,210` | 高 | 可能影响使用该 loader 的训练/推理 |
| `Pixel_F1_score()` 可出现 0/0 NaN | `metrics.py:29-34` | 中 | 不影响统一 official evaluator |
| loc 指标来自独立定位头而非四分类输出 | `full_evaluate_unified.py:992-1008` | 高（语义） | 不影响 damage 指标 |
| official recall/F1 在 GT support=0 时记 0 | `full_evaluate_unified.py:302-303,453-455` | 中 | 影响 event-wise 空类解释 |
| research/训练空类 IoU、F1 强制为 0 | 多处 epsilon/safe_div | 中 | 影响 event-wise 和 checkpoint 排名 |
| 旧 `clf f1` 是调和平均，不是 mean F1 | `train_DamageFormer.py:186,232` | 中 | 影响跨脚本比较 |
| AdaptSeg `fast_hist(a,b)` 参数名含糊 | `AdaptSeg/utils/func.py:55-57` | 低 | 当前未使用 |

## 17. 是否影响已有实验结果

### 17.1 不需要因核心公式重新评估的指标

如果已有结果满足：

```text
使用 full_evaluate_unified.py
使用 --eval_protocol official
damage GT 只包含 0/1/2/3 或 255
```

则以下已有整体指标公式正确：

```text
background_iou
intact_iou
damaged_iou
destroyed_iou
damaged_precision / recall / f1
destroyed_precision / recall / f1
target_3class_miou
official_style_4class_miou
damaged_to_intact
destroyed_to_intact
destroyed_to_damaged
destroyed_to_background
```

这些指标不需要因为 TP/FP/FN 或 confusion 方向问题而重跑。

### 17.2 需要检查或重跑的结果

以下结果建议重新评估：

1. 所有需要把 loc 定义改为最终四分类 `damage_pred > 0` 的模型。
2. 所有论文 event-wise 指标：若空类规则改为 `GT support=0 -> NaN`，应统一重新汇总。
3. 使用旧 `Pixel_F1_score()` 并出现 NaN 或空 building batch/event 的结果。
4. 将旧 harmonic `clf f1` 当成 arithmetic mean F1 比较的表格，应重新计算和命名。
5. 使用其他带 255 的 test/val 划分计算 loc 指标或 object voting 的模型。
6. 使用 `make_data_loader_roi_mask.py` 生成 ROI mask，并且训练/验证 crop padding 或原标签中实际含 255 的实验。

当前 standard official test 不含 255，所以仅修复 ignore mask 而不改变 loc 定义时，现有 standard test 的 loc 和 object-voting 结果理论上不变。

## 18. 修改后需要重新评估哪些模型

若只修复指标代码而不改模型权重：

### 必须重评

如果决定把论文中的 loc 定义改为最终四分类输出：

```text
所有论文最终对比模型的 loc_iou / loc_f1
所有 event-wise 指标表中的模型
```

包括但不限于：

```text
DamageFormer baseline
A4
A4-D
A4-IRC
A4-DG
A4-DG-beta0.2
A4-DG-beta0.3
A4-DG-beta0.5
A4-ExistHead
A4-DG-ExistHead
A4-DG-ExistHead-IC
A4-HDC-alpha0.3
```

如果只修复 255 mask 而继续使用独立定位头，则当前 standard official test 因实测不存在 255，不要求全部重评；应对修改前后做一次代表性回归验证即可。

object-voting 只有在以下情况需要全部重评：

```text
改变 building source 定义
评估数据包含 255
同时修正 voting 策略
```

### 不必因本次公式审计重评

如果只使用已有 official overall damage confusion 生成的以下指标，则无需重跑推理：

```text
四类 IoU
三类 target mIoU
damaged/destroyed recall
damaged/destroyed to-intact/to-damaged/to-background
```

若保存了完整 `confusion_4class_official.csv`，这些指标可以直接从 confusion matrix 重新计算，无需再次前向推理。

### 可能需要重新训练

仅当实验使用了 `make_data_loader_roi_mask.py`，且训练 crop/padding 中确实产生 255 并通过错误 bmask 进入模型时，修复后才需要考虑重新训练。

普通 `make_data_loader.py` + unified official evaluator 的模型不需要因为本次核心指标公式审计而重新训练。

## 19. 建议修改顺序

### P0

1. 在所有 loc confusion 更新前保留并应用 `valid = label != 255`。
2. 明确区分：

```text
loc_head_iou / loc_head_f1
damage_building_iou / damage_building_f1
```

3. 修复 ROI bmask：

```python
(label > 0) & (label != 255)
```

4. 修复旧 `Pixel_F1_score()` 的 0/0。

### P1

1. official event-wise 指标在 GT support=0 时将 recall、F1 和类别误分率记为 NaN。
2. 输出每类 `support`，避免把空类的 0 当成模型失败。
3. 新增明确命名的 `target_3class_mean_f1`，不要与旧 harmonic F1 混用。

### P2

1. 将所有 confusion helper 的参数统一命名为 `target, pred` 或 `gt, pred`。
2. 为 confusion 和指标函数增加一个人工可验证的小矩阵单元测试。

建议测试矩阵至少覆盖：

```text
非对称 confusion，验证行列方向
GT damaged -> pred intact
GT destroyed -> pred intact/damaged/background
空 destroyed 类
包含 255 ignore
precision=recall=0
```

## 20. 最终判断

BRIGHT 当前正式 official damage 指标的核心计算没有发现 TP/FP/FN、Recall/Precision 或 confusion 方向 bug。最值得修复的不是四类 IoU 本身，而是：

```text
255 在 loc/building mask 中被误认为建筑
loc 指标来源与最终四分类输出不一致
旧 F1 的除零问题
不同协议对空类和 mean F1 的定义不统一
```

修复这些问题后，建议以统一 official confusion 为唯一 damage 指标数据源，并以：

```text
target_3class_miou
damaged_iou
destroyed_iou
damaged_recall
destroyed_recall
damaged_to_intact
destroyed_to_intact
destroyed_to_damaged
destroyed_to_background
```

作为建筑损毁研究的核心指标组。
