# BRIGHT 评估标准检查报告

检查日期：2026-06-22  
检查范围：`full_evaluate_unified.py`、`scripts/eval/`、`bda_benchmark/` 中的评估、推理、指标及相关训练验证代码。  
研究目标：灾前 RGB/optical + 灾后 SAR，输出 `0 background / 1 intact / 2 damaged / 3 destroyed`，重点关注建筑损毁三类而非背景类。

## 一、总体结论

当前最适合论文统一使用的是：

```text
full_evaluate_unified.py --eval_protocol official
```

该协议中的以下指标定义正确，且与研究目标一致：

```text
official_style_4class_miou
target_3class_miou
damaged_iou
destroyed_iou
damaged_recall
destroyed_recall
damaged_to_intact
destroyed_to_intact
```

但仍存在以下重要问题：

1. `loc_iou / loc_f1` 不是由四分类结果的 `target > 0` 和 `pred > 0` 计算，而是由独立定位头 `output_loc` 计算。
2. event-wise 空类处理只在“GT 和预测都没有该类”时返回 NaN；若 GT 没有但模型产生该类假阳性，则 IoU 为 0。若论文要求“只要 GT support=0 就记 NaN”，当前代码不完全符合。
3. `research` 协议与 `official` 协议统计域不同，不能混合比较。`research` 仅在 GT 建筑区域统计 damage confusion，并把空类强制记为 0。
4. 旧 `paper` 协议和 `bda_benchmark/util_func/metrics.py` 会把空类 IoU 近似记为 0，不适合 event-wise 论文表格。
5. 若干旧 checkpoint 评估 shell 指向项目中不存在的 `full_evaluate_BGCA*.py`，目前不能作为可靠统一入口。
6. checkpoint 用 `strict=False` 加载，虽然打印 missing/unexpected keys，但不会因结构不匹配而停止；且 evaluator 没有自动读取 checkpoint 内保存的训练参数。
7. 训练阶段默认只用前 50 个 validation batches 选择 best checkpoint，可能导致 `best_valid_miou / best_damaged_iou / best_destroyed_iou` 对完整验证集不稳定。

## 二、类别定义检查

### 2.1 统一入口

`full_evaluate_unified.py:59-65` 明确定义：

```text
0 background
1 intact
2 damaged
3 destroyed
```

`scripts/eval/eval_object_voting.py:34` 使用相同顺序。

模型损失和辅助模块也使用相同语义：

- `bda_benchmark/model/modules/damage_ordinal_losses.py:6-10`
- `bda_benchmark/model/modules/bgca_losses.py:189-192`
- `bda_benchmark/model/DamageFormer_BGCA.py:670-675`

未发现 damaged 和 destroyed 的 ID 被写反。

### 2.2 数据集定位标签

`bda_benchmark/dataset/make_data_loader.py` 通过：

```python
loc_label = clf_label.copy()
loc_label[loc_label == 2] = 1
loc_label[loc_label == 3] = 1
```

生成二分类建筑标签。因此对合法四分类标签而言：

```text
loc_label == (damage_target > 0)
```

intact、damaged、destroyed 都属于 building，没有排除 destroyed。

### 2.3 语义注释风险

`bda_benchmark/script/standard_ML/infer_DamageFormer.py:46-49` 的颜色注释仍使用 “No damage / Minor damage / Major damage / Destroyed”等旧表达，其中 class 0 的注释尤其容易把 background 与 no-damage 混淆。

这不改变数值计算，但建议论文和输出文件统一只使用：

```text
background / intact / damaged / destroyed
```

### 2.4 255 ignore

统一入口的四分类 confusion 在 `full_evaluate_unified.py:294-299` 通过合法范围：

```python
(target >= 0) & (target < num_classes)
```

过滤标签，因此 `255` 会被排除，damage 指标不会把 255 当作类别。

但定位计算先执行：

```python
loc_gt_bin = (loc_np > 0)
```

若测试标签中存在 255，它会先被转换成 building=1，随后无法再被过滤。标准 test TIFF 如果完全没有 255，则实际结果不受影响；从鲁棒性和定义上看，这仍是一个应修正的问题。

## 三、official_style_4class_miou

`full_evaluate_unified.py:472-481` 的实际定义是：

```text
IoU_0 = background IoU
IoU_1 = intact IoU
IoU_2 = damaged IoU
IoU_3 = destroyed IoU

official_style_4class_miou = nanmean(IoU_0, IoU_1, IoU_2, IoU_3)
```

结论：

- 在整体 test set 四类都有有效 union 时，它就是严格的四类平均 IoU，定义正确。
- 在 event-wise 某类 union 为 0 时，使用 `nanmean` 跳过该类，此时数值实际上是“该 event 中有效类的平均 IoU”，不再是严格固定四类平均。

因此建议论文中明确区分：

```text
overall official_style_4class_miou：固定四类总体指标
event-wise mIoU：present-class nanmean，或另行规定固定四类策略
```

## 四、target_3class_miou

`full_evaluate_unified.py:481`：

```python
target_3class_miou = official_nanmean(ious[1:4])
```

即：

```text
mean IoU over intact, damaged, destroyed
```

background 明确被排除，定义与研究目标一致。

该指标可以作为论文主指标。整体 test set 上可信；event-wise 时仍需注明空类跳过规则。

## 五、loc_iou / loc_f1

当前 `official` 和 `research` 协议实际计算为：

```text
GT building = loc_label > 0
Pred building = argmax(output_loc) > 0
```

不是用户提出的：

```text
GT building = damage_target > 0
Pred building = damage_pred > 0
```

其中 GT 两者在合法标签上等价，但 Pred 不等价：

- 当前指标衡量独立定位头 `output_loc`。
- 用户定义衡量最终四分类 damage 输出自身的建筑/背景分割能力。

因此当前 `loc_iou / loc_f1` 本身数学上合理，但指标语义应写成：

```text
localization-head IoU / F1
```

不能宣称它是从最终四分类预测折叠得到的 building IoU/F1。

建议论文同时保留或至少明确选择一种：

```text
loc_head_iou：由 output_loc 计算
damage_building_iou：由 damage_target>0 与 damage_pred>0 计算
```

如果论文只允许一个定位指标，更建议报告最终输出派生的 `damage_building_iou`，因为它与最终四分类结果直接一致。

## 六、damaged_to_intact / destroyed_to_intact

`full_evaluate_unified.py:504-512`：

```text
damaged_to_intact
= confusion[2,1] / sum(confusion[2,:])
= count(gt=2,pred=1) / count(gt=2)

destroyed_to_intact
= confusion[3,1] / sum(confusion[3,:])
= count(gt=3,pred=1) / count(gt=3)
```

分母是对应 GT 类别像素总数，不是预测像素数，定义正确。

`research` 协议中的同名 rate 也使用 GT 行作为分母，但其 confusion 仅统计 GT 建筑区域；对这两个特定 rate 而言分子和分母仍正确。

空 GT 类别时 `safe_div` 返回 0。整体 test set 通常可接受，但 event-wise 应更推荐 NaN。

## 七、confusion matrix 方向

`full_evaluate_unified.py:298-299` 使用：

```python
encoded = num_classes * target + pred
reshape(num_classes, num_classes)
```

因此方向明确为：

```text
行 = GT
列 = Prediction
```

CSV 表头也写为 `gt\pred`。

`bda_benchmark/util_func/metrics.py:91-96` 和 BGCA 训练验证的 `fast_hist` 也是相同方向。

因此：

```text
conf[2,1] = damaged -> intact
conf[3,1] = destroyed -> intact
```

当前没有行列反转错误。

## 八、空类处理

### 8.1 official 协议

`official_per_class_from_conf()` 的规则：

```text
若 tp + fp + fn > 0：正常计算 IoU
若 tp + fp + fn == 0：IoU = NaN
```

这意味着：

- GT 无该类、预测也无该类：NaN。
- GT 无该类、但预测产生该类假阳性：IoU=0。
- GT 有该类但完全预测错：IoU=0。

这是标准 IoU 的 union-based 定义，但不完全等同于“GT support=0 就记 NaN”。

对于本研究的 event-wise 表格，建议采用：

```text
若 support = sum(conf[class,:]) == 0：
该 event 的该类 IoU、recall、误分率记 NaN
```

假阳性仍可通过 precision、FP 数或背景/其他类 IoU 体现。

### 8.2 research 协议

`class_metrics_from_conf()` 使用 `safe_div(...)=0`，空类直接得到 0。  
因此 `research` 的 event-wise `valid_miou` 会被不存在的 damaged/destroyed 人为拉低，不适合作为论文 event-wise 主表。

### 8.3 paper 协议和旧 Evaluator

`summarize_paper_conf()`、`bda_benchmark/util_func/metrics.py` 在分母中加入 epsilon，空类结果接近或等于 0。

此外 `evaluate_paper()` 预先建立全部 event 的零 confusion；没有样本或匹配失败的 event 也会输出全 0 指标。这些结果不应直接进入论文均值。

### 8.4 训练阶段 checkpoint 选择

BGCA 训练验证中的 `compute_iou_from_hist()` 同样通过 epsilon 把空类记 0。并且：

```text
--val_max_batches 默认 50
```

即 best checkpoint 默认基于验证集前 50 个 batch，而非完整 val set。若某些灾种或 destroyed 样本没有出现在这 50 个 batch 中，checkpoint 排名可能偏离最终研究目标。

建议正式训练/重选 checkpoint 时使用：

```text
--val_max_batches 0
```

或在训练结束后对所有 checkpoint 用完整 val set 统一重排，再只在 test set 做一次最终报告。

## 九、插值检查

统一评估入口只对 logits 做尺寸对齐：

```python
F.interpolate(logits, ..., mode="bilinear")
```

标签没有经过 bilinear，处理正确。

数据增强中的标签缩放使用：

```text
cv2.INTER_NEAREST
```

辅助损失中离散 label/mask 的缩放也主要使用 `nearest`。

未发现评估阶段对类别标签使用 bilinear 的问题。

## 十、测试集一致性

当前 `scripts/eval/` 中可用的 official 脚本统一指向：

```text
bda_benchmark/dataset/splitname/standard_ML/test_set.txt
```

未发现这些 official shell 把 val set 当 test set。

但有三个管理风险：

1. `full_evaluate_unified.py` 的 `--data_list_path` 是任意路径，不验证是否为 official test list。
2. `eval_all_official_checkpoints.sh` 允许通过环境变量 `TEST_LIST` 覆盖，结果目录中没有强制记录列表文件 hash。
3. 旧 cross-event/DFC25 脚本使用各自数据划分，不能与 standard_ML official 结果混合比较。

建议正式结果同时保存：

```text
data_list_path
test list 文件 SHA256
样本数量
dataset root
eval protocol
```

## 十一、checkpoint 与模型参数

### 11.1 已确认一致

A4-DG-beta0.3：

```text
训练：--mddm_variant dual_scale_gate --dg_large_beta 0.3
评估：--mddm_variant dual_scale_gate --dg_large_beta 0.3
```

对应：

- `scripts/train/run_A4_DG_beta03_120k.sh`
- `scripts/eval/eval_A4_DG_beta03_official.sh`
- `scripts/train/run_M07_A4DG_b03_120k.sh`
- `scripts/eval/eval_M07_A4DG_b03_official.sh`

A4-HDC-alpha0.3：

```text
训练：--use_hdc --hdc_mode blend --hdc_alpha 0.3
评估：--use_hdc --hdc_mode blend --hdc_alpha 0.3
```

对应：

- `scripts/train/run_M11_A4HDC_a03_120k.sh`
- `scripts/eval/eval_M11_A4HDC_a03_official.sh`

### 11.2 可能不一致或不够安全

1. `eval_all_official_checkpoints.sh` 中通用 `A4_DG` 没有显式传 `--dg_large_beta`，因此使用默认值 1.0。它只适合训练时 beta 也是 1.0 的 A4-DG，不能用于 beta0.2/0.3/0.5 checkpoint。
2. `eval_official_style_models.sh` 的 A4_DG 同样没有显式 beta，只能安全用于 beta=1.0 checkpoint。
3. `eval_A4_DG_120k.sh` 没有显式 beta；其配套训练脚本也没有显式 beta，二者都落到默认 1.0，所以彼此一致，但命名没有体现 beta。
4. `load_checkpoint()` 使用 `strict=False`。错误架构可能只打印 missing/unexpected keys 后继续评估，产生看似正常但不可信的结果。
5. 训练 checkpoint 中已保存 `args`，但统一 evaluator 不会读取并核对这些 args；参数一致性完全依赖 shell 人工维护。
6. `eval_M06A/C/D`、`eval_M06_all_checkpoints.sh`、`eval_M08_checkpoints.sh`、`eval_M10_checkpoints.sh` 会查找：

```text
tools/full_evaluate_BGCA_Method05_06.py
full_evaluate_BGCA_Method05_06.py
tools/full_evaluate_BGCA.py
full_evaluate_BGCA.py
```

当前项目中这些文件均不存在。这些脚本属于失效/遗留入口，不应再用于正式比较。

## 十二、哪些指标可信

### 可直接用于整体 official test set

```text
official_style_4class_miou
target_3class_miou
background_iou
intact_iou
damaged_iou
destroyed_iou
damaged_precision / recall / f1
destroyed_precision / recall / f1
damaged_to_intact
destroyed_to_intact
```

前提：

```text
使用 --eval_protocol official
使用固定 standard_ML/test_set.txt
模型构造参数与 checkpoint 一致
missing/unexpected keys 为 0，或经过人工解释
```

### 有条件可信

```text
loc_iou / loc_f1
```

它们可信地反映独立 localization head，但不等于最终 damage prediction 折叠后的 building 指标。

### 不建议作为论文统一主指标

```text
research valid_miou
research final_miou
旧 Evaluator 的 event-wise mIoU
paper 协议中空 event/空类的 0 值
不同协议之间直接比较的 mIoU
```

特别是：

```text
final_miou = 0.5 * loc_iou + 0.5 * valid_miou
```

这是项目自定义复合指标，不是标准四分类指标，也会把定位头和损毁分类混在一起，不建议作为论文主要结论。

## 十三、建议论文最终指标

建议主表按以下优先级报告：

1. `target_3class_miou`：核心主指标。
2. `damaged_iou`、`destroyed_iou`：核心类别指标。
3. `damaged_recall`、`destroyed_recall`：避免只看 IoU 掩盖漏检。
4. `damaged_to_intact`、`destroyed_to_intact`：关键低估损毁错误。
5. `official_style_4class_miou`：与完整四分类/官方风格结果对照。
6. `intact_iou`：说明三类建筑平均的组成。
7. 建筑定位指标：建议新增/改名为最终四分类派生的 `damage_building_iou/f1`；若保留独立头指标，命名为 `loc_head_iou/f1`。

建议 event-wise 报告：

```text
target_3class_miou
damaged_iou
destroyed_iou
damaged_recall
destroyed_recall
damaged_to_intact
destroyed_to_intact
support_damaged
support_destroyed
```

并规定 `support=0` 时对应类别指标为 NaN，不参与 event macro average。

## 十四、建议 official evaluation 命令模板

### 14.1 A4-DG-beta0.3

```bash
python full_evaluate_unified.py \
  --eval_protocol official \
  --model_type DamageFormer_BGCA \
  --model_path /path/to/checkpoint.pth \
  --dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --data_list_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/test_set.txt \
  --image_size 640 \
  --batch_size 1 \
  --num_workers 2 \
  --save_dir /path/to/output \
  --roi_mode dilated_soft \
  --roi_fusion multiplicative \
  --roi_beta 1.0 \
  --roi_alpha 0.1 \
  --roi_dilate 5 \
  --use_mddm \
  --mddm_variant dual_scale_gate \
  --mddm_gamma_init 0.0 \
  --mddm_use_roi \
  --dg_gate_reduction 4 \
  --dg_gamma_init 0.0 \
  --dg_gamma_mode sigmoid01 \
  --dg_large_beta 0.3
```

### 14.2 A4-HDC-alpha0.3

```bash
python full_evaluate_unified.py \
  --eval_protocol official \
  --model_type DamageFormer_BGCA \
  --model_path /path/to/checkpoint.pth \
  --dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --data_list_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/test_set.txt \
  --image_size 640 \
  --batch_size 1 \
  --num_workers 2 \
  --save_dir /path/to/output \
  --roi_mode dilated_soft \
  --roi_fusion multiplicative \
  --roi_beta 1.0 \
  --roi_alpha 0.1 \
  --roi_dilate 5 \
  --use_mddm \
  --mddm_variant base \
  --mddm_gamma_init 0.0 \
  --mddm_use_roi \
  --use_hdc \
  --hdc_mode blend \
  --hdc_alpha 0.3 \
  --lambda_exist 0.1 \
  --lambda_severity 0.2 \
  --lambda_hdc 0.5 \
  --exist_loss_type bce
```

## 十五、建议修改优先级

### P0：论文结果前必须处理

1. 正式结果统一使用 `--eval_protocol official`。
2. 禁止 beta0.2/0.3/0.5 checkpoint 走未显式设置 beta 的通用 A4-DG shell。
3. 检查每次加载的 missing/unexpected keys；正式评估应要求关键结构完全匹配。
4. event-wise 指标按 GT support 决定 NaN，并输出 support。
5. 明确 `loc_iou/f1` 是独立定位头，或新增最终四分类派生的 building 指标。

### P1：强烈建议

1. evaluator 自动读取 checkpoint 中的 `args` 并与命令行参数逐项核对。
2. 保存 test list SHA256、样本数和完整模型配置。
3. 全验证集重排 checkpoint，避免默认前 50 batch 选择偏差。
4. 删除或迁移指向不存在 evaluator 的旧 shell，避免误运行。

### P2：整理性修改

1. 统一旧脚本中的类别文字和颜色注释。
2. 将 `research final_miou` 明确标记为 legacy/custom composite metric。
3. event macro average 同时输出有效 event 数，避免不同模型/类别参与均值的数量不一致。

## 最终判断

项目的核心 official 四分类 confusion、三类建筑 mIoU、damaged/destroyed IoU、recall，以及 damaged/destroyed→intact 的方向和分母均正确。当前主要风险不是核心公式写反，而是：

```text
评估协议混用
定位指标语义不一致
event-wise 空类规则不统一
checkpoint 参数依赖人工维护
strict=False 掩盖结构不匹配
旧评估 shell 已失效
best checkpoint 只基于部分 validation batches
```

在解决上述 P0 项后，`target_3class_miou + damaged_iou + destroyed_iou + damaged/destroyed recall + 两个 to_intact rate` 能较好地对齐本研究“提升建筑区域损毁识别，尤其减少严重损毁被判为 intact”的目标。
