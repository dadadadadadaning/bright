# BRIGHT Official 评估代码修改完整自检报告

检查日期：2026-06-22  
项目工作区：`D:\software\dowmload\BRIGHT-master`  
服务器目标路径：`/root/shared-nvme/BRIGHT-master`

## 1. 检查结论

本次 official 评估扩展的代码级回归检查通过：

```text
Python 语法检查：通过
parser 参数检查：通过
六模型配置参数解析：通过
旧 official 指标公式回归：通过
confusion[GT, Prediction]：通过
damaged/destroyed 误分率分母：通过
255 ignore：通过
damage_building 指标：通过
event ID 提取：通过
event-wise 空类规则：通过
event-wise 输出 schema：通过
跨模型 summarizer CLI 与排序：通过
```

当前环境没有 PyTorch 和 Bash，因此未执行真实 checkpoint 前向评估，也不能直接运行 `.sh`。工作区中没有找到已有 `summary_official.json`，因此没有可用的修改前真实数值文件进行逐值对比。

这不影响人工 confusion、公式、schema 和 summarizer 的验证，但服务器端仍需用一个真实 checkpoint 做最终 smoke test。

## 2. 修改和检查的文件

主要文件：

- `full_evaluate_unified.py`
- `scripts/eval/run_official_eventwise_all_models.sh`
- `scripts/eval/summarize_eventwise_official_results.py`
- `eval_analysis/model_eval_config.json`

配套更新：

- `scripts/eval/eval_object_voting.py`
- `scripts/eval/summarize_official_results.py`
- `scripts/eval/summarize_all_official_results.py`

新增本报告：

- `eval_analysis/eval_code_modification_check_report.md`

## 3. 基础语法检查

执行：

```text
python -m py_compile full_evaluate_unified.py
python -m py_compile scripts/eval/summarize_eventwise_official_results.py
```

结果：通过，无语法错误。

同时也已检查配套 Python 文件：

```text
scripts/eval/eval_object_voting.py
scripts/eval/summarize_official_results.py
scripts/eval/summarize_all_official_results.py
```

## 4. Parser 参数检查

由于当前 Python 环境没有 PyTorch，无法直接执行完整模块的普通 `--help`；自检通过注入最小 import stub 后调用真实 `build_arg_parser().format_help()`。

已确认存在：

```text
--eval_protocol
--mddm_variant
--dg_large_beta
--use_damage_exist_head
--use_hdc
--hdc_alpha
```

六模型配置逐一通过真实 argparse 解析：

| Model | mddm_variant | dg_large_beta | use_hdc | hdc_alpha | use_damage_exist_head |
|---|---|---:|---|---:|---|
| M03_A4 | base | 1.0 | False | 0.3 default | False |
| M05_A4D | large_scale | 1.0 | False | 0.3 default | False |
| A4-DG | dual_scale_gate | 1.0 | False | 0.3 default | False |
| A4_DamageExistHead | base | 1.0 | False | 0.3 default | True |
| M07_A4DG_b03 | dual_scale_gate | 0.3 | False | 0.3 default | False |
| M11_A4HDC_a03 | base | 1.0 | True | 0.3 | False |

## 5. 原有 official 指标兼容性

没有修改前程序备份和真实 summary 可比较，因此使用非对称人工 confusion，将修改后函数结果与旧公式独立计算结果逐项比较。

以下旧字段仍存在且非空：

```text
official_style_4class_miou
target_3class_miou
background_iou
intact_iou
damaged_iou
destroyed_iou
damaged_recall
destroyed_recall
damaged_to_intact
destroyed_to_intact
```

验证结果：

1. 四类 IoU 仍为 `TP/(TP+FP+FN)`。
2. `official_style_4class_miou` 仍为四类 IoU 的 `nanmean`。
3. `target_3class_miou` 仍只使用 `ious[1:4]`，不含 background。
4. `damage_building_iou` 是独立新增字段，没有覆盖 `loc_iou`。
5. `loc_iou == loc_head_iou`。
6. `loc_f1 == loc_head_f1`。

## 6. Confusion matrix 人工测试

测试输入：

```python
gt = np.array([
    [0, 1, 2, 3],
    [3, 2, 1, 0],
])

pred = np.array([
    [0, 1, 1, 1],
    [2, 2, 3, 0],
])
```

实际 confusion：

```text
[[2, 0, 0, 0],
 [0, 1, 0, 1],
 [0, 1, 1, 0],
 [0, 1, 1, 0]]
```

已确认：

```text
conf[2,1] = 1
conf[3,1] = 1
conf[3,2] = 1
conf[1,3] = 1
```

因此矩阵方向仍为：

```text
行 = GT
列 = Prediction
```

## 7. 误分率分母检查

基于上述 confusion：

```text
damaged_to_intact = conf[2,1] / conf[2,:].sum() = 1/2
destroyed_to_intact = conf[3,1] / conf[3,:].sum() = 1/2
destroyed_to_damaged = conf[3,2] / conf[3,:].sum() = 1/2
destroyed_to_background = conf[3,0] / conf[3,:].sum() = 0
```

测试通过。代码使用 GT 行和作为分母，没有使用预测列和。

## 8. 255 Ignore 测试

测试：

```python
gt = np.array([[255, 0, 1, 2, 3]])
pred = np.array([[1, 0, 1, 2, 3]])
```

结果：

```text
四分类 confusion 总像素数 = 4
loc head confusion 总像素数 = 4
damage building confusion 总像素数 = 4
```

255 位置未进入任何 confusion，也未被视为 building。

修复使用：

```python
valid = target != 255
gt_building = target in {1,2,3}
```

## 9. Damage Building 指标测试

测试：

```python
gt = [[0, 1, 2, 3]]
damage_pred = [[0, 1, 0, 3]]
```

实际二分类 confusion：

```text
[[1, 0],
 [1, 2]]
```

结果：

```text
TP = 2
FN = 1
FP = 0
damage_building_iou = 2/3
damage_building_recall = 2/3
damage_building_precision = 1
damage_building_f1 = 0.8
```

全部符合预期。

## 10. Event ID 提取测试

以下映射全部通过：

```text
la_palma-volcano_00000123 -> la_palma-volcano
turkey-earthquake_00001082 -> turkey-earthquake
ukraine-conflict_00000468 -> ukraine-conflict
bata-explosion_00000007 -> bata-explosion
```

同时已用完整 `standard_ML/test_set.txt` 检查：

```text
878 个样本
14 个 event
unknown = 0
```

没有返回完整 sample ID。

## 11. Official Event-wise 输出检查

通过人工 event state 实际调用写出函数，确认生成：

```text
summary_official.json
event_metrics_official.csv
event_confusion_4class.csv
event_destroyed_diagnosis.csv
analysis_config.json
```

`event_metrics_official.csv` schema 已确认包含：

```text
num_images
num_destroyed_pixels
destroyed_precision
destroyed_recall
destroyed_to_background
destroyed_to_intact
destroyed_to_damaged
loc_head_iou
damage_building_iou
impact_score
destroyed_impact_score
```

正式模型评估时还会保留：

```text
per_class_metrics_official.csv
confusion_4class_official.csv
official_metrics.txt
```

## 12. Event-wise 空类处理

构造 `num_destroyed_pixels=0` 的 event 后，实际结果：

```text
destroyed_iou = NaN
destroyed_recall = NaN
destroyed_to_intact = NaN
destroyed_to_damaged = NaN
destroyed_to_background = NaN
destroyed_impact_score = 0
support_warning = NO_DESTROYED_SUPPORT
```

测试通过。

没有 GT damaged 时，对应 damaged IoU、Recall 和三种 GT damaged 误分率也会设为 NaN。

## 13. Summarizer 检查

已修复并支持两组参数别名：

```text
--root / --eval_root
--output_dir / --out_dir
```

使用伪造的两个模型、两个 event official 结果树，实际执行：

```bash
python scripts/eval/summarize_eventwise_official_results.py \
  --eval_root <temporary_eval_root> \
  --out_dir <temporary_output_dir>
```

成功生成：

```text
all_model_official_summary.csv
event_wise_summary.csv
event_wise_destroyed_diagnosis.csv
worst_events_top10.csv
model_event_comparison.csv
```

实际汇总数量：

```text
Official summaries: 2
Event rows: 4
Destroyed diagnosis rows: 4
Worst-event rows: 20
Model-event comparison rows: 36
```

排序检查：

```text
all_model_official_summary.csv：
target_3class_miou 降序，通过

event_wise_destroyed_diagnosis.csv：
destroyed_impact_score 降序，通过
```

还验证了：

```text
M07_A4DG_b03 的 destroyed_to_intact
delta_vs_M03_A4 = M07 value - M03 value
```

测试中 M07 比 M03 低 0.1，输出 delta 为 `-0.1`，符合“负值表示改善”的约定。

## 14. Checkpoint 加载信息

`load_checkpoint()` 仍使用 `strict=False` 并继续打印：

```text
Missing keys
Unexpected keys
```

新增写入 `summary_official.json`：

```text
load_missing_keys_count
load_unexpected_keys_count
load_missing_key_examples
load_unexpected_key_examples
```

跨模型总表包含两个 count 字段。

## 15. Unified Runner 检查

静态检查确认 runner 只调用：

```bash
full_evaluate_unified.py --eval_protocol official
```

实际参数名正确使用：

```text
--model_path
--dataset_path
--data_list_path
--save_dir
```

不会调用不存在的：

```text
--model_param_path
--test_data_list_path
```

配置解析已在本地成功找到：

```text
M03_A4 -> best_valid_miou.pth
M05_A4D -> best_valid_miou.pth
A4-DG -> best_valid_miou.pth
```

其他模型在当前同步到 Windows 的权重目录中未全部找到；runner 会打印 warning 并继续。

由于当前系统无 Bash，未执行 shell 语法运行检查。服务器端第一次运行建议先设置：

```bash
EVAL_DESTROYED_CHECKPOINT=0
```

完成一个主 checkpoint smoke test 后再开启 destroyed 专项 checkpoint。

## 16. 仍然存在的风险

1. 当前环境没有 PyTorch，尚未执行真实模型前向。
2. 没有修改前 evaluator 备份或旧 `summary_official.json`，无法做真实 checkpoint 数值逐项 diff。
3. `.sh` 尚未在 Linux Bash 环境实际执行。
4. 六模型中的部分权重目录只存在于服务器或尚未同步，需要核对 `model_eval_config.json`。
5. `strict=False` 仍允许结构不完全匹配；正式结果必须检查两个 load count。
6. `impact_score` 是解释性启发式指标，不等于删除 event 后整体 IoU 的精确变化。
7. event-wise 小 destroyed support 仍可能产生高方差，必须同时查看 `support_warning`。

## 17. 服务器端最终 Smoke Test

建议先用 M03_A4：

```bash
cd /root/shared-nvme/BRIGHT-master

python -m py_compile full_evaluate_unified.py
python -m py_compile scripts/eval/summarize_eventwise_official_results.py

EVAL_DESTROYED_CHECKPOINT=0 \
bash scripts/eval/run_official_eventwise_all_models.sh
```

若只想手动跑一个 checkpoint，应使用配置中的 M03 参数和：

```bash
python full_evaluate_unified.py \
  --eval_protocol official \
  --model_type DamageFormer_BGCA \
  --model_path /path/to/best_valid_miou.pth \
  --dataset_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset \
  --data_list_path /root/shared-nvme/BRIGHT-master/bda_benchmark/dataset/splitname/standard_ML/test_set.txt \
  --image_size 640 \
  --batch_size 1 \
  --num_workers 2 \
  --save_dir /root/shared-nvme/BRIGHT-master/eval_results_official/M03_A4/best_valid_miou \
  --roi_mode dilated_soft \
  --roi_fusion multiplicative \
  --roi_beta 1.0 \
  --roi_alpha 0.1 \
  --roi_dilate 5 \
  --use_mddm \
  --mddm_variant base \
  --mddm_gamma_init 0.0 \
  --mddm_use_roi
```

检查：

```bash
python - <<'PY'
import json
from pathlib import Path

d = Path("/root/shared-nvme/BRIGHT-master/eval_results_official/M03_A4/best_valid_miou")
s = json.loads((d / "summary_official.json").read_text())
required = [
    "official_style_4class_miou",
    "target_3class_miou",
    "background_iou",
    "intact_iou",
    "damaged_iou",
    "destroyed_iou",
    "damaged_recall",
    "destroyed_recall",
    "damaged_to_intact",
    "destroyed_to_intact",
    "loc_iou",
    "loc_f1",
    "loc_head_iou",
    "loc_head_f1",
    "damage_building_iou",
    "damage_building_f1",
]
assert all(key in s and s[key] is not None for key in required)
assert s["loc_iou"] == s["loc_head_iou"]
assert s["loc_f1"] == s["loc_head_f1"]
print("real checkpoint smoke test: PASS")
print("missing/unexpected:", s["load_missing_keys_count"], s["load_unexpected_keys_count"])
PY
```

## 18. 六模型下一步评估

确认服务器权重候选路径后运行：

```bash
cd /root/shared-nvme/BRIGHT-master

bash scripts/eval/run_official_eventwise_all_models.sh

python scripts/eval/summarize_eventwise_official_results.py \
  --eval_root /root/shared-nvme/BRIGHT-master/eval_results_official \
  --out_dir /root/shared-nvme/BRIGHT-master/eval_analysis
```

目标模型：

```text
M03_A4
M05_A4D
A4-DG
A4_DamageExistHead
M07_A4DG_b03
M11_A4HDC_a03
```

首先检查：

```text
eval_analysis/all_model_official_summary.csv
```

随后按 `destroyed_impact_score` 查看：

```text
eval_analysis/event_wise_destroyed_diagnosis.csv
```

比较 M07 与 M03：

```text
eval_analysis/model_event_comparison.csv

筛选：
model = M07_A4DG_b03
metric = destroyed_to_intact
```

其中：

```text
delta_vs_M03_A4 < 0
```

表示 M07 在对应 event 上减少了 destroyed→intact 错误。
