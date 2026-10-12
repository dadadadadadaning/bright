# BRIGHT Windows CPU-only official 评估

这套脚本只用于本地 evaluation / inference / metrics，不训练模型，也不修改远程服务器代码。

## 环境检查

```powershell
cd D:\software\dowmload\BRIGHT-master
conda activate bright
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`torch.cuda.is_available()` 输出 `False` 是允许且符合预期的。默认配置为：

```text
device=CPU
image_size=640
batch_size=1
num_workers=0
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
```

本地 evaluator 对缺少 `imageio` 的精简环境提供 Pillow 只读兼容层，不修改原始数据加载器。

完整 878 张 × 多 checkpoint 的 CPU 推理会非常慢。当前机器自检时，较简单的 BGCA
约为 15 秒/张；44 个 checkpoint 全量运行可能超过 160 小时。建议先 dry-run，再只对
1 个 checkpoint 运行 20 张 smoke test。

## 1. Dry-run

只扫描权重、推断参数并生成命令，不运行推理：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local_eval\dry_run_local_eval_cpu.ps1
```

检查：

```text
eval_analysis_cpu\local_eval_registry.csv
eval_analysis_cpu\local_eval_commands.txt
```

`PARAM_UNCERTAIN_UNKNOWN` 和不受 unified evaluator 支持的架构不会被盲目执行。

## 2. 生成 20 张测试列表

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local_eval\make_test_subset_20.ps1
```

输出：

```text
bda_benchmark\dataset\splitname\standard_ML\test_set_20.txt
```

## 3. 运行 20 张 smoke test

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local_eval\run_local_eval_cpu.ps1 -SmokeTest -MaxCheckpoints 1
```

脚本优先使用当前 `CONDA_PREFIX\python.exe`。也可显式指定：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local_eval\run_local_eval_cpu.ps1 `
  -SmokeTest -MaxCheckpoints 1 `
  -PythonExe D:\anaconda3\envs\bright\python.exe
```

smoke test 使用独立目录，避免与完整 official 结果混淆：

```text
eval_results_official_local_cpu_smoke20\
eval_analysis_cpu_smoke20\
```

确认首个 checkpoint 正常后，如需对全部已识别 checkpoint 做 20 张检查：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local_eval\run_local_eval_cpu.ps1 -SmokeTest
```

## 4. 运行完整 878 张 official 评估

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local_eval\run_local_eval_cpu.ps1 -Overwrite
```

脚本逐 checkpoint 执行。单个失败会记录后继续后续模型。

## 输出

逐 checkpoint 结果：

```text
eval_results_official_local_cpu\<standard_model>\<checkpoint_hash>\
```

总表和报告：

```text
eval_analysis_cpu\all_model_official_summary.csv
eval_analysis_cpu\event_wise_summary.csv
eval_analysis_cpu\event_wise_destroyed_diagnosis.csv
eval_analysis_cpu\worst_events_top10.csv
eval_analysis_cpu\final_model_selection_report.md
eval_analysis_cpu\eval_failed_models.csv
eval_analysis_cpu\logs\
```

总表按 `target_3class_miou` 降序。日志和失败 CSV 保留 checkpoint、命令及错误摘要，便于复查。

## 如何判断 DG-beta0.3 和 HDC

- DG-beta0.3：与 `M04_A4DG_b10` 比较；如果 `damaged_iou` 恢复，同时
  `destroyed_to_intact` 没有反弹，说明 beta 控制有效。
- HDC：与 `M03_A4_MDDMbase_ROId5` 比较；如果 `destroyed_to_intact` 降低，
  且 `target_3class_miou` 没有明显下降，说明层级式分类值得继续。
- `destroyed_to_background` 高通常偏向定位/ROI 问题；
  `destroyed_to_damaged` 高通常偏向损毁严重程度边界问题。
