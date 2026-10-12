# BRIGHT 本地 Windows CPU-only official 评估

本目录只服务于本地 inference / metrics，不训练模型，也不修改服务器代码。

## 环境

```powershell
cd D:\software\dowmload\BRIGHT-master
conda activate bright
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

预期为 CPU 版 Torch，`torch.cuda.is_available()` 返回 `False`。

## Dry-run

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local_eval\dry_run_local_eval_cpu.ps1
```

输出：

```text
eval_analysis_cpu\local_eval_registry.csv
eval_analysis_cpu\local_eval_commands.txt
```

registry 按优先级排序：主线模型 → ROI/alignment 消融 → IRC/基线/未知架构。
未知架构只登记，不执行。

## 20 张 smoke test

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local_eval\make_test_subset_20.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\local_eval\run_local_eval_cpu.ps1 `
  -SmokeTest -MaxCheckpoints 1
```

smoke 结果写入独立目录：

```text
eval_results_official_local_cpu_smoke20
eval_analysis_cpu_smoke20
```

## 完整 878 张评估

只运行第一优先级主线模型：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local_eval\run_local_eval_cpu.ps1 -PriorityMax 1
```

运行全部可识别模型：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local_eval\run_local_eval_cpu.ps1
```

默认配置：

```text
protocol=official
device=cpu
image_size=640
batch_size=1
num_workers=0
CUDA_VISIBLE_DEVICES=""
```

已经存在 `summary_official.json` 的 checkpoint 会跳过；单个失败不会中断整批。
本机实测约 12 秒/图，全部可执行 checkpoint 预计需要约 120–160 小时。

## 结果

```text
eval_analysis_cpu\all_model_official_summary.csv
eval_analysis_cpu\event_wise_summary.csv
eval_analysis_cpu\event_wise_destroyed_diagnosis.csv
eval_analysis_cpu\worst_events_top10.csv
eval_analysis_cpu\eval_failed_models.csv
eval_analysis_cpu\logs\
```

自动文档：

```text
01_model_inventory.md
02_evaluation_plan.md
03_model_effect_interpretation.md
04_event_wise_destroyed_diagnosis.md
05_final_model_selection_report.md
```

当前本地没有 M07_A4DG_b03 和 M11_A4HDC_a03 正式权重，不能据此判断
beta=0.3 或 HDC 是否有效；需要先补充 checkpoint 与训练 `args.json`。
