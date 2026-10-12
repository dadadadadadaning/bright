# 02 本地 CPU official 评估计划

## 流程

1. dry-run 扫描权重并核对参数。
2. 使用 20 张列表做单 checkpoint smoke test。
3. 按优先级 1 → 2 → 3 执行完整 878 张 official 评估。
4. 单模型失败写入 `eval_failed_models.csv`，后续模型继续。
5. 汇总 overall、event-wise 和 destroyed 诊断。

## 命令

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local_eval\dry_run_local_eval_cpu.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\local_eval\run_local_eval_cpu.ps1 -SmokeTest -MaxCheckpoints 1
powershell -ExecutionPolicy Bypass -File .\scripts\local_eval\run_local_eval_cpu.ps1
```

## checkpoint 规则

`best_valid_miou` → `best_balanced_score` → `best_damaged_iou` → `best_destroyed_iou` → `best_model` → `best`。只有无 best 时才回退到 `final_model` 或 `latest_resume`。

## 当前状态与耗时

- registry checkpoint：26
- 已完成正式 878 张结果：24
- 20 张 smoke test 已通过。
- 实测约 12 秒/图；全部可执行 checkpoint 预计约 120–160 小时。
- 风险：CPU 长任务、Windows 休眠/重启、未知自定义架构、历史 args.json 缺少新参数。
