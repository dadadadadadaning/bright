# BRIGHT Windows CPU-only 评估流程检查报告

检查日期：2026-06-22  
项目路径：`D:\software\dowmload\BRIGHT-master`  
检查目标：仅在 Windows 本地执行 evaluation / inference / metrics，不训练，不修改服务器代码。

## 1. 环境是否可用

使用解释器：

```text
D:\anaconda3\envs\bright\python.exe
```

实测输出：

```text
torch 2.12.1+cpu
torch.cuda.is_available() = False
```

结论：环境可以导入 PyTorch，本地评估环境可用。

检查过程中发现，直接启动新的 PowerShell 时可能先找到
`D:\anaconda3\python.exe`，该 base Python 没有安装 Torch。已修复
`run_local_eval_cpu.ps1` 和 `dry_run_local_eval_cpu.ps1`：

1. 优先使用当前 `CONDA_PREFIX\python.exe`；
2. 验证该解释器能否导入 Torch；
3. 失败时回退到 `D:\anaconda3\envs\bright\python.exe`；
4. 将同一个解释器传给所有评估子进程。

## 2. 本地是否为 CPU-only

是。

本地 wrapper：

```text
full_evaluate_unified_local_cpu.py
```

执行了以下限制：

```text
CUDA_VISIBLE_DEVICES=""
torch.cuda.is_available() 必须为 False
num_workers 强制为 0
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
```

如果 wrapper 意外检测到 CUDA，会主动停止，而不会切换到 GPU。

## 3. full_evaluate_unified.py 参数支持情况

实际运行 `--help` 后确认以下参数全部存在：

```text
--eval_protocol
--mddm_variant
--dg_large_beta
--use_hdc
--hdc_alpha
--use_damage_exist_head
```

结论：本地主 evaluator 是支持这些模型参数的版本，无需从服务器同步。

## 4. checkpoint CPU 加载

位置：

```text
full_evaluate_unified.py:201
```

当前代码：

```python
checkpoint = torch.load(model_path, map_location="cpu")
```

20 张 smoke test 的 checkpoint 加载结果：

```text
Missing keys: 0
Unexpected keys: 0
```

结论：checkpoint 使用 CPU 加载，且本次测试权重与模型结构完全匹配。

## 5. CUDA-only 代码检查

扫描范围：

```text
full_evaluate_unified.py
full_evaluate_unified_local_cpu.py
scripts/local_eval/*.py
```

未发现评估路径调用：

```text
.cuda()
torch.cuda.synchronize()
autocast
GradScaler
```

主 evaluator 的设备选择为：

```python
torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

`pin_memory` 也以 `torch.cuda.is_available()` 为条件。本地 wrapper 额外隐藏
CUDA 并验证 CPU-only，因此没有 CUDA 专用调用风险。

## 6. Swin 预训练权重下载

原始 `DamageFormer.py` 构造函数仍包含：

```python
swin_t(weights=Swin_T_Weights)
```

本次没有修改原始模型或服务器代码。本地 wrapper 在模型实例化前将模型模块中的：

```python
Swin_T_Weights
```

替换为：

```python
None
```

BGCA 模型则由 unified evaluator 传入：

```python
use_pretrained_swin=False
```

实测：

1. 原始 DamageFormer 可以在 CPU 下离线初始化；
2. smoke 日志中没有 `Downloading`、HTTP 或权重下载记录；
3. checkpoint 随后正常覆盖模型参数。

结论：本地评估不会联网下载 Swin 预训练权重。

## 7. Dry-run 检查

执行命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local_eval\dry_run_local_eval_cpu.ps1
```

结果：

```text
识别 checkpoint：44
可执行 checkpoint：41
PARAM_UNCERTAIN / unsupported：3
latest_resume 被选择：0
Linux /root/shared-nvme 路径：0
使用 bright Python 的命令：44/44
```

输出文件存在：

```text
eval_analysis_cpu\local_eval_registry.csv
eval_analysis_cpu\local_eval_commands.txt
```

模型参数检查：

- 本地存在 4 个 `M04_A4DG_b10` checkpoint；
- 命令均显式包含 `--mddm_variant dual_scale_gate`；
- 命令均显式包含 `--dg_large_beta 1.0`；
- 本地没有 `M07_A4DG_b03` checkpoint，因此无法生成或验证 beta=0.3 的实际评估命令；
- 本地没有 `M11_A4HDC_a03` checkpoint，因此无法生成或验证 HDC 的实际评估命令。

未知模型正确标记为 `PARAM_UNCERTAIN_UNKNOWN`：

```text
BRIGHT\DamageFormer_BuildingGuidedContrast_lambda001_20260531_211811
BRIGHT\UNet_20260514_141625
BRIGHT_mymodel\MyModel01_complete_20260517_135420
```

这些架构不受当前 unified evaluator 支持，不会被盲目评估。

## 8. 20 张 smoke test

测试列表：

```text
bda_benchmark\dataset\splitname\standard_ML\test_set_20.txt
```

实测行数：20。

执行命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\local_eval\run_local_eval_cpu.ps1 `
  -SmokeTest -MaxCheckpoints 1 -Overwrite
```

测试模型：

```text
M01_BGCA_none
best_model.pth
```

结果：通过。

确认事项：

1. 模型初始化成功；
2. checkpoint CPU 加载成功；
3. 20/20 张 forward 完成；
4. `Missing keys=0`；
5. `Unexpected keys=0`；
6. 没有 CUDA 报错；
7. 没有联网下载预训练权重；
8. Python `json.load()` 可正常读取 `summary_official.json`；
9. summarizer 成功汇总 1 个 checkpoint、1 个 event；
10. `eval_failed_models.csv` 没有失败行。

输出目录：

```text
eval_results_official_local_cpu_smoke20\
  M01_BGCA_none\
    best_model_cac19206\
```

已生成：

```text
summary_official.json
per_class_metrics_official.csv
confusion_4class_official.csv
event_metrics_official.csv
event_destroyed_diagnosis.csv
event_confusion_4class.csv
analysis_config.json
local_eval_metadata.json
```

smoke test 耗时约 4 分 1 秒，即约 12.06 秒/张。

注意：这 20 张都属于 `bata-explosion`，smoke 指标只用于验证流程，不能作为
完整 official 结果或模型性能结论。

## 9. 完整评估是否完成

未完成，也尚未启动完整 878 张 × 41 个可执行 checkpoint 的长任务。

当前正式输出状态：

```text
eval_results_official_local_cpu 中 summary_official.json 数量：0
eval_analysis_cpu\all_model_official_summary.csv 数据行：0
```

按照 smoke 实测速度估算：

```text
878 images × 41 checkpoints × 12.06 seconds
≈ 120.6 hours
```

更复杂的 DG、IRC、large-scale 模型可能更慢，实际总耗时可能达到约
120–160 小时。因此本次检查没有将五天级任务静默挂到后台。

确认需要完整运行时使用：

```powershell
cd D:\software\dowmload\BRIGHT-master
conda activate bright
powershell -ExecutionPolicy Bypass -File .\scripts\local_eval\run_local_eval_cpu.ps1
```

正式配置已确认：

```text
batch_size=1
num_workers=0
CUDA_VISIBLE_DEVICES=""
image_size=640
official protocol
```

脚本支持断点式继续：已经存在 `summary_official.json` 的 checkpoint 默认跳过，
单 checkpoint 失败会记录后继续。

## 10. 模型失败情况

### Smoke test

没有模型失败。

### 完整评估

因为尚未启动，所以没有正式推理失败记录。

以下 3 个 checkpoint 会被主动跳过，不属于运行时失败：

| 模型目录 | 原因 |
|---|---|
| `DamageFormer_BuildingGuidedContrast_lambda001` | 自定义结构不受当前 unified evaluator 支持 |
| `UNet_20260514_141625` | UNet 不受当前 unified evaluator 支持 |
| `MyModel01_complete_20260517_135420` | MyModel 自定义结构不受当前 unified evaluator 支持 |

完整运行后的失败记录位置：

```text
eval_analysis_cpu\eval_failed_models.csv
eval_analysis_cpu\logs\
```

## 11. 本次检查中修复的问题

只修改了本地 CPU 脚本，没有修改服务器代码：

1. 修复新 PowerShell 进程选择到无 Torch 的 base Python；
2. dry-run 和正式 runner 现在都会验证 Python 是否能导入 Torch；
3. 所有生成命令统一使用可用的 bright Python；
4. 对旧 `args.json` 未记录 `dg_large_beta` 的 M04 模型，显式补充
   `--dg_large_beta 1.0`；
5. smoke 与完整 official 结果使用独立目录，避免混表；
6. 本地缺少 `imageio` 时使用 Pillow 只读兼容层；
7. Matplotlib cache 改到临时目录，避免 Windows 用户目录权限 warning。

## 12. 下一步建议

1. 当前流程已经证明能够在本地 CPU 上真正完成 official 推理、event-wise 输出和汇总。
2. 不建议立即一次性运行全部 41 个 checkpoint；优先运行论文候选 checkpoint。
3. 建议先分别评估：

```text
M03_A4_MDDMbase_ROId5 / best_valid_miou
M05_A4D_MDDMlarge / best_valid_miou
M04_A4DG_b10 / best_valid_miou
A4_DamageExistHead / best_valid_miou
```

4. 本地目前没有 M07 beta0.3 和 M11 HDC 权重。若需要判断它们是否有效，必须先将对应
checkpoint 复制到 `saved_weights`，并带上训练 `args.json`。
5. 完整评估期间可随时查看：

```text
eval_analysis_cpu\logs\
eval_analysis_cpu\eval_failed_models.csv
```

6. 全部完成后重点查看：

```text
eval_analysis_cpu\all_model_official_summary.csv
eval_analysis_cpu\event_wise_destroyed_diagnosis.csv
eval_analysis_cpu\final_model_selection_report.md
```

## 最终结论

Windows CPU-only BRIGHT 评估流程已经通过真实 20 张推理验证，可以运行。

当前尚不能声称“完整 official 评估完成”：正式目录仍为 0 个结果，原因是完整任务预计
需要约 120–160 小时。流程本身已跑通；下一步是由用户决定按候选模型分批运行，还是
启动全部 checkpoint 的长时间评估。
