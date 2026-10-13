# BRIGHT：1500 步后的现场取证与 CPU 汇聚续训 V2

这是一个独立的修复续训工具。它读取现有三组 1500 步 checkpoint，先检查实际模型前向，再完成 Val450，最后继续第 1501–5000 步。旧实验目录、BRIGHT 模型源码、原案例库和旧 checkpoint 均不覆盖。

## 这次实际修改了什么

旧恢复中的 `Original-semantics CPU pooling reference gate failed` 尚未定位到确定的硬件或软件根因。仅重新计算保存的输入不能解释当次完整模型前向的差异。

本工具先在同一次实际前向调用中保存输入、实际 CUDA 输出、当次原始 CPU 参考输出、布局、线程及精度信息。原比较仍使用 `atol=rtol=2e-5`，失败不会改写成通过。

续训明确使用另一种执行方案：**冻结前端的 soft_pool_unpool 在 CPU 上执行原始 FP32 算法，结果送回 GPU**。模型主干和三个 Reader 仍在原 GPU 上运行。该方案避开此区域汇聚算子的 CUDA scatter 路径；不是异常后自动回退，不修改索引、Q、温度、key、标签、损失、Reader 结构或累加精度。

CPU 汇聚可能显著变慢。它是改变执行后端的独立续训，不是已经证明根因消失的 CUDA 修复。结果应注明 **1500 步原后端前缀 + 3500 步 CPU 汇聚后端续训**。固定案例库由原 GPU 后端提取；新查询的汇聚执行设备改变，所以也不能宣称库和查询仍严格来自完全相同的数值执行路径。

## 一条入口的执行顺序

1. 核对原失败实验、1500 步 checkpoint、原工具和 E0/V3 协议；单独绑定最新恢复失败的 `failure.json`。
2. 等待已声明的两个 SAR 任务结束，再等待原 GPU 空闲；SAR 的 `failed` 只表示它已结束，不表示它成功。
3. 新进程进行一次实际模型的 CUDA 连续索引前向，到已绑定的首图统计汇聚处保存现场输出；严格数值不一致允许作为**诊断结果**保留，但不算训练检查通过。缺少现场输出或出现 CUDA 断言会停止。
4. CPU 分析现场输出与重新计算结果。使用原线程数和 4 线程分别重放，另做独立 FP64 对照；FP64 仅用于分析，不进入训练。
5. 新进程使用显式 CPU FP32 汇聚完成同一张图的完整前向和三个头的预测检查，不调用优化器更新。再检查该次统计汇聚与两种独立 FP64 参考的差异，容差不变。
6. 新进程加载三组权重及 AdamW 状态，重评 checkpoint 1500 的完整 Val450 并核对冻结来源及库完整性。
7. 检查原训练 1500 步的样本顺序，恢复 checkpoint RNG；随后才更新 1501–5000 步，保留原学习率计划及中间验证。
8. 完成最后 checkpoint 的 Val450、保存预测复核和最终完整性检查后，才写正式 `complete=true`。

有两层运行锁：与旧 V1 恢复共享锁，并对新结果父目录加锁，防止两个恢复同时运行。不要同时启动旧 `start_recovery.sh`。

## 手动下载后如何运行

首选下载仓库根目录的 `RUN_JOINT_LIVE_RECOVERY_V2.py`，保存到服务器 `/root/shared-nvme/BRIGHT-master/New/`。该文件内嵌全部工具并验证 SHA256，运行时不访问网络，不依赖额外 ZIP 或 `.sha256` 文件。发布说明同时给出该入口文件本身的 SHA256。

```bash
cd /root/shared-nvme/BRIGHT-master/New
/root/shared-nvme/conda-envs/bright4090/bin/python RUN_JOINT_LIVE_RECOVERY_V2.py
```

默认在后台启动一次，输出 PID、工具目录和启动日志；断开 SSH 不会停止它。**打印 PID 只表示启动，不表示数值检查或训练通过。** 每 30 秒在启动日志输出子程序阶段和最近进度。

若只想运行前向及数值检查、不续训：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python RUN_JOINT_LIVE_RECOVERY_V2.py --stage probe
```

`--foreground` 在当前终端运行；`--extract-only` 只校验并提取工具；`--help` 查看所有路径参数。无需设置 `KEY_OUTPUT_ROOT` 等前几轮临时环境变量。

默认读取的原训练目录：

```
/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_MEMORY_JOINT_SYNC_V1/run_20261013_073724_212223_3841256
```

默认读取的最新数值失败恢复目录：

```
/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_POOL_RECOVERY_V1/run_20261013_123313_019791_4018193
```

新结果写入 `SRAC_JOINT_LIVE_RECOVERY_V2/run_...`。如果这些默认输入已移动，使用 `--failed-run`、`--tool`、`--latest-failed-recovery` 指定**真实原目录**，不要创建空目录代替。当前单图探针要求最近数值失败对应 Val 的第一张图；如果现场证据不符合，会明确停止，不擅自换成另一张图。

## 怎样看进度和最终结果

启动器打印的 `tail -n 60 -f "日志路径"` 可以直接运行。日志中 `RUNNING train ...` 可能仍在重评 Val；只有 `Joint recovery fit` 且 step≥1501，才实际发生新增训练更新。

查看原样报告：

```bash
"/root/shared-nvme/conda-envs/bright4090/bin/python" "打印出的工具目录/recovery.py" \
  --stage report --root "本次打印出的OUTPUT_ROOT"
```

主要文件：

| 文件 | 含义 |
|---|---|
| `workflow_state.json` / `report.txt` | 全流程状态、三组已更新/新增步数、失败位置 |
| `live-probe/capture.pth` | 同一次 CUDA 实际前向输入及两端输出 |
| `live_analysis/summary.json` | 现场与离线差异，不能等同训练通过 |
| `cpu-probe/summary.json` | CPU 汇聚的单图完整前向与三组预测检查 |
| `cpu_analysis/summary.json` | CPU 后端与独立 FP64 的数值检查 |
| `train/checkpoint_reval/result.json` | 新后端下 1500 步 Val450；不是新训练结果 |
| `train/train.jsonl` | 原 1500 步记录和本次新增更新 |
| `train/checkpoints/` | 新 checkpoint；旧 checkpoint 保留在原处 |
| `train/final_val/result.json` | 5000 步最后模型的完整 Val450 |

任何失败都读本次 `report.txt` 和它列出的子日志。不要凭上一轮 numeric audit 的 `passed` 重复启动旧恢复工具，也不要放宽容差或跳过出错图像。

## 实验解释

三组仍是 `r0`（当前图像内部上下文对照）、`rt`（T128 历史 Train 支持）、`rf`（F256 历史 Train 支持）。`r0` 不是严格意义的“完全无支持 query-only”。本次不新加可拒绝门控，也不硬筛掉背景支持；原 3411 条库保持固定。各组在同一新后端、同一图像流和预算下继续训练。

比较重点是同次最终 Val450 的 `rt−r0`、`rf−r0`、各损毁类别 IoU、误报和误改。不能将本次混合后端续训与从零训练的纯 GPU 5000 步混为同一实验。Test 不访问。

CPU 测试范围和未验证项见 `VALIDATION.md`。没有真实服务器 GPU、E0 权重及数据的执行证据之前，不能宣称服务器故障已解决或训练已完成。
