# BRIGHT 六分支正式续训

从原始实验的 1000 步检查点恢复 E/F/B/D/R/L 六个分支，默认训练到原计划的
2500 步。完整恢复各分支模型、AdamW 状态和随机状态，核对原采样顺序，再执行
1001–2500 步。代码调用服务器上经过校验的原始更新函数、NLL、Lovasz 和评估器。

此入口是独立目录中的检查点续训。此前单图探针通过不能证明长程 CUDA 故障已修复；
检查点频率和诊断标记会改变执行时序，不声明精确复现历史故障轨迹。

## 启动

使用服务器原环境执行下载的归档：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -u /tmp/bright_slot_memory_resume_v1.pyz
```

默认输入：

```text
/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_V1/run_20261020_025710_3840142
```

默认输出基目录：

```text
/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_RESUME_V1
```

每次启动创建新的 `resume_时间_PID` 目录，并将本次归档复制为该目录的
`trainer.pyz`。后台子进程独立于 Xshell 会话，启动器返回 PID、日志路径和查看命令。
同一输出基目录通过持续持有的文件锁防止重复启动。

启动器成功返回只表示子进程已创建。查看 `train.log` 和 `summary.json` 判断实际进度：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /tmp/bright_slot_memory_resume_v1.pyz status
```

最初会核对继承的输入文件指纹，然后重新执行 1000 步 Val450 和固定 Train450
评估。因此首次出现 `RESUME TRAIN` 之前可能需要一段时间。

## 已保存评估的只读检查

若之前在 `evaluate_val_1000` 阶段报 `Inconsistent class nll sum`，
可先用本版检查已有的评估文件：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /tmp/bright_slot_memory_resume_v1.pyz check-evaluation --root /实际的续训目录
```

此命令不使用 GPU、不训练、不改写或提升旧报告的完成状态。它重新读取逐图日志，
独立重建各分支的全局及逐事件标量、类别分项和计数，并核对已保存结果与校验和。
只有已有报告文件参与检查，输出通过不代表整个训练完成。

早期版本把两条不同的 FP32 归约路径强行按 `rtol=2e-6, atol=2e-7`
比较：原始总 NLL 使用 `F.nll_loss`，类别贡献使用各类别掩码后的 `sum`。
百万像素 CUDA 归约可合法超出这一容差。本版保持原始 NLL 数值和类别贡献不变，
把两者差值明确记录为 `nll_reduction_diagnostics`，并以逐图日志的独立重建检查
每个汇总字段。原损失计算、Lovasz 分项检查、路由容差和训练流程均保持原样。
若这一阶段尚未产生任何新更新或检查点，检查通过后用默认 `start` 从原1000步重新开始。

## 训练和评估

- 保留原始六分支顺序、原损失、5000 步学习率调度及 500 步后的倍率、裁剪和数据划分。
- 每 50 个完整六分支更新步保存恢复检查点；1500、2000、2500 步也保存。
- 验证恢复边界的 Val450 和固定 Train450，之后按原计划验证 1500、2000、2500 步
  Val450，并在终点评估同一固定 Train450。
- 评估沿用原评估器的随机状态与模型模式恢复逻辑。
- 仅在所有更新、所需评估、文件校验及运行环境退出检查通过后，将新报告标记为完成。
- 原始实验目录、原代码、原检查点和旧评估报告不改写；不访问 Test。
- 任一分支报错即停止，不跳过样本、不重试该步、不修改 NLL 或概率、不保存部分更新状态。

新 `train.jsonl` 先复制与恢复检查点对应的完整历史前缀，再追加本次更新。
`last_complete_step` 只计入六个分支都已完成且已写入日志的步。
失败步中已返回的优化器更新单独记录，不能作为恢复状态。

检查点保存全部六分支的模型、优化器、RNG、原计划和输入指纹；重新加载核对完整
内容指纹后发布检查点及 sidecar。保存不会改变优化器或 RNG。完整输入审计在启动、
原验证边界和最终检查阶段执行。

## 再次中断后的恢复

先查看失败目录的 `summary.json`、`traceback.txt` 和 `markers.jsonl`。
确认继续时，用 `--parent-root` 指定该正式续训目录：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -u /tmp/bright_slot_memory_resume_v1.pyz start --parent-root /实际的续训目录
```

程序选择这个目录中最近的有效完整检查点，并创建另一个新目录；保留上一次失败证据。
拒绝活动中的父运行、已完成父运行、诊断探针检查点和不匹配的实验来源。
检查点之后未完整写入的日志不作为恢复前缀。若首次续训尚未保存任何新检查点，
可从原 1000 步检查点重新启动。

默认入口仅支持本实验第一轮数据范围内的既定终点，不修改训练计划或延长训练。

## 本地构建和测试

```bash
python slot_memory_resume_v1/build_zipapp.py
BRIGHT_ABLATION_REFERENCE=/path/to/joint_anchor_fresh_chain_v2 python -m unittest discover -s slot_memory_resume_v1 -p 'test_*.py' -v
```

归档仅包含运行代码和本文档，不包含数据、日志、权重或测试夹具。测试范围及局限见
仓库中的 `VALIDATION.md`。
