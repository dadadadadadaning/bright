# 完整 SAR 四组 → SAR＋soft：可显式共享 Memory 所用 GPU

`RUN_SAR_FULL_CHAIN_V2.py` 默认启动完整实验，不是20步取证模式。单文件内含校验过的工具包，服务器运行时不联网下载，不修改已有模型、实验源码、失败结果或Memory库。

## 直接共享 GPU 运行（本次需求）

将 `RUN_SAR_FULL_CHAIN_V2.py` 保存到服务器 New 目录，执行完整一行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/RUN_SAR_FULL_CHAIN_V2.py --run-now --replace-queued
```

`--run-now`保留Memory进程，跳过Memory结束、独立probe锁和GPU空闲等待。CPU来源核对和原preflight仍执行，随后开始完整SAR四组。四组全部合格后自动接入SAR＋soft；Soft驱动也使用同一个共享策略，不会再次等待Memory结束。

`--replace-queued`只取消同一输出父目录中latest.json绑定、身份匹配且仍在等待的V1/V2完整SAR启动器。短暂暂停该启动器后重新核对状态及子进程，确认尚未开始实验才终止。已有训练或身份不符时拒绝替换；没有旧队列时直接启动新任务。不发送信号给Memory，也不使用旧PID或模糊pkill。独立20步probe属于另一入口，本选项不取消它，直接模式也不等待它。

只取消旧完整SAR队列而不启动：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/RUN_SAR_FULL_CHAIN_V2.py --cancel-queued
```

共享模式把真实GPU进程、可用显存、利用率及`wait_skipped=true`写入日志和实验状态；即使GPU忙，也不会伪造`idle=true`。Q与Soft的manifest都记录`execution_policy=concurrent`。Q四组和Soft仍按顺序执行，并非同时启动12组训练。

共享可能让Memory和SAR变慢或出现显存不足；不自动降低batch、改变精度或修改科学验证门槛。这是调度方式变更，不是冻结A1报错的修复。没有`--run-now`时仍使用下述默认排队模式。

V2继续使用`SRAC_SAR_FULL_CHAIN_V1`结果父目录与锁，以识别已排队的V1并阻止重复启动。新运行有独立目录，`chain_state.json`格式为`SAR_FULL_CHAIN_V2`。

## 默认执行顺序

1. 后台启动并加锁，拒绝同一输出父目录重复启动。
2. 发现当前Memory恢复父/子进程，按PID及进程启动时间等待它们退出；正常或报错退出都视为终止。之后等待已启动的SAR短程取证，并等待GPU连续两次空闲。不会停止任何已有任务；新启动的其他GPU任务仍需避免竞争。
3. SAR Q原CPU preflight、四组各2步smoke训练及各2张独立验证。
4. baseline、rgb_capacity、shifted_sar、sar_cue四组依次各2000步；batch2×accum8，BF16，同一个E0。保留原初始/末次Val450及各组独立Val450。
5. 原四组来源与保存预测验证、SAR/baseline配对分析、Q审计。
6. 仅在四组全部成功且冻结检查通过后，自动将**新Q结果目录**绑定给SAR＋soft。旧失败来源不会被选中。
7. 使用baseline-Q和SAR-Q两个来源，分别训练unsplit、capacity_control、hard、soft四种读出头，共八头；执行原smoke、完整训练、独立Val450及八组比较。沿用原工具固定预算，每头2000次更新。

全过程不访问Test。流程成功与方法取得正收益是两件事，最终以完整Val报告为准。

## 冻结故障处理

此前baseline在2000步后出现`Frozen A1 parameter or buffer changed`，该根因仍未在4090上证实解决。本工具提供更完整的现场证据，不宣称已经修复模型或硬件。

每个Q训练子进程使用与原数学计算相同的训练器，在第1步、每20步及Val前后检查冻结状态，并检查前向时的模式与梯度开关。失败保存变化张量的名称、前后数值、初始哈希和检测步骤。检查点间隔内的精确变更时刻不作保证。原冻结、来源、运行环境和独立评价检查均保留。

再次失败会立即停止后续阶段，无自动重试、放宽门槛或修改成功标志。失败在最后检查前可能仍没有可正式恢复的中途权重；本入口不增加断点续训，也不把诊断快照当作合格checkpoint。

旧baseline的2000步权重虽存在，但未通过冻结校验，无法作为合格的匹配对照接入。默认四组从同一E0重新训练到2000步，既不是从2001步续训，也不是复用旧baseline配上新三组。旧文件原样保留。

## 一条命令后台启动

手动将`RUN_SAR_FULL_CHAIN_V2.py`保存到`/root/shared-nvme/BRIGHT-master/New/`，核对发布页面SHA256后执行这一行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/RUN_SAR_FULL_CHAIN_V2.py
```

默认运行两套完整实验。会立即打印PID、日志路径、新Q来源目录和Soft目录。可断开SSH；启动不等于通过。重复执行不会启动第二条同父目录流水线。

状态查看也是一行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /root/shared-nvme/BRIGHT-master/New/RUN_SAR_FULL_CHAIN_V2.py --status
```

默认结果在`/root/shared-nvme/BRIGHT-checkpoints/SRAC_SAR_FULL_CHAIN_V1/run_*/`，目录时间使用北京时间UTC+8。`--status`读取本入口latest.json所绑定的那次启动，不会扫描其他实验并猜测最新结果。需要指定历史根目录时加`--root /绝对路径`。

新根目录内：

- `chain_state.json`：总流程状态；Q完成后不会再修改Q自己的认证文件。
- `launcher.log`：总日志。
- `sar_q/workflow_state.json`：四组Q流水线进度。
- `sar_q/frozen_witness/<阶段>/failure.json`：若发生冻结变化，保存具体差异。
- `sar_q/formal/val_<组>/summary.json`：Q独立Val450。
- `sar_soft/workflow_state.json`：Soft进度。
- `sar_soft/comparison/report.json`：完整八头比较。

`WAIT_DECLARED_MEMORY`、`WAITING_EXISTING_SAR_PROBE`、`GPU_QUEUE`表示排队，不是训练失败。若当前Memory持续运行，新任务会继续等待，不能保证指定时刻完成全部实验。

## 可选范围

只跑完整Q四组：在启动命令末尾加`--target sar-q`。

若以后已有完整合格Q来源，只单独启动Soft：加`--target sar-soft --sar-root /完整通过的Q根目录`。旧失败来源会被拒绝，Soft仍会重新做独立来源核验；输出使用新目录。

`--plan`只打印计划，`--extract-only`只校验并解出工具；二者都不会启动训练。

## 验证范围

CPU测试覆盖完整计划替换、两阶段成功条件、Q失败不启动Soft、冻结证据预算和来源哈希、环境恢复、重复启动与后台锁传递、嵌入包校验、只读状态入口。原Q/Soft的认证检查沿用原代码。

真实服务器4090、实际权重和BRIGHT数据尚未在此云工作区执行。不能把CPU测试或队列启动当作正式实验完成。
