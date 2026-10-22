# BRIGHT 夜间顺序实验

这个独立控制器不修改已安装的实验代码。先只读等待 `original_3407_01` 与 `no_reader_3407_01` 完整完成；已有任务失败就停止，不重新训练旧任务。控制器本身不占 GPU。

默认顺序：等待已有配对完成 → 比较 `original−no_reader` → no_label 三步 smoke → no_label 2500 步 → 比较 `original−no_label`。

加 `--include-read-aux`：上述步骤之后，再运行 read_aux（辅助权重0.01）三步 smoke 与2500步，并比较 `read_aux−original`。这是事先安排的探索，不保证提分，不根据当前Val分数自动选择是否运行。

所有新任务使用seed3407，沿用original的source、支持、学习率倍率和固定Train集合大小。每项均为新后端初始化，独立2500步，不续训旧模型，不访问Test。

| 对比 | 验证目的 |
|---|---|
| original − no_reader | 现有区域后端之外，支持读取的额外价值 |
| original − no_label | 保留视觉支持时，显式历史标签路径的额外价值 |
| read_aux − original | 相同候选下，显式读取监督能否改善最终分类 |

固定输出目录为父目录下 `no_label_3407_night_01`、`read_aux_001_3407_night_01`；各短程输出在对应名称后加 `_smoke`。已完整且配方一致的输出可以复用；未完成或失败的输出不会覆盖，也不会自动重试或恢复。当前实验包没有自动续训入口。

控制器有独立互斥锁，拒绝重复夜间队列。启动每个GPU子任务前等待已有实验锁释放，且GPU0无计算进程；它不会终止原先的训练或其他GPU进程。外部程序不遵守本锁时仍可能发生竞争，本实验入口也会再次检查自己的运行锁。

等待单个前置条件默认最多12小时；整个控制器最多24小时（包含等待）。到期会退出；若它自己启动的子任务仍在运行，会终止这个子任务并保留已写输出，不会杀死被等待的原任务。无法保证全部任务在一夜内完成。

## 启动

将脚本放在 `/root/shared-nvme/BRIGHT-master/New/bright_readout_night_20261022/night_followup.py`。前面的原配对任务可保持运行。下面命令只执行一次：

```bash
mkdir -p /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1/logs
```

```bash
nohup /root/shared-nvme/conda-envs/bright4090/bin/python -B -u /root/shared-nvme/BRIGHT-master/New/bright_readout_night_20261022/night_followup.py --include-read-aux >> /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1/logs/night_followup_3407.log 2>&1 < /dev/null &
```

只补no_label时去掉 `--include-read-aux`。

```bash
tail -n 60 -f /root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_READOUT_DESIGN_V1/logs/night_followup_3407.log
```

Ctrl+C只结束tail。不要通过重复运行启动命令刷新进度。

每个控制器在父目录下创建唯一 `night_followup_...`，启动日志打印 `NIGHT OUTPUT=...`。其 `status.json` 记录阶段、子进程日志与退出码；每个子任务独立日志；比较结果在 `comparison_reader`、`comparison_label`、`comparison_read_aux` 中。只有全部指定任务与比较成功才打印 `NIGHT COMPLETE`。

`NIGHT HEARTBEAT` 表示控制器仍活跃，可能正在等待当前no_reader或GPU，不表示新训练已经开始。`NIGHT START no_label_train` 等明确表示已启动对应子任务。失败后保留状态与traceback，勿删目录重新运行来掩盖失败。

本控制器的标准库测试验证状态检查、互斥、失败传播和自身子进程超时清理。它不证明新方法提高精度，也不证明历史CUDA故障根因已修复。
