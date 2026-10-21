# 固定 2500 步检查点的夜间 Test 路由诊断

本包针对 `recover2100_dlhrjb52` 的既有故障。该运行已经完成六分支训练到 2500 步、Val450 和固定 Train 诊断；Test878 在第 622 张 `turkey-earthquake_00000627` 的 D 分支检查子槽 `route_q` 时失败。父 Q 检查在该调用中已通过。B、D、R、L 共享 soft 子槽路由，失败发生在 D 读取支持内容之前。

现有日志没有给出负值、逐像素权重和偏差以及原检查的具体返回值，因而还不能确认输入被改动、数值检查异常或其他执行问题。本包补充评估阶段的这些证据，不宣称修复历史故障。

## 运行范围

- 固定使用上述运行的同一个 checkpoint2500，恢复六分支状态，优化器更新数为 0。
- 保留原模型、支持集、损失/评估实现和路由接受条件；不截断、重归一化、放宽容差或跳过图片。
- 原恢复包的 SHA256 必须为 `aec1a463e26ce8d9620863801ba44420fe951da91a6add9e222736d9a171197c`。
- 启动前检查原 2500 步检查点、完整 Val450/固定 Train 证据，以及故障 Test 的 621 张完整记录。旧文件保持原样，输出写入新的诊断目录。
- Test 已经参与历史探索。本包只诊断固定方案的运行问题，不用 Test 标签拟合参数、选择检查点或选择多次重试中的最高分。

## 一次启动的固定顺序

| 阶段 | 输入 | 目的 |
| --- | --- | --- |
| `target_watched` | 故障图 1 张 | 保存父 Q、子槽及单槽路由 CPU 快照，检查单图是否复现 |
| `prefix_passive` | 原名单前 622 张 | 保留原检查算子与顺序；在失败时保存原检查操作数和返回值 |
| `prefix_watched` | 原名单前 622 张 | 在提取、准备支持、各分支前后、概率检查前后比较路由字节内容 |
| `full_test_watched` | 全部 878 张 | 仅当前一阶段完整通过后运行；使用同样的快照观察 |

每个阶段使用全新的进程和 CUDA 上下文。前两个诊断中的推理失败会被保留，然后进入预先指定的独立对照；配置、输入来源或完整性检查失败会停止后续阶段。不存在重复运行直到成功的循环。前缀失败会阻止完整 Test，而不是跳过失败图片。

`watched` 会复制到 CPU，产生 CUDA 同步并改变运行时序。`passive` 在成功检查中不增加 CUDA 运算或复制，但 Python 插桩也有执行开销。任一对照通过都不等于精确复现历史执行或证明根因已消除。所有输出保留 `historical_failure_resolved=false`。

## 启动和查看

将经过 SHA256 校验的发布包下载到 `/tmp/bright_test_route_night_v1.pyz` 后执行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_test_route_night_v1.pyz start
```

默认每阶段 7200 秒、总计 28800 秒。快照诊断比普通推理慢；这些是上限，不是运行时长预测。可在首次启动时显式设置预算，例如 `start --case-timeout 10800 --total-timeout 43200`。

启动前的文件核对在前台进行，打印 `NIGHT PREFLIGHT`。成功后打印 PID 和新输出目录。程序已经自行脱离终端，不需要再加 `nohup`。可以断开 SSH，但服务器必须保持开机、分配和存储可用。

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_test_route_night_v1.pyz logs
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_test_route_night_v1.pyz status
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_test_route_night_v1.pyz stop
```

三条命令分别查看实时日志、读取汇总、请求停止。本包记录自己的最新目录，不依赖 SSH 会话中的 `BRIGHT_RUN` 等变量。`logs` 中 Ctrl+C 只结束查看。`stop` 核对 PID 启动时间和命令后只停止本包进程；超时会停止本阶段的子进程组。

新目录形如：

```
/root/shared-nvme/BRIGHT-checkpoints/SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS/test2500_night_XXXXXXXX
```

启动输出同时提供该目录中 `runner.pyz logs/status` 的固定路径。它不会随着下次启动改变，比 `/tmp` 入口更适合回看特定任务。

## 次日看什么

- `night_job_finished=true`：夜间任务结束，可能成功、失败、超时或被停止。
- `test_complete=true` 且 `complete=true`：完整 Test878 以及模型、优化器、RNG、文件和路由观察检查均通过。
- `cases`：各阶段返回码、完成图片数、错误和首个失败位置。
- `diagnostic_failures`：保留早先失败；后续成功不会抹掉失败事实。
- `heartbeat.json`：每 30 秒更新的调度心跳，不能代替图片计数或推理完成证据；任务结束后不会继续更新。

每个阶段目录保存 `case.log`、`summary.json`、逐图 `images.jsonl`，以及适用的诊断文件：

| 文件 | 含义 |
| --- | --- |
| `route_snapshots.jsonl` | 各边界的对象/存储共享关系、CPU 数值统计与内容变化 |
| `route_failure.json` / `route_failure_cpu.pt` | 原始路由检查的失败操作数与返回值；CUDA 上下文已报错时不再读取 GPU |
| `route_change.json` / `route_change_cpu.pt` | 实际观察到的路由变化以及前后 CPU 副本 |
| `watch_failure.json` / `watch_failure_cpu.pt` | 异常位置及此前已经取得的 CPU 副本；异常后不增加 GPU 读取 |
| `target_routes_cpu.pt` | 目标图成功时的 CPU 路由快照 |

CPU 副本只为当前图片保留，完整张量只在故障或目标图时落盘；正常图片记录紧凑统计。相同对象的别名只复制一次；后续字节一致时复用统计，减少 CPU 求和开销。

判断顺序：先看是否真实出现负值/非有限值/归一化偏差，再看共享路由是否跨分支变化，最后核对原 GPU 检查操作数的 CPU 结果。GPU 原返回与复制操作数的 CPU 结果不一致只能作为进一步排查线索，不能单独证明硬件、驱动或 PyTorch 故障。若只在不同时序下表现不同，仍需保留“未定位根因”的结论。

## 构建与验证

`build.py` 从固定 SHA 的恢复包构建。除入口外，所有继承模块保持逐字节一致，并在 `NIGHT_BASE.json` 记录摘要。新增逻辑仅在 `night_*.py` 中。

本地使用 CPU 进行路由、推理一致性、报告完整性和调度测试；无法代替服务器的 RTX4090 复现。验证范围见 `VALIDATION.md`。
