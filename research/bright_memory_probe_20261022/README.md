# 独立 CUDA 数据完整性检查

此工具用于继续调查 BRIGHT 路由输入出现意外比特差异的原因。它不导入 BRIGHT，不读取模型、检查点、Train、Val 或 Test，不执行优化器更新。

## 已知证据和结论边界

用户的 `test2500_night__qzn3ns4/prefix_watched/route_change_cpu.pt` 中，CPU 保存的 `source.b.route_q` 前后快照有以下差异：

- 形状 `[1, 18, 65536]`，仅一个 float32 元素不同；位置为 flat=399783、slot=6、pixel=6567。
- `0x3dd5a746` → `0x3d55a746`，异或为 `0x00800000`，仅指数最低位发生变化。
- 数值 `0.10432295501232147` → `0.052161477506160736`，精确减半；符号和尾数不变。
- 该像素的权重和 `1.0000000075583557` → `0.947838530052195`。

这不能用普通浮点求和误差解释。观察记录显示变化发生在 D forward 前后，B/D/R/L 共享的 route_q 有多个别名；这并不是六个独立张量同时故障。保存的快照尚不能确定变化源于 GPU 存储、非法写入、传输、主机内存或驱动/runtime。版本号未变也不能排除原始 CUDA 内核或别名写入。

不要据此直接宣称显卡损坏，也不要增大路由容差、自动重新归一化或跳过失败图片。此工具用于区分“只在模型执行路径中出现”和“独立 CUDA 负载也能观察到”。

## 运行方式

使用现有 Python 3.10 / Torch 2.1 环境执行发布的 `bright_memory_probe_v1.pyz`：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_memory_probe_v1.pyz start
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_memory_probe_v1.pyz logs
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_memory_probe_v1.pyz status
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_memory_probe_v1.pyz stop
```

`start` 默认后台检查约 20 分钟；在独立进程中运行，可以断开 SSH。控制进程设置额外 180 秒启动及收尾余量，超时终止自己创建的子进程；发生异常后不会循环重启。`logs` 用的是 tail，Ctrl+C 只退出查看。`stop` 校验 PID 的启动时间和命令，只向本次控制进程请求停止。

输出位于 `SRAC_SLOT_MEMORY_ABLATION_DIAGNOSTICS/memory_probe_*`，不会覆盖先前实验。启动时检查 GPU 0 是否空闲，并持有现有续训及独立评估使用的锁；发现其他计算进程会拒绝启动，不终止其他任务。CUDA 设备使用 nvidia-smi 所报告 GPU 0 的 UUID，避免已有 CUDA_VISIBLE_DEVICES 环境变量指向另一张卡。

如果需要固定某次运行，可执行该输出目录中的 `probe.pyz status`；临时包默认查询最近一次独立检查。可选 `--seconds 30..3600 --mib 64..2048`，通常直接使用默认参数。

## 检查了什么

1. 两个持久 CUDA int32 缓冲区合计 1024 MiB，按位检查，不进行浮点归一化。
2. CPU 独立产生七种模式：全零、全一、交替 A/5、地址相关散列、原异常权重比特 `0x3dd5a746`、跨地址循环的单一置位模式。
3. CPU→GPU A→GPU B 写入后，在设备端比较 A/B，再将两者阻塞读回 CPU，与重新生成的预期模式逐字比较。A 和 B 同时发生相同错误也能被 CPU 参考发现。
4. 两次校验之间执行短时全一 FP32 矩阵乘法。结果应精确等于 2048，同时可检查计算结果异常。
5. 首次不一致即保存有限大小的 CPU 证据，再尝试一次读回以区分瞬态和持续可见变化，然后结束。CUDA 异常后不继续读取 GPU 或恢复 CUDA RNG。

这是有限分配范围、有限时长的诊断；不是全显存认证、厂商诊断或物理地址故障定位。两个缓冲区来自同一设备；PyTorch/驱动/传输/CPU 都仍然参与。`checked_bytes` 是累计重复读取的流量，不是覆盖了多少不同物理显存。通过所有模式只能表示本次没有观察到差异。

## 资源及磁盘

默认测试缓冲区合计 1 GiB 显存，矩阵约 48 MiB，此外还有 PyTorch/CUDA 上下文及算子工作空间。不会预分配全部显存，不改时钟、电压或驱动。

普通成功运行只保存程序副本、简短日志、状态及 nvidia-smi 信息。仅遇到不一致时保存一个至多 16 MiB 分块的前后证据，两份压缩 NPZ 的未压缩数据合计最多约 80 MiB，压缩格式有少量额外开销。不会保存整个 1 GiB 缓冲区，也不会生成模型检查点。

## 如何解释结果

| phase | 含义与下一步 |
|---|---|
| `no_mismatch_observed` | 本次有限检查完成且七类模式均执行；不能排除硬件或模型问题。继续检查 D forward 的算子边界、别名/生命周期和跨设备复现。 |
| `data_mismatch` | 脱离 BRIGHT 也出现数据不一致；保留 mismatch.json、NPZ、环境记录，优先调查运行环境/驱动/硬件路径，可向平台提供证据并申请另一物理 GPU 对照。尚未独立证明 GPU DRAM 损坏。 |
| `failed` | 查看 traceback.txt。CUDA 不可用、显存不足或 CUDA 异常均不会被标记为通过。 |
| `wall_time_limit` / `stopped_by_user` | 超时或主动停止，不能算完整通过。 |
| `insufficient_pattern_coverage` | 时长内未覆盖七类模式，也不算完整通过。 |

此工具不会自动重新运行 Test878。2500 步训练与完整 Val 的既有完成记录保留；Test 尚未完整通过的事实不会被修改。

## 本地验证

```bash
python -B -m unittest -v test_probe
python -B build.py
python -B bright_memory_probe_v1.pyz --help
```

开发环境为 Python 3.10.22、PyTorch 2.1.2+cpu。CPU 合同测试覆盖已观察到的指数位差异、相同错误绕过双缓冲比较的情况、读回失败仍保留首份证据、七模式逐字检查、停止和超时，以及真实子进程的完成状态。GPU 实机运行须由服务器执行；本地 CPU 测试不算 CUDA 验证。
