# 当前 Memory 恢复失败的数值对照

这个工具只重放一份已保存的失败输入，回答连续索引 CUDA 与原 CUDA、CPU FP32、CPU FP64 分别差多少。它不启动训练、不修改阈值、不改 BRIGHT 源码和旧结果。

默认选择本次失败：

```
/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_POOL_RECOVERY_V1/run_20261013_113711_128984_3991264/train/pool_debug/failure.json
```

自动从该文件读取新 `input_000002.pth`，核对图像、调用编号、输入 hash，绑定该恢复目录的 runtime.json/evidence.json。不要换成最初 call9005 的旧输入。

## 运行

将交付的单文件 `RUN_POOL_NUMERIC_AUDIT_V2.py` 放到 `/root/shared-nvme/BRIGHT-master/New/`，在通过发布时提供的 SHA256 校验后执行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -u /root/shared-nvme/BRIGHT-master/New/RUN_POOL_NUMERIC_AUDIT_V2.py
```

文件内已包含全部辅助代码，运行时解压到自身旁边一个新的目录。无需联网、安装器、KEY_OUTPUT_ROOT、重新建库或重跑模型训练。依赖服务器已使用的 PyTorch 环境。

程序自动新建 `SRAC_JOINT_POOL_NUMERIC_AUDIT_V2/run_...`，打印 `OUTPUT_ROOT=`。CPU 对照先运行；CUDA 对照等待 GPU 连续两次空闲后，分别在独立子进程执行原生和连续索引版本，各重复 3 次。它观察 GPU 空闲状态，不独占预留 GPU；运行期间不要同时启动其他 GPU 任务。

原 CUDA 子进程若触发断言，会立即结束；连续索引对照在新的进程里执行。正常返回但数值不一致时，工具保留输出继续比较，这仅适用于诊断，不会使恢复训练获得通过标记。

CUDA 数值开关按保存的来源配置核对；诊断显式设置 `CUDA_LAUNCH_BLOCKING=1` 以定位异常。CPU 对照/比较暂用最多 4 个线程并恢复原设置。跨后端直接比较各自第一次成功输出，其余重复分别与本后端第一次比较，不代表逐一比较了全部交叉组合。

需要只做 CPU 对照时追加 `--cpu-only`，结果明确标记为不完整对照。已有结果可用 `--report /实际结果根目录` 查看。使用 `--recovery-root` 可显式选择另一份符合相同格式的数值门槛失败；不要指向最初 CUDA 断言目录。

## 四种计算及结果含义

| 对照 | 用途 |
|---|---|
| 原 CPU FP32 | 重现当前恢复门槛使用的参考算法 |
| 原 CUDA FP32 | 检查原算法在同一输入上的跨设备差异和重复稳定性 |
| 连续索引 CUDA FP32 | 检查候选是否有额外差异 |
| CPU FP64 | 直接提高已保存 FP32 输入的计算精度；scatter 和独立分组归约两种实现互相核对 |

FP64 只用于误差分析：不重新计算 raw.square()，不重新生成 Q，不归一化权重，不删重复路由，分母仍 clamp_min(1e-6)。它没有替换原训练门槛。

原门槛逐元素判断 `abs(actual-reference) <= 2e-5 + 2e-5*abs(reference)`，并要求同 shape、同 dtype。FP64 对照单独标注允许混合 dtype。最大绝对差对应的位置未必超限；报告同时记录最大归一化误差、超限坐标、有效区域质量、均值/二阶矩及派生标准差差异。

如果保存的输入本身验证了前 256 通道与后 256 通道的 FP32 平方关系，才分析 mean/second/std。这一统计汇聚下游使用 tokens/mass/mean/std，未使用 reconstruction；后者超限数量不是错分像素数。

## 怎样决定下一步

- 仅连续索引候选偏离：候选仍未验证通过，不能直接恢复；原算法历史上也发生过索引断言，不能据此盲目切回。
- 两种 CUDA 都与 CPU32 有近似差异，且与独立 FP64 更接近：支持浮点计算/参考精度方面的解释。下一步验证明确命名的新数值方案，完整重评 1500 步权重后再考虑续训；不能默默改大原阈值。
- 两种 CUDA 都偏离 FP64 或重复结果不稳定：继续定位汇聚实现和运行环境，不启动长训练。
- 隔离重放全部通过：只能说明这一次没有复现，还要检查完整前向中的调用上下文。

CPU FP64 是独立高精度参考，也不是无条件正确的“真值”；因此工具保留两个 FP64 实现的互相校验。单个输入通过不能证明 Val450 或续训已经通过。

## 输出

- `report.txt`：可直接粘贴的简明报告，包括 Memory 比较和两组 SAR 的记录状态及失败日志尾部。
- `summary.json`：调度状态。`execution_complete` 与 `diagnostic_complete`、`all_numeric_passed` 分开；所有训练通过标记始终为 false。
- `cpu/summary.json`、`native/summary.json`、`contiguous/summary.json`：误差、重复性、运行配置和定位信息。
- `compare/summary.json`：原生与连续索引的双向比较；相对容差参照端不同，边界处方向可能影响结果。
- `sar_status.json`：只读采集 SAR Q/soft 的状态和失败阶段日志。不推断进程存活，不自动重启。
- `outputs.pt`：成功计算的 CPU 输出，供直接交叉比较；源快照和旧 checkpoint 不变。

512×256×256 输入的 CPU 输出约 409 MB，加两组 CUDA 首次输出后约需 0.7 GB 新磁盘空间。它不复制整个训练数据集。CPU 工作区也会消耗额外内存。

如果程序执行失败，查看打印的阶段日志路径；不需要重新训练 1500 步。原 checkpoint 的保留情况由之前恢复流程证明；本工具不会更新其权重、优化器或训练记录。
