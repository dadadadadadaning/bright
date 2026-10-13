# BRIGHT 当前联合训练：区域池化 CUDA 越界诊断

本包针对 `SRAC_REGION_MEMORY_JOINT_V1/run_20261013_143518_3825644`，安装代码到
`/root/shared-nvme/BRIGHT-master/New`。旧报告、FP32 库、E0 协议、模型权重和原仓库代码全部保留。

## 当前证据与边界

- 原联合流程在 `train/initial_val` 打印 240/450 后，冻结前端的 `SuperpixelEmbedding` 区域统计池化触发 `ScatterGatherKernel index out of bounds`。
- 正式 5000 步训练尚未开始；这不是一次可用于比较 R0/RT/RF 效果的训练结果。
- 原 `PoolGuard` 在 CUDA 操作前已经做 CPU 快照及索引范围检查。不能仅凭栈判定标签越界，也不能直接归因于 Memory 或显存不足。
- 候选代码只连续化候选索引和扩展索引，保留原公式、重复路由、注意力权重、循环次序及 `mass.clamp_min(1e-6)`。这是待验证的实现候选，不是已经定位并解决的根因。
- **本包不自动启动训练，不把失败现场改成通过，不放宽容差，不截断 IDs，不跳过失败图。**

## 自动执行什么

1. 只读旧 `train/manifest.json`、`summary.json`、`pool_debug/{failure,guard,progress,manifest}.json`，绑定 SHA256 并找到确切失败快照。
2. 独立 CPU 子进程：检查快照并计算原公式参考。
3. 独立 CUDA 子进程：重放原 scatter/gather 公式三次，恢复快照记录的布局，检查输出与 CPU 参考、与首轮的差异。
4. 独立 CUDA 子进程：重放 `pool_ops.py` 的连续索引候选三次，使用相同参考与容差。
5. 独立 CUDA 子进程：用**旧训练包的原 Runtime、原模型、原库与原池化**，验证具体失败图的完整初始联合前向。
6. 如果失败图通过，再在另一个新 CUDA 进程中，按照原顺序执行完整 initial Val450。执行原三组初始化检查，保存并重读预测，验证冻结状态；**不执行任何优化器更新**。

每个 CUDA 子进程都在启动前设置 `CUDA_LAUNCH_BLOCKING=1`，用于更准确地定位操作。原历史 runtime 数值配置（包括 TF32、确定性算法开关、CUBLAS 配置）必须保留并记录；不会偷偷改成确定性算法来“修复”。同步执行也属于诊断干预，不能声称原异步故障已经消失。

一次 CUDA device-side assert 会使该进程的 CUDA 上下文不可继续使用，所以各组严格分进程、按顺序执行。失败组仍保留 FAIL，不会因为后续候选成功被覆盖。

## 下载并启动

从本版本 GitHub 页面下载 `bright_joint_pool_diagnose_v1.zip`，上传到：

```text
/root/shared-nvme/BRIGHT-master/New/bright_joint_pool_diagnose_v1.zip
```

在服务器使用发布消息中的 ZIP SHA256 校验后解压到新的 `New/pool_diagnose_XXXXXXXX` 目录，再执行包内 `start_server.sh`。

```bash
(
  set -euo pipefail
  PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
  ZIP=/root/shared-nvme/BRIGHT-master/New/bright_joint_pool_diagnose_v1.zip
  test -s "$ZIP"
  UNPACK="$(mktemp -d /root/shared-nvme/BRIGHT-master/New/pool_diagnose_XXXXXXXX)"
  "$PYTHON" -m zipfile -e "$ZIP" "$UNPACK"
  bash "$UNPACK/joint_pool_diagnose_v1/start_server.sh"
)
```

入口自动校验所有发布文件的 SHA256。脚本会打印诊断 PID、结果目录、日志、读取报告的完整命令。
查看日志时 Ctrl+C 仅关闭 tail，不停止后台诊断。不要重复启动同一诊断；原失败训练进程正常已退出，不需要 kill GPU 上其他作业或重启服务器。

新诊断结果保存在：

```text
/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_POOL_DIAG_V1/run_<时间>_<PID>
```

依赖保留原工具目录：

```text
/root/shared-nvme/BRIGHT-master/New/joint_train_ZAtNI2Wi/region_memory_joint_train_v1
```

如果实际搬移，可通过 `POOL_OLD_TOOL` 显式指定，原版本校验仍然必须通过。失败现场默认读取：

```text
/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_MEMORY_JOINT_V1/run_20261013_143518_3825644
```

## 报告如何解释

| 结果 | 下一步判断 |
| --- | --- |
| CPU 快照索引非法或非有限 | 保留现场，先查 Q、ID 或输入来源；停止 GPU 对照 |
| CUDA 原实现失败，连续索引通过 | 布局／原 CUDA 路径成为候选原因；还需候选完整流程和库／查询一致性验证，不能直接宣称修复 |
| 两种固定输入 GPU 重放都通过，单图前向失败 | 关注完整前端与上游计算，读取同步模式下的具体失败栈 |
| 单图通过，原顺序 Val450 失败 | 关注序列运行状态、生命周期、上游算子；保留新现场 |
| 所有诊断都通过 | 此诊断条件下未重现；不能认定间歇性原故障已解决，尤其同步改变了执行时序 |

优先提供新结果中的 `summary.json`（各组 PASS/FAIL）、`historical_evidence.json`，以及失败组 `.log` 末尾。
`diagnostic_complete=true` 表示计划的诊断执行结束；不表示每组都通过。`all_executed_stages_passed` 与每组 `passed` 单独记录。
初始 Val450 检查完成也始终不是训练后正式成绩，顶层 `full_validation_evidence=false`、`training=false`。

## 核心代码

- `pool_ops.py`：原池化算法、连续索引候选、布局恢复和比较。
- `replay_worker.py`：单个快照的 CPU / CUDA 独立重放。
- `sequence_worker.py`：原训练包的完整前向与初始 Val 复现，不训练。
- `diagnose.py`：失败现场读取、子进程编排和报告。
- `start_server.sh`：从 New 目录后台启动。

CPU 合成测试使用：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -m unittest discover -s /你的解压目录/joint_pool_diagnose_v1 -p 'test_*.py' -v
```

云端没有真实 BRIGHT 数据、服务器失败快照或 CUDA。CPU 测试只能验证诊断工具的逻辑、池化数学对照和保护行为；实际根因结论以服务器执行后的报告为依据。
