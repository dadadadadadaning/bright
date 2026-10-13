# BRIGHT：诊断通过后启动原三组联合训练

本启动器对应已完成的诊断：

```text
/root/shared-nvme/BRIGHT-checkpoints/SRAC_JOINT_POOL_DIAG_V1/run_20261013_150401_3833455
```

该次 CPU / 原 CUDA / 连续索引候选 / 失败单图 / 原完整初始 Val450 全部通过。
原完整前向是在 `CUDA_LAUNCH_BLOCKING=1` 下通过的。它不证明此前异步故障的根因，也不证明连续索引候选更好。

下一步保留 **原 native 池化、原 Q、原冻结前端、现有 FP32 库**，在相同同步设置下启动新实验：

```text
预检 → 真实20步smoke → 重新初始化R0/RT/RF → 每组5000步 → 保存/重载 → 末步Val450
```

原失败结果不续训、不覆盖。前面的初始 Val450 不是训练后成绩，新训练不能直接跳过初始化检查。

## 放置与运行

上传 ZIP 至：

```text
/root/shared-nvme/BRIGHT-master/New/bright_joint_sync_launcher_v1.zip
```

按发布消息中的 SHA256 校验后解压到 `New` 下的新目录。启动方式：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python /解压目录/joint_sync_launcher_v1/start_joint_sync.py
```

可先加 `--dry-run`：只校验并打印计划，不创建训练结果或进程。

本脚本调用已存在且未修改的原训练包：

```text
/root/shared-nvme/BRIGHT-master/New/joint_train_ZAtNI2Wi/region_memory_joint_train_v1
```

它校验五组诊断和完整 Val450 的通过字段、原生池化后端、原代码校验、旧配置及初始头检查；显式传递旧配置，并在启动新 Python 前设置 `CUDA_LAUNCH_BLOCKING=1`、恢复旧 CUBLAS 配置。终端遗留的 `JOINT_*` 参数不会改变实验。原包仍负责所有来源、库、模型和训练终检。

脚本直接创建独立后台会话，不需要额外 nohup。只启动一次；启动 PID 不等于 smoke 或训练完成。

## 输出与查看

新结果目录：

```text
/root/shared-nvme/BRIGHT-checkpoints/SRAC_REGION_MEMORY_JOINT_SYNC_V1/run_<UTC时间>_<PID>
```

启动时打印精确日志与读取报告命令。在同一父目录的 `run_*.launch.json` 中另存诊断 SHA、原工具校验、命令和同步配置。
**launch.json 是启动时的固定记录，不是实时进度。** 进度看 `workflow_state.json`、`train/summary.json`，末步分数看 `train/final_val/result.json`。

主要比较：

- R0 − 本次 grid_probq：新图内联合分类器的效果。
- RT − R0：T128 找到的历史案例相对图内参考的额外效果。
- RF − R0：F256 找到的历史案例相对图内参考的额外效果。
- RF − RT：两种检索表示的差异。

统一使用完整 Val450 的前景三类 mIoU，并查看各类别及事件表现。本次不使用 Test。
同步执行可能增加耗时，不能用诊断速度承诺5000步总耗时。如果再次报错，原包停止并保留现场；本启动器没有自动重试、跳过检查或替换池化实现。

代码测试使用合成 JSON 和模拟子进程，没有云端真实 BRIGHT 或 CUDA 训练。真实完整初始 Val450 通过来自用户上传的服务器日志；当前没有新的5000步训练后成绩。
