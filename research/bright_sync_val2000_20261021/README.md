# 第2000步完整 Val450 同步验证

用途：前31张同步索引诊断通过后，用保存的第2000步检查点计算全部450张验证图。

- 固定父运行：`SRAC_SLOT_MEMORY_ABLATION_RESUME_V1/resume_20261021_112222_495713`。
- 固定原始父程序 SHA256：`029137749200f9b4f4d0d5d26e4015dcb5b43a4c3a667fb616c97b8d4fed3570`。
- 六个分支 E/F/B/D/R/L 和 Grid，使用原始验证顺序、前端、支持库、损失及评价实现。
- 启动子进程前设置 `CUDA_LAUNCH_BLOCKING=1`。实际设置及原设置同时写入 manifest/result，其他协议字段必须一致。
- 不启用 TorchDispatchMode，不逐算子跟踪、不额外复制索引张量。原始模型和校验器照常执行。
- 零优化器更新、不调用反向传播、不访问 Test；恢复优化器只是为了验证状态未变。
- 与前31张诊断一样，在恢复 RNG 前初始化剩余训练数据迭代器以保持该启动方式；不消费训练批次，不声称重现历史工作进程。
- 保留首次异常和调用栈；错误后不访问 CUDA RNG 或复制 GPU 张量。未覆盖的 loss/metric/loader 错误也保留原异常。
- 新结果存入 `SRAC_SLOT_MEMORY_ABLATION_EVAL_ONLY_V1/sync_val2000_*`，原运行、检查点和失败记录保持不变。
- 只有全部450张顺序、逐图目标和混淆矩阵证据、文件/状态/运行时检查通过，才完成新验证报告。CPU合成结果不能作为完整GPU验证证据。
- 共享已有运行锁与评估锁；不自动重试、不启动续训。

## 服务器命令

Python：`/root/shared-nvme/conda-envs/bright4090/bin/python`。

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_sync_val2000_v1.pyz start
```

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_sync_val2000_v1.pyz logs
```

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_sync_val2000_v1.pyz status
```

启动时输出新目录内固定副本的 logs/status 命令，可跨 SSH 会话使用。Ctrl+C 退出日志跟踪不停止后台验证。

成功标志：`phase=standalone_evaluation_complete`、`complete=true`、`image_count=450`、`full_validation_evidence=true`、`optimizer_updates=0`。原实验仍未完成2500步。

失败时停止，保留 `evaluation_failure.json`、`traceback.txt`、`summary.json` 与已完成图片日志。不要反复 start 覆盖问题；每次 start 都会创建一次新验证。

## 解释边界

前31张通过只说明该次同步、带逐算子观察的诊断未复现错误。本程序去掉逐算子观察，但同步设置与原始失败运行不同。完整通过可提供声明设置下的 Val450 结果，不能证明异步执行错误已修复，也不能把原失败运行改为成功。跨执行设置的微小指标差异仍需谨慎解释。

本地仅有 CPU 测试环境，未实测远端 RTX4090 数据。构建及测试记录见 `VALIDATION.md`。
