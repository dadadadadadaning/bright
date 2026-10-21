# 从已核验的 2100 步恢复六分支实验

该版本针对 `finish2500_a4x9h5ay` 的 `update_2136` 失败。E 在该步已经完成优化器更新；F 在 `model.forward` 的父区域 `validate_routes(q, ids, ...)` 入口失败。不能把 E 单独更新过的状态当作六分支共同检查点。

## 已知与未知

- 日志最后完整配对更新为 2135，但最新持久化完整检查点是 2100。
- 2100 检查点 SHA-256 为 `da8474ca8ba741aa10a0b4a5f3278e0d69a05b803c67c7c6cd9cf910572027f1`；所有分支优化器计数均为 2100。
- 该次失败发生于父区域 Q，而非子槽插值路由。组合错误文字本身不能区分负值和归一化超限。
- E 检查通过而随后 F 检查失败，值得核查共享输入是否变化及原始 GPU 检查结果；现有日志尚不能确定根因。
- 成功的同步 Val2000 证据继续单独保留；没有把 2135 当作已验证或已保存检查点。

## 运行流程

1. 校验原始代码、数据清单、已完成 Val2000、失败记录、2100 检查点字节及载入后的完整状态。
2. 在新的 `recover2100_*` 目录恢复六个头、各自 AdamW 状态、RNG 和相同数据顺序；仅复制日志的完整前 2100 步。
3. 重算 2101–2135，并执行至 2500，共 400 个配对更新。原本 E 的单独第 2136 次更新不继承。
4. 每 10 个完整配对更新保存检查点。失败步骤不会写成完成步骤，也不会保存部分分支作为共同检查点。
5. 训练进程退出后，分别启动固定 Train2000、Val450@2500、固定 Train2500 和 Test878@2500 的独立评估进程。评估使用已有同步 Val2000 成功方案的 `CUDA_LAUNCH_BLOCKING=1`，明确记录该设置。
6. 仅在前序训练、完整 Val 和固定 Train 检查通过后访问 Test。所有六分支用固定 2500 步；Test 不参与配置、检查点选择和训练损失。由于已有 Test 暴露，结果仍声明为探索性。

原模型、支持候选构造、NLL/Lovasz、学习率调度、样本顺序与原路由检查代码保留。没有归一化、裁剪、跳过图片或自动重试。原失败目录及其文件不会改写。

## 观察的内容与局限

`parent_q_observer.py` 在每次支持构造、前向、损失、反向开始、优化器更新前后读取 Q 和 IDs 的独立 CPU 副本，逐字节比较该图片内冻结输入。记录原张量存储、版本、哈希、负值/非有限数量、CPU float64 求和偏差。禁止 Q/IDs 与可写头参数、优化器状态共享存储。

发现变化立即保留变化前后 CPU 张量并终止该配对更新。原检查器仍决定路由合法性；额外快照统计不用于放宽接收条件。`PassiveRoutes` 保留原检查代码和布尔/求和调用顺序，失败时记录原始检查操作数；CUDA 异常后不发起新的 GPU 诊断读取。

**CPU 拷贝会同步 GPU 并改变执行时序，可能减慢训练。此版本是带观察的恢复实验，不能称为历史故障的精确重放，也不能把成功完成解释为根因已经修复。** 在本地 CPU 上的更新等价测试不等于 CUDA 数值轨迹等价证明。

新运行的 `request.json`、训练 `manifest.json`/`summary.json` 及 `workflow_summary.json` 均记录此执行变化。最终完成以 `workflow_summary.json` 的 `complete=true`、`training_complete=true`、`validation_complete=true`、`test_complete=true` 为准；训练子阶段 summary 保留训练完成、评估待执行的历史记录。

## 启动与查看

使用发布的 `bright_qwatch_recover2100_v1.pyz`，先核对同目录 SHA-256 文件，再执行：

```bash
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_qwatch_recover2100_v1.pyz start
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_qwatch_recover2100_v1.pyz logs
/root/shared-nvme/conda-envs/bright4090/bin/python -B /tmp/bright_qwatch_recover2100_v1.pyz status
```

启动器使用训练/评估锁，并将自身固定复制到新运行的 `trainer.pyz`。`logs` 中 Ctrl+C 仅退出日志跟随，不会终止后台训练。跨 SSH 会话不依赖临时环境变量。避免重复执行 start 来重试同一失败。

主要输出：

- `workflow.log`、`workflow_summary.json`：整个流程。
- `summary.json`、`train.jsonl`、`markers.jsonl`：配对训练与分支执行位置。
- `parent_q_observations.jsonl`、`parent_q_observer_summary.json`：逐阶段冻结输入检查。
- 若输入变化：`parent_q_change.json`、`parent_q_change_cpu.pt`。
- 若训练阶段异常：`parent_q_failure.json`、`parent_q_before_cpu.pt`、`traceback.txt`。
- 若原路由校验失败：`route_failure.json`、`route_failure_cpu.pt`，含原检查操作数；非路由异常可能没有这些文件。
- `checkpoints/`：新的完整配对检查点。
- `assessments/`、`metrics.csv`：完成后各独立评估和分 split 的指标。

前置检查失败不会开始训练。运行中若再次失败，查看本次新目录的失败信息和最近检查点，不能把旧日志末尾或 `PID` 当作完成证明。

## 源码与构建

基础档案为 `bright_finish2500_test_v1.pyz`，SHA-256 `cb8a94d4235d2f0ce70f7c70c9d8e1c32285ba7a7a8faf58eceaf472b9e624e2`。全部旧模块保持字节不变，`QWATCH_BASE.json` 保存逐文件摘要。新增入口、恢复训练模块和父 Q 观察模块；入口改为新恢复流程。

`recover_training.py` 与原 `train_to_endpoint.py` 的差异限制为观察声明、将评估参考节点保留为已认证的 2000 步、以及记录原始更新函数与观察模块的来源。检查点间隔由恢复 context 设置为 10。完整 CPU 验证及边界见 `VALIDATION.md`。
