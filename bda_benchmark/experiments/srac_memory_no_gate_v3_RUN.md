# SRAC Memory 建库与检索训练：服务器运行顺序

这次只做两项**独立**改动：①同容量、同事件×类别配额的案例覆盖选样；②从 Train 中挖掘被旧检索误导的 Intact，等预算训练新的检索投影。前端、Q、J1/J2/J3、原 Reader 和旧权重均不更新；没有采纳／拒绝门控训练。两项改动先各自与旧 J3 比较，不把成绩相加。

所有命令从 `/root/shared-nvme/BRIGHT-master` 执行。下面的 E0 路径必须是已通过 `legacy/native`、完整 Val450、`1e-5` 原复现检查的目录。每个输出目录必须是**新路径**。

```bash
cd /root/shared-nvme/BRIGHT-master
export PYTHON=/root/shared-nvme/conda-envs/bright4090/bin/python
export E0_ROOT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_HARDQUERY_MEMORY_V1/formal_legacy_20261010_204130
export E0_PROTOCOL="$E0_ROOT/protocol"
unset MATMUL_TF32 CUDNN_TF32 DETERMINISTIC
```

## A. 同容量案例覆盖实验：不训练任何权重

```bash
export BANK_EXP=bda_benchmark/experiments/damageformer_srac_memory_bank_diverse_v1
bash "$BANK_EXP/run.sh" preflight
export BANK_DIV_OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_BANK_DIVERSE_V1/bank_$(date +%Y%m%d_%H%M%S)
bash "$BANK_EXP/run.sh" build
cat "$BANK_DIV_OUTPUT/summary.json"
export BANK_DIV_VAL_OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_BANK_DIVERSE_V1/val_$(date +%Y%m%d_%H%M%S)
bash "$BANK_EXP/run.sh" evaluate
cat "$BANK_DIV_VAL_OUTPUT/summary.json"
bash "$BANK_EXP/run.sh" report
```

这次 Val 同次前向比较旧库与新覆盖库，以及原 Top‑8 与来源多样 Top‑8。`build` 输出 `complete=true` 只表示 Train 建库完成；只在 Val `summary.json` 的 `complete`、`full_validation_evidence`、`reproduction_passed` 全为 `true` 时解释正式成绩。原始容量和事件×主类配额保持一致，具体见[建库说明](../bda_benchmark/experiments/damageformer_srac_memory_bank_diverse_v1/README.md)。

## B. 检索投影训练：固定原库和 Reader

先从已通过 E0 的原运行填写**相同的**权重和数据路径（不匹配会被来源校验拒绝），并确认是否已有与这个 E0 匹配的完整 Train2918 查询缓存：

```bash
export JOINT_RUN=/root/shared-nvme/BRIGHT-checkpoints/SRAC_SNAPSHOT_JOINT123_V1/run_20261009_191815/joint
export A1_CKPT=/root/shared-nvme/BRIGHT-checkpoints/best/best_val_miou.pth
export PROFILE=legacy POOL_MODE=native
export QUERY_RUN="$E0_ROOT/queries"
test -f "$QUERY_RUN/query_manifest.json" && echo 'Train query cache exists'
```

若没有，且 `$QUERY_RUN` 尚未创建，请先运行原实验的 `queries` 阶段；它只读 Train 图像，不训练 Reader／门控。若目录存在但不完整，改用一个全新的 `QUERY_RUN` 路径，保留失败证据。详见[原查询缓存说明](../bda_benchmark/experiments/damageformer_srac_hardquery_memory_v1/README.md)。

```bash
export RUN_ROOT="$E0_ROOT"
export PROTOCOL_RUN="$E0_PROTOCOL"
bash bda_benchmark/experiments/damageformer_srac_hardquery_memory_v1/run.sh queries
```

查询缓存完成后训练检索投影：

```bash
export METRIC_EXP=bda_benchmark/experiments/damageformer_srac_memory_metric_v2
bash "$METRIC_EXP/run.sh" preflight
export METRIC_V2_OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_METRIC_V2/fit_$(date +%Y%m%d_%H%M%S)
bash "$METRIC_EXP/run.sh" train
cat "$METRIC_V2_OUTPUT/summary.json"
```

两组投影都进行 2000 次更新：`j1_hard` 是原 J1 困难查询抽样，`mislead` 则侧重真 Intact／J1 正确但旧 Top‑8 标签偏 Damaged，以及 J1 漏判的真 Damaged。它们共享支持子集、软标签损失、优化器和预算。`summary.json` 的 `mining` 应先检查两种目标查询数量；代码不会悄悄用 Val 样本填充。[训练细节](../bda_benchmark/experiments/damageformer_srac_memory_metric_v2/README.md)

训练通过后，在**原库、同一冻结 J3 Reader**上验证两组新投影：

```bash
export METRIC_EVAL_EXP=bda_benchmark/experiments/damageformer_srac_memory_metric_eval_v2
bash "$METRIC_EVAL_EXP/run.sh" preflight
export METRIC_V2_VAL_OUTPUT=/root/shared-nvme/BRIGHT-checkpoints/SRAC_MEMORY_METRIC_V2/val_$(date +%Y%m%d_%H%M%S)
bash "$METRIC_EVAL_EXP/run.sh" evaluate
cat "$METRIC_V2_VAL_OUTPUT/summary.json"
bash "$METRIC_EVAL_EXP/run.sh" report
```

评价旧 `j3`、`j3_j1_hard` 和 `j3_mislead` 的四类 IoU、前景 mIoU、Damaged 救回、Intact→Damaged 误改、逐事件结果与邻居软标签相容性。[评估细节](../bda_benchmark/experiments/damageformer_srac_memory_metric_eval_v2/README.md)。只在完整 Val450、旧五臂 `1e-5` 复现和不变性检查通过后报告成绩；任何调参只用 Train/Val，已看过的 Test878 不再用于选择方案。

新包的本地检查只是 CPU 合成测试，不能替代服务器 Train2918 建库／检索训练与正式 Val450；目前没有这两项的新真实性能结果。不要将区域标签解释成逐栋损毁比例，也不要将训练完成解释成 Memory 已提分。
