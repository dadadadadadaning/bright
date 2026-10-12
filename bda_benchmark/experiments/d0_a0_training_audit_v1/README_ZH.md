# D0 与 A0 训练过程审计

本实验只读取现有训练日志、检查点和验证集，不访问官方测试集。

## 训练期间：生成完整曲线

```bash
bash experiments/d0_a0_training_audit_v1/run.sh curves
```

## A0 完成后：统一 Val450 重评估并比较

确保没有其他训练占用 GPU，再运行。程序会分别评估原生 D0、迁移后但未训练的
A0-step0，以及训练后的 A0：

```bash
bash experiments/d0_a0_training_audit_v1/run.sh evaluate
```

结果保存在：

```text
/root/data/BRIGHT-eval-results/D0_A0_TRAINING_AUDIT_V1/
```

其中包含训练/验证 CSV、汇总 JSON、Markdown，以及环境安装 matplotlib 时生成的 PNG 曲线。

## D0 与 A0 的一次性锁定 Test

只有 D0 和 A0 权重都已经冻结、不再根据 Test 调参时运行：

```bash
ALLOW_TEST_ACCESS=YES \
bash experiments/d0_a0_training_audit_v1/run.sh test_locked
```

结果写入：

```text
/root/data/BRIGHT-eval-results/D0_A0_TEST878_LOCKED_V1/
```

重复运行会保留已有的两个预测结果，只重新生成比较汇总，不会重复推理。
