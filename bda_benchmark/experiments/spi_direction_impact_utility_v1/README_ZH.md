# Direction–Impact Utility E0/E1

本实验解释为什么 B1 Benefit-only 只在部分灾害事件有效。它不训练最终模型，也不访问
`meta_confirm`、Val450、官方实例或 Test。

## E0：Event-Shift Audit

严格重放原 V2.1 的 14 折 LOEO，并保存每个修改簇的 OOF Benefit/Harm 分数。随后按事件报告：

- P3、fixed S2、B1 与精确单簇 Metric Oracle；
- Benefit AUROC/AUPRC、效用加权 AUROC；
- 分数分位数、簇接受率、像素接受率；
- B1 对可用 Oracle 空间的恢复比例。

OOF 重放必须复现原 V2.1 的 B1 总体增益，否则审计拒绝继续。

## E1：Exact Utility / Risk–Impact Audit

对每个 held-out 事件，以该事件 fixed-S2 混淆矩阵计算精确单簇效用：

```
u(C) = metric(H_s2) - metric(H_s2 - delta_C)
```

比较一阶标签与精确标签的符号一致率，并报告普通 AUC 与 `|utility|` 加权 AUC。
风险覆盖曲线按每个 held-out 事件内部的 Harm OOF 排名，固定拒绝最高
0.5%、1%、2%、5%、10%、20%、50%，不根据结果选择比例。

## 科学边界

- 精确效用和 Oracle 只用于诊断，不能进入推理模型；
- held-out 事件不参与模型训练或阈值选择；
- 当前实验不得产生 Val450/Test 预测；
- E0/E1 通过后才允许创建 Direction、Impact Weighting 或 Conservative Rejection 训练版本。

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash experiments/spi_direction_impact_utility_v1/run.sh preflight

nohup env SEED=7341 LOEO_STEPS=1000 BATCH_SIZE=2048 \
  bash experiments/spi_direction_impact_utility_v1/run.sh auto \
  > /root/data/BRIGHT-train-logs/SPI_DIRECTION_IMPACT_UTILITY_V1_seed7341.log 2>&1 &
```

如需覆盖特征组，使用实验专用变量 `UTILITY_GROUPS`。脚本不会读取通用的
`GROUPS` 环境变量，以免其他实验遗留的 `GROUPS=0` 污染本实验。
