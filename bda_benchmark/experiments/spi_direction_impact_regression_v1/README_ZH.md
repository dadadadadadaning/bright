# Direction-Conditioned Impact-Aware Utility E2

E2 使用相同低容量线性双头回归器做四臂因果对照：

| 变体 | 连续效用 | Direction one-hot | Impact weighting |
|---|---|---|---|
| C1 | 是 | 否 | 否 |
| C2 | 是 | 是 | 否 |
| C3 | 是 | 否 | 是 |
| C4 | 是 | 是 | 是 |

两个输出分别预测保留 fixed-S2 修改对 mIoU 和 Damaged F1 的精确单簇效用。
目标使用 source-train 中位数缩放的 signed-log 变换，避免长尾效用造成回归不稳定。

每折只用另外 13 个事件训练；阈值仅在另外 13 个事件的 meta-calibration
上选择。候选拒绝率固定为 1%、2%、5%、10%，且 Damaged F1 下降的候选不允许进入选择。

## 准入

C4 是预注册主版本，要求：

- 相对 fixed S2 mIoU 至少 +0.30pp；
- Damaged F1 不下降；
- 至少 10/14 事件 mIoU 与 Damaged F1 不下降；
- bootstrap 95% CI 下界大于 0；
- Harm utility-weighted AUROC 至少 0.60；
- Direction 和 Impact 的单项归因检查通过。

本实验只使用 meta_train/meta_calibration，禁止访问已消费的 meta_confirm、Val450和Test。

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash experiments/spi_direction_impact_regression_v1/run.sh preflight

nohup env LOEO_STEPS=1000 BATCH_SIZE=2048 MAXIMUM_REJECT_FRACTION=0.10 \
  bash experiments/spi_direction_impact_regression_v1/run.sh loeo \
  > /root/data/BRIGHT-train-logs/SPI_DIRECTION_IMPACT_E2_LOEO_seed7341.log 2>&1 &
```

