# Cluster Outcome Utility E2.1

本实验停止直接回归事件相关的最终 mIoU/Damaged-F1 数值，改为预测修改簇内部的
`corrected / harmed / neutral` 像素比例。P3、LocCC、fixed S2 和 D1 特征全部冻结。

四个因果隔离臂：

- B1：共享三结果头，风险为 `P(harmed)-P(corrected)`；
- B2：加入方向专用特征斜率，风险定义不变；
- B3：用概率乘簇面积，重建预期混淆矩阵变化；
- B4：在 B3 上用预测熵进行保守置信降权（预注册主臂）。

训练使用按簇归一化的软标签交叉熵，因此大簇不会支配模型拟合。面积只在决策时转换
为预期像素数。每折模型只用另外 13 个事件的 meta_train，策略只用另外 13 个事件的
meta_calibration，held-out 事件仅评估一次。拒绝比例上限固定为 10%。

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash experiments/spi_cluster_outcome_utility_v1/run.sh preflight

nohup env \
LOEO_STEPS=1000 \
BATCH_SIZE=2048 \
LEARNING_RATE=1e-3 \
DIRECTION_SHRINKAGE=1e-3 \
MAXIMUM_REJECT_FRACTION=0.10 \
BOOTSTRAP_ITERATIONS=10000 \
SEED=7341 \
bash experiments/spi_cluster_outcome_utility_v1/run.sh auto \
> /root/data/BRIGHT-train-logs/SPI_CLUSTER_OUTCOME_E21_seed7341.log 2>&1 &
```

`auto` 会先生成局部 Outcome Oracle 审计。只有该理论上限通过，才自动进入 LOEO。
若审计因中断已经成功而 LOEO 尚未运行，使用 `run.sh loeo`，不要再次覆盖审计结果。

查看进度：

```bash
tail -f /root/data/BRIGHT-train-logs/SPI_CLUSTER_OUTCOME_E21_seed7341.log
```

结果：

```text
/root/data/BRIGHT-eval-results/SPI_CLUSTER_OUTCOME_UTILITY_E21_V1/cluster_outcome_e21_loeo_seed7341.json
```

运行前不要访问 meta_confirm、Val450 或 Test；本实验只复用已有 D1 meta_train 和
meta_calibration 缓存，不重新进行图像推理。
