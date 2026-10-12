# MACU：mIoU对齐的修改簇效用

## 目的

旧标签 `corrected-harmed` 优化像素正确数，但最终评价是类别均衡 mIoU。MACU利用 meta_train 的 fixed-S2 混淆矩阵冻结 mIoU 梯度，并把每个 `area>32` 修改簇的混淆矩阵增量转换为指标边际效用。

本实验直接复用 `SPI_CLUSTER_DETAIL_D1_V1` 的缓存，不重新运行P3，不读取meta_confirm、Val450或test。

## 隔离阶段

1. `build_targets`：只用meta_train冻结指标敏感度，为train/calibration生成标签；
2. `audit_oracle`：比较旧pixel-net Oracle、线性mIoU Oracle和精确边际Oracle；
3. `train_verifier`：Oracle通过后才训练默认线性Verifier；
4. Calibration只选择一个接受阈值，输出冻结检查点；
5. 只有`FREEZE_MACU_E1_FOR_META_CONFIRM`才允许编写并运行一次性confirm。

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash experiments/spi_metric_aligned_cluster_utility_v1/run.sh preflight

nohup env \
MODEL_TYPE=linear \
VERIFIER_STEPS=1000 \
VERIFIER_BATCH_SIZE=2048 \
BOOTSTRAP_ITERATIONS=10000 \
SEED=7341 \
bash experiments/spi_metric_aligned_cluster_utility_v1/run.sh auto \
> /root/data/BRIGHT-train-logs/SPI_MACU_V1_seed7341.log 2>&1 &
```

第一轮固定使用linear。只有linear显示正增益但表达能力不足时，才把`MODEL_TYPE=mlp`作为预注册的第二阶段运行，不能同时试很多隐藏维度。

## 一次性 meta_confirm

仅当校准结果为 `FREEZE_MACU_E1_FOR_META_CONFIRM` 时运行。该入口自动从
`meta_train.pt` 继承 D1 的最小建筑/修改簇面积、通道数、匹配窗口、halo 和种子，
固定使用已保存的线性模型与阈值，不重新训练或选择阈值。

```bash
nohup env \
PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark \
MODEL_TYPE=linear \
VERIFIER_STEPS=1000 \
NUM_WORKERS=1 \
BOOTSTRAP_ITERATIONS=10000 \
SEED=7341 \
bash experiments/spi_metric_aligned_cluster_utility_v1/run.sh confirm \
> /root/data/BRIGHT-train-logs/SPI_MACU_E1_CONFIRM_seed7341.log 2>&1 &
```

确认结果写入
`/root/data/BRIGHT-eval-results/SPI_METRIC_ALIGNED_CLUSTER_UTILITY_V1/macu_e1_confirm_seed7341.json`。
若结果文件已存在，程序拒绝再次评估。确认准入条件固定为：相对 fixed S2
提升至少 `0.30pp`、Oracle 恢复率至少 `25%`、bootstrap 置信区间下界大于
零、Damaged F1 不下降且至少 `10/14` 事件不下降。
