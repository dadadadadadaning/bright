# Core–Ring Utility E0：建筑内部S2效用空间审计

## 研究问题

PI-M0已经得到 `STOP_PI_MEMORY_NO_CROSS_EVENT_UTILITY_SIGNAL`。本实验不再检索其他建筑，只回答：

> fixed S2修对和改坏的像素是否在LocCC内部具有稳定的核心/边界分布？

P3和fixed S2完全冻结，不训练模型，不访问`meta_confirm`、Val450、官方实例或Test。

## 分区

每个LocCC使用4连通、1像素腐蚀得到`core`，其在建筑内部的补集为`ring`。腐蚀后少于4像素的区域标记为无有效core，整区作为ring。`ring`不是建筑外部扩张区域。

## 无参数预测与对照

| 名称 | 定义 | 目的 |
|---|---|---|
| P3 | 原预测 | 基线 |
| fixed S2 | 整个LocCC使用S2 | 当前候选基线 |
| core_only | 只在core接受S2 | 检验有害修改是否集中边界 |
| ring_only | 只在ring接受S2 | 检验相反假设 |
| random_*_only | LocCC内同面积随机区域接受S2 | 排除收益仅来自降低覆盖率 |
| Building Oracle | 每栋建筑二选一 | 已通过的理论上限 |
| Core-Ring Oracle | core/ring分别二选一 | 判断细分决策是否还有空间 |
| Pixel Oracle | 每像素二选一 | 最终受限上界 |

`core_only`和`ring_only`只在`meta_train`中选择一个，`meta_calibration`只正式评估被选择的策略。

## 结果解释

- `ADMIT_FIXED_CORE_RING_POLICY_CONFIRM`：固定空间策略超过S2和同面积随机对照；冻结后才能读取`meta_confirm`。
- `ADMIT_CORE_RING_E1_LOW_CAPACITY_VERIFIER`：固定策略不足，但Core-Ring Oracle和预测证据均通过，可以实现小型zone verifier。
- `STOP_CORE_RING_NO_PREDICTIVE_SIGNAL_TRY_RGID_E0`：细粒度Oracle空间存在，但仍无法预测；下一步检查LocCC粘连/碎裂。
- `STOP_CORE_RING_CEILING_LOW_TRY_RGID_E0`：core/ring细分本身空间不足；转向RGID-E0。

## 服务器运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark

bash experiments/spi_core_ring_utility_v1/run.sh preflight
bash experiments/spi_core_ring_utility_v1/run.sh smoke

nohup env \
NUM_WORKERS=4 \
CORE_RADIUS=1 \
MINIMUM_CORE_AREA=4 \
SEED=7341 \
bash experiments/spi_core_ring_utility_v1/run.sh audit \
> /root/data/BRIGHT-train-logs/SPI_CORE_RING_UTILITY_E0_seed7341.log 2>&1 &
```

查看：

```bash
tail -f /root/data/BRIGHT-train-logs/SPI_CORE_RING_UTILITY_E0_seed7341.log
```

结果：

```text
/root/data/BRIGHT-eval-results/SPI_CORE_RING_UTILITY_V1/core_ring_e0_seed7341.json
```

不要同时试多个`CORE_RADIUS`并根据`meta_calibration`选择最好结果。首轮锁定`radius=1`；只有首轮结论明确且重新登记协议后，才能把radius作为新实验变量。

