# Multi-Prototype Selective Context V1

每个LocCC不再只有一个区域均值，而是最多保留I、D、X三个由高置信像素形成的类别原型。只有同时存在至少两个有效原型的区域才启用Multi-Prototype；单一状态区域继续由锁定的S2处理。

| 变体 | 含义 |
|---|---|
| M0 | 原始P3 |
| M1 | 锁定的Selective Context S2 |
| M2 | 概率兼容性的多原型上下文 |
| M3 | M2加入全局特征原型匹配 |
| M4 | M1处理单状态区域，M3处理多状态区域 |
| M5 | M4中只修改高不确定且原型分配间隔足够的像素 |

服务器：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
bash experiments/spi_multi_prototype_context_v1/run.sh preflight
SMOKE_IMAGES=10 PROFILE_TIMING=1 NUM_WORKERS=4 bash experiments/spi_multi_prototype_context_v1/run.sh smoke
```

无人值守先Calibration291、通过才Val450：

```bash
mkdir -p /root/data/BRIGHT-train-logs
nohup env PROFILE_TIMING=1 NUM_WORKERS=4 BOOTSTRAP_ITERATIONS=10000 \
bash experiments/spi_multi_prototype_context_v1/run.sh auto \
> /root/data/BRIGHT-train-logs/SPI_MULTI_PROTOTYPE_CONTEXT_V1_seed3407.log 2>&1 &
echo "PID=$!"
```

```bash
tail -F /root/data/BRIGHT-train-logs/SPI_MULTI_PROTOTYPE_CONTEXT_V1_seed3407.log
```
