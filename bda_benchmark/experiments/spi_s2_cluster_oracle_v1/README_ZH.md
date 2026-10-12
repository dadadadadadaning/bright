# S2 Modification-Cluster Oracle C0

本实验不训练模型。它只计算：把fixed S2的修改按“同一LocCC、相同P3到S2方向、8连通”划分为修改簇后，逐簇使用GT选择S2或P3，理论上能比Building Oracle提高多少。

## 五个对照

- P3：全部保留基础预测；
- fixed S2：全部接受候选修改；
- Building Oracle：每栋LocCC整体二选一；
- Cluster Oracle：每个方向一致的连通修改簇二选一；
- Pixel Oracle：每个changed pixel二选一。

Oracle的选择目标是`corrected-harmed > 0`，不是直接搜索mIoU，因此与既有Building Oracle定义一致。Cluster Oracle使用GT，只能作为上限，不能部署。

## 准入

默认要求Cluster Oracle相对Building Oracle：

- mIoU增加至少`+0.10pp`；
- 至少`10/14`事件不下降；
- 接受的正效用修改簇不少于200；
- 图像bootstrap 95% CI下界大于0。

通过输出`ADMIT_S2_CLUSTER_EVIDENCE_E0`，下一步才提取不依赖GT的修改簇空间与RGB-SAR证据。否则输出`STOP_S2_CLUSTER_GRANULARITY_CEILING_LOW`，停止细化选择粒度。

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash experiments/spi_s2_cluster_oracle_v1/run.sh preflight
bash experiments/spi_s2_cluster_oracle_v1/run.sh smoke

nohup env NUM_WORKERS=2 STATE_EVERY=50 SEED=7341 \
  bash experiments/spi_s2_cluster_oracle_v1/run.sh audit \
  > /root/data/BRIGHT-train-logs/SPI_S2_CLUSTER_ORACLE_C0_seed7341.log 2>&1 &
```

Smoke默认每张图打印一次进度。连通域只在每个LocCC的最小包围框内计算，避免对每栋建筑重复扫描整幅影像。

`audit`只处理263张meta_calibration，不读取meta_confirm、Val450、官方实例或test。不要与正在运行的SMP缓存并行，以免争用GPU和磁盘。
