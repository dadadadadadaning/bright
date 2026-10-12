# Cluster Detail D0

该实验不训练模型，复用已经完成的 Cluster-E0 calibration 缓存，回答两个问题：

1. Cluster Oracle 收益集中在哪些面积和 P3→S2 修改方向；
2. 小修改簇是否在 Stage3/4 中只覆盖一个或两个 token，导致粗尺度表征碰撞。

## D0-A

报告七个面积段和全部修改方向的簇数量、corrected、harmed、net、Oracle拒绝数、相对 fixed S2 的可恢复 mIoU 及恢复比例。同时重建原 Cluster Oracle，防止统计口径漂移。

RGB角色分数使用已经冻结的 leave-one-event-out 策略，额外报告各面积段普通AUC和像素加权AUC，用来判断此前0.5729的加权AUC是否只来自大簇。

## D0-B

只在 meta_calibration 重新执行冻结P3和fixed S2，不读取GT，并精确记录每个修改簇覆盖的Stage2/3/4硬token数。最终与原缓存的image、region、cluster、direction、area逐项对齐。

还报告同一伪建筑内RGB/SAR角色特征的量化重复率，以及Change–Anchor余弦距离。

准入D1的主要条件是：Cluster Oracle空间仍大于0.10pp，并且Stage3单token比例不低于70%，或至多两个token比例不低于85%。

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash experiments/spi_cluster_detail_d0_v1/run.sh preflight
bash experiments/spi_cluster_detail_d0_v1/run.sh smoke
```

正式审计：

```bash
nohup env NUM_WORKERS=1 PROGRESS_EVERY=10 SEED=7341 \
bash experiments/spi_cluster_detail_d0_v1/run.sh audit \
> /root/data/BRIGHT-train-logs/SPI_CLUSTER_DETAIL_D0_seed7341.log 2>&1 &
```

查看：

```bash
tail -f /root/data/BRIGHT-train-logs/SPI_CLUSTER_DETAIL_D0_seed7341.log
```

结果：

```text
/root/data/BRIGHT-eval-results/SPI_CLUSTER_DETAIL_D0_V1/cluster_detail_d0_seed7341.json
```

如果输出 `ADMIT_CLUSTER_DETAIL_D1_STAGE2_EDGE_EVIDENCE`，下一步才实现并运行Stage2、RGB Scharr/LoG和SAR鲁棒纹理D1。若输出停止决定，则不能通过训练更大的Verifier绕过D0。
