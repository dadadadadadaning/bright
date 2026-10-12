# S2 Modification-Pattern E0 / E1

本实验建立在 `P3 + LocCC + A6 + fixed S2` 上。Strip-RGID 的 Split Oracle 只有约 `+0.000011pp`，因此不再优化分裂，而是直接判断 fixed S2 对一栋建筑提出的修改是否具有可靠的空间组织。

## E0证据组

- baseline：面积、形状、changed fraction、P3 entropy/margin；
- topology：修改簇数量、最大簇比例、尺寸熵、紧致度和离散度；
- distance：到LocCC边界的连续距离分布，不再使用固定Core/Ring二分；
- direction：`Intact→Damaged`等方向与簇/位置联合；
- probability：修改前margin和修改概率增量；
- shuffled topology/distance：保持修改数量不变的空间随机对照。

只有 topology/distance/direction 至少一组同时达到AUROC、事件稳定性、对照增益和置换检验门槛，才允许训练E1。

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash experiments/spi_s2_modification_pattern_v1/run.sh preflight
bash experiments/spi_s2_modification_pattern_v1/run.sh smoke

nohup env NUM_WORKERS=2 STATE_EVERY=100 SEED=7341 \
  bash experiments/spi_s2_modification_pattern_v1/run.sh audit \
  > /root/data/BRIGHT-train-logs/SPI_SMP_E0_seed7341.log 2>&1 &
```

仅当E0输出 `ADMIT_SMP_E1_SINGLE_VERIFIER`：

```bash
nohup env VERIFIER_STEPS=1000 VERIFIER_HIDDEN=32 SEED=7341 \
  bash experiments/spi_s2_modification_pattern_v1/run.sh train \
  > /root/data/BRIGHT-train-logs/SPI_SMP_E1_seed7341.log 2>&1 &
```

E1只读取meta_train；阈值由meta_train内部固定20%图像留出确定。训练完成只冻结模型，不读取meta_confirm、Val450或test。

