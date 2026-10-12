# Cluster Detail D1：高影响修改簇的高分辨率证据

## 实验问题

D0 已证明：`area > 32` 的修改簇只占约 1.7%，却覆盖约 86.1% 的 Cluster Oracle 空间；Stage3/4 对绝大多数修改簇只有一个 token。因此 D1 只审计这些高影响簇。

流程保持：`P3冻结 -> fixed S2提出修改 -> D1判断大簇接受S2或回退P3`。面积不超过32的簇始终保留 fixed S2。

## E0证据组

- `spatial`：面积、方向、P3置信度控制组；
- `stage2_role`：Stage2 的修改区/稳定区/邻域角色池化；
- `rgb_edge`：原图分辨率 Scharr、LoG、局部对比度；
- `sar_texture`：log-SAR、方向梯度、LoG、局部中位残差与MAD；
- `stage2_direct`：RGB/SAR直接差负对照；
- `stage2_matching`：Stage2局部匹配；
- `structure_relation`：分别计算两种模态内部的修改区—稳定区关系，再比较关系差；
- `shuffled_*`：同建筑、同面积随机位置对照。

E0必须同时满足 AUC、pixel-weighted AUC、事件宏平均、事件方向、置换检验和随机位置对照。否则禁止训练E1。

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash experiments/spi_cluster_detail_d1_v1/run.sh preflight
bash experiments/spi_cluster_detail_d1_v1/run.sh smoke

nohup env NUM_WORKERS=1 SEED=7341 \
  bash experiments/spi_cluster_detail_d1_v1/run.sh auto \
  > /root/data/BRIGHT-train-logs/SPI_CLUSTER_DETAIL_D1_seed7341.log 2>&1 &
```

`auto` 会先构建 meta_train/meta_calibration 缓存并运行无参数 E0。只有 E0 通过时才训练单一两层验证器。它不会读取 meta_confirm、Val450、官方实例或 test。

## E1含义

E1从完整 fixed S2 开始，仅回退被判定有害的 `area > 32` 修改簇。Calibration 只作为筛选；输出 `FREEZE_CLUSTER_DETAIL_D1_FOR_META_CONFIRM` 后，才应另行实现一次性 meta_confirm 评估。
