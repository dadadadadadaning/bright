# 独立建筑 Memory V1

目标不是再次证明 P3 特征可分类，而是检验原始/浅层 RGB、SAR 建筑卡在已知 P3 预测后是否仍提供独立信息。

顺序固定为：

1. `MI-D0`：P3、RGB、SAR、组合卡、P3+组合卡、打乱组合卡；
2. `MI-D1`：每事件×类别容量 32/64/128/256/512/全部，随机/纯度/K-center；
3. `MI-D2`：结构、SAR、关系、质量、角色 Token 逐项消融；
4. `MI-E1`：条件残差、多原型、跨事件对比训练；
5. `MI-E2`：真实/随机/打乱/自身近邻/零支持因果对照。

独立检索特征不读取 P3 融合特征或 P3 损毁概率。P3 只用于产生部署时可用的 LocCC 建筑掩膜；P3 概率只作为条件基线进入残差头。

服务器运行：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash experiments/spi_independent_building_memory_v1/run.sh preflight
bash experiments/spi_independent_building_memory_v1/run.sh smoke

nohup env NUM_WORKERS=1 SEED=7341 \
  bash experiments/spi_independent_building_memory_v1/run.sh auto \
  > /root/data/BRIGHT-train-logs/SPI_INDEPENDENT_BUILDING_MEMORY_V1_seed7341.log 2>&1 &
```

本实验只访问既有 `meta_train/meta_calibration`，不访问 `meta_confirm/Val450/Test`。

运行结束后只读复核：

```bash
bash experiments/spi_independent_building_memory_v1/run.sh verify
```
