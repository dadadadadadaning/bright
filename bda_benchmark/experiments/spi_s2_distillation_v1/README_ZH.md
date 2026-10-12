# SPI S2 Distillation V1

该实验只验证一个问题：固定 S2 已经能改善 P3，显式蒸馏 S2 的软区域行为，是否能让轻量学生超过普通语义 Adapter。

## 变体

- K0：冻结 P3。
- K1：固定、可部署 S2 强基线。
- K2：Local 残差学生，只用原语义目标与区域外锚定。
- K3：K2 加 S2 软结构蒸馏；K3-K2 是第一因果比较。
- K4：K3 的 Local 块换成修正的轻量 GBC。
- K5：K4 增加 LocCC 区域特征方差。

没有像素/区域 Oracle，没有官方实例，没有 Test。内部 Calibration291 通过后，脚本才会访问 Standard-ML Val450。

## 服务器运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash experiments/spi_s2_distillation_v1/run.sh preflight

SMOKE_STEPS=2 SMOKE_IMAGES=2 PROFILE_TIMING=1 \
NUM_WORKERS=4 bash experiments/spi_s2_distillation_v1/run.sh smoke

mkdir -p /root/data/BRIGHT-train-logs
nohup env PROFILE_TIMING=1 NUM_WORKERS=4 BATCH_SIZE=4 \
ACCUMULATION_STEPS=4 PILOT_STEPS=500 FULL_STEPS=3000 \
bash experiments/spi_s2_distillation_v1/run.sh auto \
> /root/data/BRIGHT-train-logs/SPI_S2_DISTILLATION_V1_seed3407.log 2>&1 &

tail -F /root/data/BRIGHT-train-logs/SPI_S2_DISTILLATION_V1_seed3407.log
```

训练分支共享一次冻结 P3 前向；区域统计位于低分辨率特征网格并使用 GPU `index_add_`。不会创建全量特征缓存。

注意：训练教师是在 P3 特征网格上执行的 S2 等价规则，以保证在线增强和4090吞吐；评估中的 K1 仍是原始全分辨率 S2。结果中若 K3 明显低于 K1，应首先检查低网格教师与全分辨率 S2 的一致率，而不是继续加注意力模块。
