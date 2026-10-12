# SPI Risk Gate V1：D0–D5

## 研究问题

固定 P3 和当前 500 步 M4，只改变是否及在哪里采用 M4 残差：

```text
Z_g = Z_P3 + g(x) (Z_M4 - Z_P3)
```

这保证 D2、D3、D4 的差异只来自 Gate，而不是重新训练了不同候选模型。

| 名称 | 定义 |
|---|---|
| D0 | 冻结 P3 |
| D1 | 无门控 M4 |
| O1 | 标签感知 P3/M4 选择上限，仅诊断 |
| D2 | 固定 P3 熵 Gate，阈值0.5、温度0.1 |
| D3 | D2乘边界深度，半径2、边界下限0.2 |
| D4 | 10维风险输入的轻量可学习Gate |
| D5 | 原始 Selective Context S2 强基线 |

训练和评估都不读取官方实例；主评估仍是 Standard-ML Val450 语义 TIFF；不访问 Test。

## 自动节流顺序

1. 只推理 D0/D1/O1/D2/D3/D5。
2. O1至少提供+0.5pp上限、固定Gate接近P3且Harmed下降20%，才训练D4。
3. D4只在Train2918按事件分层得到的90%子集训练。
4. 先在内部10%校准集检查D4是否超过P3且Harmed少于D1。
5. 内部校准通过后，才对Val450进行一次D4最终审计。

## 4090效率设计

- P3和M4完全冻结，无主干反向图。
- 每个batch只做一次P3前向和一次M4小头前向。
- LocCC只在低分辨率特征图构建。
- Gate约一千参数，不生成特征缓存。
- 默认BF16、batch 4、梯度累计4、worker 4。
- 检查点每100步保存，断线可恢复。

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
bash experiments/spi_risk_gate_v1/run.sh preflight
bash experiments/spi_risk_gate_v1/run.sh smoke
```

确认 Smoke 后无人值守运行：

```bash
mkdir -p /root/data/BRIGHT-train-logs
nohup env NUM_WORKERS=4 BATCH_SIZE=4 ACCUMULATION_STEPS=4 \
  bash experiments/spi_risk_gate_v1/run.sh auto \
  > /root/data/BRIGHT-train-logs/SPI_RISK_GATE_V1_seed3407.log 2>&1 &
echo $!
```

只做无训练Stage-A：

```bash
bash experiments/spi_risk_gate_v1/run.sh stage-a
```

如果Stage-A仅因固定Gate略低于安全门槛而停止，但O1上限明确为正，可运行明确标记的探索模式。它不会篡改Stage-A结论，仍须先通过内部Calibration，之后才允许再次评估Val450：

```bash
bash experiments/spi_risk_gate_v1/run.sh explore-d4
```

## 默认路径

M4输入：

```text
/root/data/BRIGHT-checkpoints/SPI_CORE_RING_ADAPTER_V1/seed3407/pilot500/joint_last.pth
```

输出：

```text
/root/data/BRIGHT-eval-results/SPI_RISK_GATE_V1/
/root/data/BRIGHT-checkpoints/SPI_RISK_GATE_V1/seed3407/
```

若M4实际路径不同，运行时设置：

```bash
M4_CKPT=/实际路径/joint_last.pth bash experiments/spi_risk_gate_v1/run.sh auto
```
