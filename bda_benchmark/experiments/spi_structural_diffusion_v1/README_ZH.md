# SPI Structural Diffusion V1：N0–N5

## 一句话说明

LocCC只提供传播边界，不提供统一类别。每个像素始终保留自己的I/D/X概率；相邻像素仅在属于同一LocCC时交换证据。

```text
错误的硬一致性：一个LocCC -> 一个类别 -> 整区覆盖
本实验：        一个LocCC -> 多个像素分布 -> 受约束的邻接传播
```

## 变体

| 变体 | 唯一变化 |
|---|---|
| N0 | 原始冻结P3 |
| N1 | 已验证的Selective Context S2 |
| N2 | LocCC内四邻域均匀传播 |
| N3 | N2加入融合特征相似性边权 |
| N4 | N3加入RGB/SAR特征分歧抑制 |
| N5 | N4加入P3语义概率边界阻断 |

N2–N5都在P3特征网格上向量化计算，只对LocCC内部回写概率残差。无训练、无真实实例、无Test、无磁盘特征缓存。

## 核心公式

对同一LocCC内四邻域像素：

```text
w_xy = same_CC(x,y) * feature_similarity * modality_reliability * semantic_similarity
q_x(t+1) = (1-alpha_x) p_x + alpha_x sum_y normalize(w_xy) q_y(t)
```

N2依次把后三项设为1；N3、N4、N5逐项打开，从而可以严格归因。

## 服务器运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash experiments/spi_structural_diffusion_v1/run.sh preflight

SMOKE_IMAGES=10 PROFILE_TIMING=1 NUM_WORKERS=4 \
bash experiments/spi_structural_diffusion_v1/run.sh smoke
```

Smoke确认P3、速度和显存正常后，先在Train2918预先划出的内部Calibration291筛选：

```bash
NUM_WORKERS=4 PROFILE_TIMING=1 \
bash experiments/spi_structural_diffusion_v1/run.sh calibration
```

只有内部结果显示某个N2–N5超过N1，才运行Val450：

```bash
mkdir -p /root/data/BRIGHT-train-logs
nohup env PROFILE_TIMING=1 NUM_WORKERS=4 BOOTSTRAP_ITERATIONS=10000 \
bash experiments/spi_structural_diffusion_v1/run.sh audit \
> /root/data/BRIGHT-train-logs/SPI_STRUCTURAL_DIFFUSION_V1_seed3407.log 2>&1 &
echo "PID=$!"
```

查看：

```bash
tail -F /root/data/BRIGHT-train-logs/SPI_STRUCTURAL_DIFFUSION_V1_seed3407.log
```

也可以使用无人值守入口；Calibration失败会自动停止且不访问Val450，通过后才自动接续：

```bash
nohup env PROFILE_TIMING=1 NUM_WORKERS=4 BOOTSTRAP_ITERATIONS=10000 \
bash experiments/spi_structural_diffusion_v1/run.sh auto \
> /root/data/BRIGHT-train-logs/SPI_STRUCTURAL_DIFFUSION_V1_auto_seed3407.log 2>&1 &
```

正式结果只有在P3重新得到60.5020%时才有效。最终检查会把N2–N5中的最好方法同时与P3和N1比较；若不能超过N1至少0.2pp，则保留N1并停止该扩散版本。
