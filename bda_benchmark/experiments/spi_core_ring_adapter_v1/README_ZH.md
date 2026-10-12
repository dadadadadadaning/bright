# SPI Core–Ring Adapter V1

## 研究问题

本实验只检验一个命题：在不使用真实实例标注的条件下，P3 预测的建筑区域能否通过区域上下文与核心—外环对比，提高原始语义损毁预测。

它不训练 RGID，不使用官方实例，不访问测试集，也不把 MICB、RCM、CAA、FreqFusion 同时混入第一轮。

## 可归因对照

| 变体 | 唯一作用 |
|---|---|
| M0 | 冻结 P3，锁定 60.5020% 基线 |
| M1 | 只微调损毁分类头，排除“多训练几步”解释 |
| M2 | LocCC 内使用局部 5×5 上下文 |
| M3 | 在完全相同 LocCC 支持内，将局部上下文换成区域均值 Token |
| M4 | 在相同 Adapter 和支持内，将区域均值换成核心 Token，并增加核心减外环对比 |

M2、M3、M4 的参数量、训练目标、支持区域、步数完全相同，因此 `M3-M2` 测量区域 Token 的贡献，`M4-M3` 测量 Core–Ring 对比的贡献。

## 方法

P3 的融合特征为 `F`，原始损毁 logits 为 `Z0`。对第 j 个 LocCC 区域：

```text
core_j = erode(R_j, r_c)
ring_j = dilate(R_j, r_o) - R_j
v_core = mean(F over core_j)
v_ring = mean(F over ring_j)
d_j = v_core - v_ring
```

同一个参数匹配 Adapter 计算：

```text
DeltaZ = Adapter(F, v, F-v, F*v, d)
Z = Z0 + 1[x in LocCC] * DeltaZ
```

输出层零初始化，所以训练开始时 M2–M4 与 P3 完全一致。训练阶段还在 P3 原本正确且置信度至少 0.8 的像素上施加小权重 KL 锚定，降低把正确预测改坏的风险。

## 为什么不会重现此前的速度/磁盘问题

- 不导出任何全量特征缓存；磁盘只保存小型 Adapter 检查点。
- P3 冻结；M1–M4 配对联合训练，同一个 batch 总共只做一次 P3 前向，而不是每个变体各做一次。
- 连通域在低分辨率 `p_building` 上计算，并先过滤小碎片。
- 每个区域只保存整数 label map，不生成“每区域一张全图 mask”。
- 区域均值使用 GPU `index_add_`，没有逐建筑全图循环。
- M1–M4 在一个进程中配对更新，不会启动并发训练；中断后会从最近保存步数继续。

## 运行

先检查环境：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
bash experiments/spi_core_ring_adapter_v1/run.sh preflight
```

2 步、2 张图烟雾测试：

```bash
bash experiments/spi_core_ring_adapter_v1/run.sh smoke
```

无人值守自动实验（推荐）：先配对训练 M1–M4 各 500 步并审计；只有全部门槛通过才自动启动独立的 3000 步确认实验。确认实验从 P3 重新开始，不继承筛选阶段权重，避免重复数据顺序污染最终结论。

```bash
mkdir -p /root/data/BRIGHT-train-logs
nohup env NUM_WORKERS=4 BATCH_SIZE=4 ACCUMULATION_STEPS=4 \
  bash experiments/spi_core_ring_adapter_v1/run.sh auto \
  > /root/data/BRIGHT-train-logs/SPI_CORE_RING_AUTO_seed3407.log 2>&1 &
echo $!
```

查看进度：

```bash
tail -F /root/data/BRIGHT-train-logs/SPI_CORE_RING_AUTO_seed3407.log
```

若只希望跑 500 步筛选，不自动续训：

```bash
bash experiments/spi_core_ring_adapter_v1/run.sh pilot
```

## 自动准入规则

最终 M4 必须同时满足：相对 P3 mIoU 至少 +0.50 pp、Damaged F1 至少 +1.0 pp、任一前景 IoU 不下降超过 0.50 pp、至少 10/14 事件不下降、分层配对 Bootstrap 下界大于 0。此外，`M3-M2` 和 `M4-M3` 都必须至少 +0.20 pp。

若 500 步未通过，控制器自动停止，不启动 3000 步任务。应读取 JSON 中第一个失败的因果检查，而不是盲目增加训练时长或模块。
