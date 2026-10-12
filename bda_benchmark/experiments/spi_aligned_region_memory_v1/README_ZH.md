# 对齐感知区域 Memory V1

## 1. 这套代码在验证什么

本实验不复现 CSSAD，也不声称复现尚未公开完整代码的 SRAC。它只验证适合当前
DamageFormer 任务的三段因果链：

```text
已训练的 ROI-LCA
    ↓ 先在建筑附近寻找 SAR 对应证据
建筑内部局部区域 Token
    ↓ 比较方格 / SLICO / 错误 SLICO / 学习区域
Train-only 跨事件 Memory
    ↓ 与 Direct Head、随机标签 Memory 比较
通过后才允许开发 Utility Verifier 和零残差写回
```

超像素只负责回答“哪些像素一起汇总”，不负责把整个区域强制改成同一类别。

## 2. 实验变量

| 变体 | 含义 | 单独证明什么 |
|---|---|---|
| R1 `r1_grid` | 规则小方格 | 普通局部汇总基线 |
| R2 `r2_slico` | 灾前 RGB 的 SLICO，并用预测建筑图切开 | 固定结构边界是否优于方格 |
| R3 `r3_shuffled` | 结构统计相似但空间位置错误 | 正确边界是否真的重要 |
| R4 `r4_learned` | RGB-only 的轻量可学习区域 | 学习分区是否优于固定 SLICO |

四种区域都使用同一种 Token：RGB、ROI-LCA 对齐后的 SAR、差异/共同响应、
Core/Ring。`P3/D0`最终概率、标签、事件名、配准质量和几何信息禁止进入 Key。
配准质量和 P3 状态只保留给以后 Verifier 使用。

## 3. 为什么暂时锁住写回

当前 V1 只做到区域和裸检索准入。运行 `writeback` 会主动失败。这不是缺代码，
而是实验保护：若区域结构或 Memory 还没有超过 Direct Head，就训练像素写回会把
失败来源再次混在一起。满足以下条件后再建立 V2：

- 最优区域相对方格 Direct Macro-F1 至少 `+0.10pp`；
- 正确区域优于打乱区域；
- Memory 相对 Direct 至少 `+0.20pp`；
- P3/D0 错误恢复至少 `25%`；
- 正确标签 Memory 相对随机标签至少 `+15pp`。

V2 才实现：低容量 Utility Verifier、拒绝全部候选的安全阈值、区域核心软广播、
零初始化 logit 残差及 Val450 像素 mIoU。

## 4. 运行顺序

先把服务器上的最佳 ROI-LCA 权重写清楚。以下示例假定 A3 最好；若汇总结果显示
A2/A4/A5 更好，只改 `ROI_VARIANT` 和 `ROI_CKPT`，不要偷偷挑选多个权重：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

export ROI_VARIANT=a3_conf3
export ROI_CKPT=/root/data/BRIGHT-checkpoints/DAMAGEFORMER_ROI_LCA_V1/a3_conf3_probe5000_seed3407/best_val_miou.pth

bash experiments/spi_aligned_region_memory_v1/run.sh preflight
bash experiments/spi_aligned_region_memory_v1/run.sh smoke
```

先运行不需要训练区域网络的三个固定对照。此阶段只训练相同容量的 Direct Head，
不会建立 Memory：

```bash
mkdir -p /root/data/BRIGHT-train-logs/SPI_ALIGNED_REGION_MEMORY_V1
nohup env ROI_VARIANT="$ROI_VARIANT" ROI_CKPT="$ROI_CKPT" \
  bash experiments/spi_aligned_region_memory_v1/run.sh stage_fixed \
  > /root/data/BRIGHT-train-logs/SPI_ALIGNED_REGION_MEMORY_V1/stage_fixed.log 2>&1 &
```

结果：

```bash
python -m json.tool \
  /root/data/BRIGHT-eval-results/SPI_ALIGNED_REGION_MEMORY_V1/fixed_region_admission_step5000.json
```

若固定 SLICO 无效，可以停止把 SLICO 当最终贡献，但仍可测试学习区域：

```bash
nohup env ROI_VARIANT="$ROI_VARIANT" ROI_CKPT="$ROI_CKPT" \
  bash experiments/spi_aligned_region_memory_v1/run.sh stage_learned \
  > /root/data/BRIGHT-train-logs/SPI_ALIGNED_REGION_MEMORY_V1/stage_learned.log 2>&1 &
```

从比较 JSON 中选择 **Direct Macro-F1 最好且通过结构对照** 的一个区域，例如
`r4_learned`，然后才建立 Memory：

```bash
bash experiments/spi_aligned_region_memory_v1/run.sh memory r4_learned

python -m json.tool \
  /root/data/BRIGHT-eval-results/SPI_ALIGNED_REGION_MEMORY_V1/r4_learned_memory_admission_step5000.json
```

## 5. 数据隔离

```text
Train：训练区域网络、训练 Key、写入 Memory
Validation：只查询 Memory、做准入判断
Test：V1 完全禁止访问
```

代码会拒绝以下行为：Validation 写入 Memory、GT/P3概率进入 Key、硬多数投票、
Test 列表参与任何阶段。
