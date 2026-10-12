# SPI Superpixel Exemplar Memory V2

该实验不覆盖旧 Memory V1。核心变化：

1. 一个可靠超像素对应一条 Memory 记录，禁止同图同类平均；
2. 保留 Intact/Damaged/Destroyed 三分类，Background 由 P3 保持；
3. 复用冻结 P3 MICB 的 RGB/SAR/fused 专用投影；
4. RGB 查询局部 SAR 邻域，得到容错对齐特征、匹配熵和期望偏移；
5. Key 包含 mean/std/max/attention、核心—外环、SAR 分位数和区域几何；
6. Key Encoder 使用三类 CE、软标签 CE 与跨事件监督对比损失；
7. Memory 按类别和事件均衡，裸检索支持排除同图和同事件；
8. 正确标签与轮换错误标签在同一次审计中比较；
9. 安全回写保留 Background，边界衰减且 `gamma=0` 时严格复现父模型。

## 为什么先不训练回写头

旧 V1 的主要失败是正确 Memory 与打乱标签 Memory 几乎相同。因此 V2 先冻结在裸检索阶段。只有以下条件成立后，才允许训练 region-to-pixel 回写：

- 三类 Macro-F1 明显高于打乱标签；
- Damaged 与 Destroyed Recall 均不是零；
- 排除同事件后仍有有效检索；
- 正确 SLIC 后续明显优于规则网格记录；
- 邻居一致率和相似度 margin 不退化为零。

## 服务器运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash experiments/spi_superpixel_exemplar_memory_v2/run.sh preflight

mkdir -p /root/data/BRIGHT-train-logs/SPI_SUPERPIXEL_EXEMPLAR_MEMORY_V2
nohup bash experiments/spi_superpixel_exemplar_memory_v2/run.sh stage1 \
  > /root/data/BRIGHT-train-logs/SPI_SUPERPIXEL_EXEMPLAR_MEMORY_V2/stage1.log 2>&1 &
```

查看：

```bash
tail -f /root/data/BRIGHT-train-logs/SPI_SUPERPIXEL_EXEMPLAR_MEMORY_V2/stage1.log
```

Stage-1通过后才读取Validation并做一次只读裸检索：

```bash
bash experiments/spi_superpixel_exemplar_memory_v2/run.sh build_val_records
bash experiments/spi_superpixel_exemplar_memory_v2/run.sh audit_val
```

Test列表被代码拒绝。Validation不会更新Key Encoder或Memory。

## 结构消融

使用环境变量重新建立独立记录文件：

```bash
# 规则网格对照
REGION_MODE=grid TRAIN_RECORDS=/root/data/BRIGHT-checkpoints/SPI_SUPERPIXEL_EXEMPLAR_MEMORY_V2/train_grid_records.pt \
bash experiments/spi_superpixel_exemplar_memory_v2/run.sh build_train_records

# 无局部错位容忍对照
ALIGNMENT_RADIUS=0 TRAIN_RECORDS=/root/data/BRIGHT-checkpoints/SPI_SUPERPIXEL_EXEMPLAR_MEMORY_V2/train_slic_r0_records.pt \
bash experiments/spi_superpixel_exemplar_memory_v2/run.sh build_train_records
```

每套记录必须使用不同的 `METRIC_CKPT` 和 `MEMORY_PATH`，禁止混用。

