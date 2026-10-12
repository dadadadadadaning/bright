# LSM：局部结构语义 Memory（隔离实验）

这不是现有完整建筑卡 Memory 的续版。检索单位是只由灾前 RGB 结构生成的局部区域；P3 的类别图既不生成区域，也不决定哪些区域进入缓存。

## 三个阶段

- **缓存构建**：每 `IMAGES_PER_SHARD`（默认10）张图写出一个独立分片，随后释放该分片的 GPU/CPU临时对象；中断后会从最后一个完整分片继续，不会把所有局部区域长期堆积在内存中。
- **E0**：只读局部检索审计。比较 P3 局部概率、RGB结构、SAR结构、联合局部特征、打乱标签和随机查询；重点评估 `P3=Background/Intact, GT=Affected` 的 Recall@K。Memory 读取时固定为训练集内每个“事件×语义类”最多 256 个区域，避免局部库无限增大。
- **E1**：仅 E0 通过才允许训练。五个相同容量 Head 比较真实邻居、随机邻居、打乱邻居、零支持和无检索。主指标是 Affected 漏判 F1，不能仅看总体 F1。
- **E2**：本版本有意锁定。只有真实邻居相对全部对照表现出独立因果增益，才允许另建像素级整合实验；不访问 meta_confirm、Val450 或 Test。

## 运行顺序

```bash
bash experiments/spi_local_structural_memory_v1/run.sh preflight
bash experiments/spi_local_structural_memory_v1/run.sh smoke
bash experiments/spi_local_structural_memory_v1/run.sh build
bash experiments/spi_local_structural_memory_v1/run.sh e0
bash experiments/spi_local_structural_memory_v1/run.sh verify
```

若缓存构建被中断，以相同的 `CACHE_ROOT` 重跑 `build` 或 `auto` 即可续跑。首次建议使用新的缓存目录，并先做5张图的分片烟雾测试：

```bash
env CACHE_ROOT=/root/data/BRIGHT-checkpoints/SPI_LOCAL_STRUCTURAL_MEMORY_V1/seed7341_retry2 \
    IMAGES_PER_SHARD=5 SMOKE_IMAGES=10 \
    bash experiments/spi_local_structural_memory_v1/run.sh smoke
```

仅当 `verify` 显示 E0 admitted 时，再运行：

```bash
bash experiments/spi_local_structural_memory_v1/run.sh e1
bash experiments/spi_local_structural_memory_v1/run.sh verify
```
