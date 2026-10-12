# 伪实例层级语义 Memory（PI-HSM v1）

## 实验回答什么

该实验不再让整栋建筑检索“相似建筑的 S2 效用”。它复用已经生成的 D1 修改簇缓存，把每栋 LocCC 视为父节点、其中的修改簇视为子节点，检验训练集标注库能否提高 Intact / Damaged / Destroyed 的跨事件识别。

每个查询由四部分组成：子簇特征、父建筑均值、子父差异和修改方向。Memory 每个类别容量相同，并记录事件、图像、面积与父内面积占比。正式检索强制排除同事件和同图条目。

## 与旧 PI-M0 的区别

- 旧方法整栋池化；本方法保留父建筑与内部修改簇两级结构。
- 旧方法预测 S2 效用；本方法检索训练标签所代表的语义类别。
- 旧方法容易检索到场景近重复；本方法按记录级 provenance 屏蔽同事件、同图。
- 旧 Memory 会被常见类占满；本方法 Intact、Damaged、Destroyed 等容量，并在每类内部均衡事件。
- Memory 不是硬投票覆盖预测，而是经标签/几何嵌入和 cross-attention 形成残差。

## 数据纪律

首轮只读取既有 `meta_train.pt` 和 `meta_calibration.pt`，不重新执行 P3，不读取 `meta_confirm`、Val450、官方实例或 Test。该版本是机制筛选实验；只有输出 `ADMIT_PI_HIERARCHICAL_MEMORY_FULL_MODEL_E1` 才允许接入主网络。

## 服务器运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash experiments/spi_pseudo_instance_hierarchical_memory_v1/run.sh preflight
```

启动训练和跨事件 calibration 筛选：

```bash
nohup env \
SEED=7341 \
D1_CACHE_ROOT=/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed7341 \
OUTPUT_ROOT=/root/data/BRIGHT-checkpoints/SPI_PI_HIERARCHICAL_MEMORY_V1/seed7341 \
MEMORY_HIDDEN=128 \
MEMORY_CAPACITY_PER_CLASS=2048 \
MEMORY_TOPK=4 \
PARAMETRIC_STEPS=1000 \
RETRIEVAL_STEPS=300 \
BATCH_SIZE=1024 \
bash experiments/spi_pseudo_instance_hierarchical_memory_v1/run.sh auto \
> /root/data/BRIGHT-train-logs/SPI_PI_HIERARCHICAL_MEMORY_V1_seed7341.log 2>&1 &
```

查看进度：

```bash
tail -f /root/data/BRIGHT-train-logs/SPI_PI_HIERARCHICAL_MEMORY_V1_seed7341.log
```

提取结论：

```bash
grep -E 'stage|memory_size|event_nonnegative|retrieval_gate|Decision|decision|Saved:' \
/root/data/BRIGHT-train-logs/SPI_PI_HIERARCHICAL_MEMORY_V1_seed7341.log
```

结果文件：

```text
/root/data/BRIGHT-checkpoints/SPI_PI_HIERARCHICAL_MEMORY_V1/seed7341/
├── pi_hierarchical_memory_frozen.pth
└── calibration_screen.json
```

## 如何读结果

`parametric` 是相同层级编码器但关闭 Memory；`memory` 是打开跨事件 Memory 后的结果。这样可以直接判断提升来自检索，而不是来自 MLP 容量。

准入要求全部满足：Memory macro-F1 至少提高 0.01，事件宏平均不下降，Damaged 和 Destroyed recall 都不下降，至少 10 个事件不下降，并且 Memory 非空。失败时保持 P3 + fixed S2，不训练更大的检索网络。

