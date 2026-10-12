# Hierarchical Building Retrieval V1

该实验不修改 P3 或 fixed S2，只回答：两阶段检索是否能在排除查询事件后，为一个修改簇找到效用一致的建筑病例。

流程：

```text
查询修改簇所属伪建筑
  -> 建筑统计卡粗检索 Top-32
  -> 候选建筑内部修改簇精排 Top-8
  -> 修改方向过滤
  -> 每个来源事件最多2个邻居
  -> 预测 S2 修改净效用正/负
```

四个受控变体：

```text
B0 原始单向量修改簇检索
B1 建筑路由
B2 B1 + 相同修改方向
B3 B2 + 邻居事件多样性
```

本版只复用 D1 中面积至少33像素的修改簇，因此病例卡描述的是“建筑内已发生S2修改的区域”，还不是完整屋顶所有像素。只有 R0 通过，才值得重新生成包含稳定区、核心、边界和关键 Token 的完整建筑卡。

运行：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash experiments/spi_hierarchical_building_retrieval_v1/run.sh preflight

nohup env \
SEED=7341 \
D1_CACHE_ROOT=/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed7341 \
RESULT_ROOT=/root/data/BRIGHT-eval-results/SPI_HIERARCHICAL_BUILDING_RETRIEVAL_V1 \
COARSE_BUILDINGS=32 \
MEMORY_TOPK=8 \
MAXIMUM_PER_EVENT=2 \
bash experiments/spi_hierarchical_building_retrieval_v1/run.sh audit \
> /root/data/BRIGHT-train-logs/SPI_HBR_R0_seed7341.log 2>&1 &
```

查看：

```bash
tail -f /root/data/BRIGHT-train-logs/SPI_HBR_R0_seed7341.log
```

只有输出 `ADMIT_HIERARCHICAL_BUILDING_RETRIEVAL_R1` 才继续训练方向/效用条件化 Metric Encoder。若输出 `STOP_HBR_R0_NO_CROSS_EVENT_UTILITY_NEIGHBOURS`，不要调整 Top-K 后重复试验，应先生成完整建筑 Token 缓存。

R0停止后，先运行一次方向审计，判断弱信号是否集中在特定 P3->S2 修改方向：

```bash
env \
SEED=7341 \
D1_CACHE_ROOT=/root/data/BRIGHT-checkpoints/SPI_CLUSTER_DETAIL_D1_V1/seed7341 \
RESULT_ROOT=/root/data/BRIGHT-eval-results/SPI_HIERARCHICAL_BUILDING_RETRIEVAL_V1 \
COARSE_BUILDINGS=32 MEMORY_TOPK=8 MAXIMUM_PER_EVENT=2 \
bash experiments/spi_hierarchical_building_retrieval_v1/run.sh direction
```

该审计不会准许 Memory 直接接入推理；它只决定完整建筑 Token 缓存应该覆盖全部方向，还是只覆盖有稳定信号的方向。
