# QSD-RCM：质量约束的结构—损毁残差病例库

本实验不增加外部预训练模型，也不修改 P3。它只回答一个问题：在成像质量兼容时，先按建筑结构找候选，再按损毁响应找邻居，能否检索到对 P3 错误具有相同“修正方向”的跨事件病例。

## 病例卡的三部分

- `Structure`：RGB 屋顶结构与建筑几何，只回答“是不是同类建筑”。
- `Damage`：SAR 物理纹理与建筑内相对异常，只回答“损毁反应是否相似”。
- `Quality`：成像质量，只用于剔除不可比较病例，不参与类别投票。

病例库按 `事件 × P3预测类别` 限量，而不是按 GT 类别筛选。查询时永远排除当前事件，避免同事件近重复样本形成捷径。

## 连续实验

1. `D0/E0`：从随机、平铺、结构、损毁、层级、质量约束、可观测路由和学习式度量八种检索中，检查谁能找到相同 P3 修正方向。
2. `E1`：固定邻居，只比较硬标签、软标签、P3 残差、打乱残差和零值，判断真正有用的是答案还是残差信息。
3. `E2`：依次加入相似度、邻居一致性、事件多样性、有效样本量和质量门控，检查保守弃权能否留下可靠病例。

只有最终方案同时超过原始 P3 和已有 P3-only 校准器，且明显超过打乱病例，才允许进入像素写回。否则 Memory 保持为诊断实验，不接入模型。

## 服务器运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash experiments/spi_qsd_residual_case_memory_v1/run.sh preflight

nohup env \
SEED=7341 \
MEMORY_HIDDEN=64 \
METRIC_STEPS=1000 \
BATCH_SIZE=512 \
MEMORY_CAPACITY=256 \
COARSE_K=64 \
MEMORY_TOPK=8 \
MAXIMUM_PER_EVENT=2 \
DEVICE=cuda:0 \
bash experiments/spi_qsd_residual_case_memory_v1/run.sh auto \
> /root/data/BRIGHT-train-logs/SPI_QSD_RCM_V1_seed7341.log 2>&1 &
```

查看进度：

```bash
tail -f /root/data/BRIGHT-train-logs/SPI_QSD_RCM_V1_seed7341.log
```

结果仅来自既有 `meta_train` 与 `meta_calibration` 建筑卡；不会访问 `meta_confirm`、Val450 或 test。

