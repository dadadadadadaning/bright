# Conditional Memory Screen V2

目的不是调大 Memory，而是验证病例卡是否包含 P3 之外的条件信息。

连续对照：

1. 原始 P3；
2. P3-only 同容量校准器；
3. P3 + 真实独立卡；
4. P3 + 打乱独立卡；
5. RGB 结构粗检索、SAR 异常重排、检索 P3 误差残差；
6. 保持邻居不变但打乱 Memory value。

只有真实特征超过 P3-only 和打乱特征，且真实检索超过打乱 value，才允许构建更昂贵的 CROMA/msGFM/SARMAE 建筑卡。

```bash
bash experiments/spi_conditional_memory_screen_v2/run.sh preflight
bash experiments/spi_conditional_memory_screen_v2/run.sh audit
```

本实验只访问既有 `meta_train` 和 `meta_calibration` 卡，不访问 `meta_confirm`、Val450 或 Test。

