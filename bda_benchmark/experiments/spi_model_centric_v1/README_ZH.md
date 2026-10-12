# Model-centric DamageFormer V1

该实验不修改原始 `DamageFormer.py` 和 P3 checkpoint。新模型继承锁定的
`DamageFormer_S8_MICB_DSMIL`，只添加零初始化并联插件。

## 变体

- `p0_parent`：父模型复现控制，只评估；
- `h1_hierarchy`：层级 Building/Affected/Severity 解码；
- `h2_rgb_structure`：Stage2 RGB Scharr/Gaussian/Strip；
- `h3_sar_frequency`：Stage2 SAR Haar/MAD专家；
- `h4_local_interaction`：RGB锚定局部SAR匹配；
- `h5_modality_specific`：RGB与SAR两个模态专用适配器；
- `h6_full`：三个模型创新联合，只能在单项实验通过后运行。

## 安全性质

1. 所有插件的残差增益从0开始；
2. 初始模型输出严格等于P3；
3. RGB结构增强不进入建筑定位支路；
4. 父模型全部冻结，只训练当前启用的插件；
5. 层级头有独立条件监督，避免零增益时内部头没有梯度；
6. 训练阶段不访问 meta_calibration、meta_confirm、Val450 或 Test。

## 推荐顺序

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash experiments/spi_model_centric_v1/run.sh preflight

env VARIANT=h1_hierarchy STEPS=2 \
  bash experiments/spi_model_centric_v1/run.sh smoke

nohup env VARIANT=h1_hierarchy STEPS=500 BATCH_SIZE=2 \
  ACCUMULATION_STEPS=8 NUM_WORKERS=4 \
  bash experiments/spi_model_centric_v1/run.sh auto \
  > /root/data/BRIGHT-train-logs/SPI_MODEL_H1_seed7341.log 2>&1 &
```

只有H1通过后，才分别运行：

```bash
for variant in h2_rgb_structure h3_sar_frequency h4_local_interaction; do
  env VARIANT="$variant" STEPS=500 \
    bash experiments/spi_model_centric_v1/run.sh auto
done
```

不要并行运行这些变体，也不要先运行 `h6_full`。500步只是筛选模块是否打开和
方向是否正确；正式结论需要延长训练并使用事件级评估与bootstrap。

## 监控

训练日志中的 `rgb_gain`、`sar_gain`、`interaction_gain`、`hierarchy_gain` 用于判断
模块是否真正打开。若启用模块在500步后增益仍接近0，说明其没有被优化器采用，
不应仅通过增加训练步数强行继续。
