# Safe AO Model V2

## V1失败后修正了什么

- RGB结构由Stage2语义特征改为输入级Scharr/LoG浅层结构；
- SAR频率由Stage2语义Haar改为输入级多尺度MAD、方向梯度和高通；
- 局部匹配不再修改完整四类融合特征，只产生Affected证据；
- 删除允许负数的层级混合系数；
- 最终残差必须非负，并同时加到Damaged和Destroyed；
- 定位支路和P3严重度差保持不变；
- 评估补充定位指标、事件结果、bootstrap和偏移鲁棒性。

## 消融变体

- `a1_rgb_shallow`：只用RGB输入级结构；
- `a2_sar_shallow`：只用SAR鲁棒纹理；
- `a3_same_position`：RGB/SAR同位证据；
- `a4_local_match`：局部匹配，不使用置信门；
- `a5_confident_local`：局部匹配和置信门，主实验。

## 实验顺序

先预检和主实验Smoke：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash experiments/spi_safe_affected_model_v2/run.sh preflight

env VARIANT=a5_confident_local STEPS=2 BATCH_SIZE=1 ACCUMULATION_STEPS=2 \
  NUM_WORKERS=1 bash experiments/spi_safe_affected_model_v2/run.sh smoke
```

随后逐个运行500步隔离pilot，不能并行：

```bash
for variant in a1_rgb_shallow a2_sar_shallow a3_same_position a4_local_match a5_confident_local; do
  env VARIANT="$variant" STEPS=500 BATCH_SIZE=2 ACCUMULATION_STEPS=8 \
    NUM_WORKERS=4 bash experiments/spi_safe_affected_model_v2/run.sh auto
done
```

只有 `a5_confident_local` 在零偏移评估通过后，才运行：

```bash
env VARIANT=a5_confident_local STEPS=500 \
  bash experiments/spi_safe_affected_model_v2/run.sh robustness
```

## 停止规则

内部评估同时要求：mIoU提升至少0.30pp、Damaged F1不下降、至少10/14事件
非负、bootstrap下界大于0、严重度logit差误差小于1e-5。失败后不访问
meta_confirm、Val450或Test，也不把多个失败变体组合起来。
