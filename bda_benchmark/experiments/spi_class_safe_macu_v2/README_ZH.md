# CS-MACU V2：类别安全的修改簇效用验证

## 动机

MACU-E1在meta_confirm上仍有`+0.3524pp` mIoU，但Damaged F1下降，且D0显示
`accepted_damaged_harm=156`，明显高于`rejected_damaged_benefit=36`。V2不扩大
骨干网络，只在同一组冻结修改簇特征上训练两个线性输出：

- Metric Benefit：S2修改是否提高train-frozen mIoU一阶效用；
- Damaged Harm：S2修改是否降低train-frozen Damaged F1一阶效用。

只有Benefit分数超过阈值且Harm分数低于阈值时才接受S2。

## 14折LOEO

每一折完整排除一个事件：

1. 使用meta_train中其余13个事件拟合标准化、两个线性头和效用梯度；
2. 使用meta_calibration中其余13个事件选择双阈值；
3. 只在被排除事件的train+calibration图像上产生预测；
4. 聚合14折的out-of-event预测，每张图像恰好预测一次。

meta_confirm不参与训练、阈值选择或LOEO评估。它已经被D0消费，只提供进入V2开发的
历史决策JSON。

## 运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash experiments/spi_class_safe_macu_v2/run.sh preflight

nohup env \
PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark \
LOEO_STEPS=1000 BATCH_SIZE=2048 BOOTSTRAP_ITERATIONS=10000 SEED=7341 \
bash experiments/spi_class_safe_macu_v2/run.sh auto \
> /root/data/BRIGHT-train-logs/SPI_CS_MACU_V2_LOEO_seed7341.log 2>&1 &
```

LOEO只有同时满足以下条件才冻结最终模型：相对fixed S2提升至少`0.30pp`、
Damaged F1不下降、至少`10/14` held-out事件不下降、bootstrap CI下界大于零，且
所有开发图像恰好获得一次事件外预测。
