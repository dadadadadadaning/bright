# CS-MACU V2.1：三臂LOEO

V2的双头交集策略在14折中全部回退fixed S2，但Harm Head跨事件AUROC很高。
V2.1在相同的事件隔离下独立比较：

- B0 fixed S2；
- B1 Benefit-only；
- B2 Harm-only（预注册主实验）；
- B3 Benefit+Harm（V2对照）。

每一折仍然只用其余13个事件训练和选择阈值。三个实验臂分别生成held-out预测，
不能根据held-out结果选择最好的一臂。最终准入只由B2决定。

每个实验臂记录候选总数、超过拒绝上限的数量、Damaged F1下降数量、mIoU不超过
fixed S2的数量、安全非平凡候选数量、有效候选数量及最终是否回退fixed S2。

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash experiments/spi_class_safe_macu_v21/run.sh preflight

nohup env \
PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LOEO_STEPS=1000 BATCH_SIZE=2048 \
BOOTSTRAP_ITERATIONS=10000 SEED=7341 \
bash experiments/spi_class_safe_macu_v21/run.sh loeo \
> /root/data/BRIGHT-train-logs/SPI_CS_MACU_V21_LOEO_seed7341.log 2>&1 &
```

B2准入门槛在运行前固定为：mIoU相对fixed S2至少`+0.20pp`、Damaged F1严格
提高、至少`10/14`事件mIoU不下降、至少`10/14`事件Damaged F1不下降、
bootstrap CI下界大于零且接受比例低于1。
