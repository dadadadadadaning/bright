# MACU Failure Audit D0

该实验只读取已经生成的 `meta_train`、`meta_calibration` 和已经消耗过的
`meta_confirm` D1缓存，以及冻结的MACU线性模型。它不重新读取图像、不训练模型、
不选择阈值，也不访问Val450或Test。

主要回答：

1. Damaged F1下降主要来自“接受了有害S2簇”还是“拒绝了有益S2簇”；
2. 失败集中在哪些修改方向、面积和事件；
3. train/calibration/confirm的分数和AUROC是否发生漂移；
4. 同时要求mIoU正效用和Damaged非负效用的Class-Safe Oracle是否仍有足够空间。

运行：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
export PYTHONPATH=/root/data/BRIGHT-master:/root/data/BRIGHT-master/bda_benchmark:${PYTHONPATH:-}

bash experiments/spi_macu_failure_audit_d0_v1/run.sh preflight
bash experiments/spi_macu_failure_audit_d0_v1/run.sh audit
```

只有结果为 `ADMIT_CS_MACU_V2_DEVELOPMENT_ONLY` 时，才实现新的双目标V2。
该决定只允许开发V2，不能把已经使用过的meta_confirm再次当作独立确认集。
