# Selective RGID-E0（LocCC 选择性分裂审计）

本实验只改变 `fixed S2` 使用的区域标签图，不改变 P3 权重、建筑支持范围、S2 的纯度/边际阈值或软融合规则。

## 对照

- R0：原始 LocCC + fixed S2。
- R1：接受 RGID-G3 产生的全部有效分裂 + fixed S2。这里“全部”指全部有效提案；只有一个稳定中心、边界门拒绝或几何回滚失败的 LocCC 不可能被强制制造成两个可靠子区。
- R2：仅对高 split-risk LocCC 使用 RGID 分裂 + fixed S2。
- R3：随机选择与 R2 数量相同的可分裂 LocCC + fixed S2。
- R4：用 GT 对每个 LocCC 在 R0/R1 中二选一，仅作理论上限。

RGID 头只使用 `meta_train` 的语义标签产生伪实例训练；split-risk 阈值也只在 `meta_train` 选择。`meta_calibration` 只评估一次冻结阈值。程序不读取 `meta_confirm`、Val450、官方实例或 Test。

## 服务器运行

无论当前终端在哪个目录，都建议使用绝对路径：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
bash experiments/spi_selective_rgid_e0_v1/run.sh preflight
bash experiments/spi_selective_rgid_e0_v1/run.sh smoke
```

正式运行：

```bash
nohup env NUM_WORKERS=4 BATCH_SIZE=1 ACCUMULATION_STEPS=4 RGID_STEPS=500 SEED=7341 \
bash /root/data/BRIGHT-master/bda_benchmark/experiments/spi_selective_rgid_e0_v1/run.sh audit \
> /root/data/BRIGHT-train-logs/SPI_SELECTIVE_RGID_E0_V1_seed7341.log 2>&1 &
```

查看进度：

```bash
tail -f /root/data/BRIGHT-train-logs/SPI_SELECTIVE_RGID_E0_V1_seed7341.log
```

提取关键结果：

```bash
grep -E 'optimizer_step|Selected train-only|Selective RGID evaluate|gains_miou_pp|Decision:|Saved:' \
/root/data/BRIGHT-train-logs/SPI_SELECTIVE_RGID_E0_V1_seed7341.log
```

## 判定

R4 相对 R0 至少 `+0.20pp` 且至少 10/14 事件不下降，才说明“分不分”有足够理论空间。R2 还必须相对 R0 至少 `+0.10pp`、优于 R1 和 R3、bootstrap 下界大于 0、至少 10/14 事件不下降且 Damaged F1 不下降。

只有全部通过才输出 `ADMIT_SELECTIVE_RGID_META_CONFIRM`；否则保持 `P3 + LocCC + A6/fixed S2`。
