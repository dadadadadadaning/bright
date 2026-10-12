# P3 + P5-C2 安全残差实验

## 要回答的问题

已有 E3 用固定滤波 SAR 和局部 Cross-Attention 替换了 P3 的原 SAR/融合路径，
因此虽然它保留了 P3 的建筑定位与 MICB/DSMIL，但初始化不再等于 P3。

本实验复用已经训练完成的 E3，并把它变成 P3 的候选修正支路：

```text
P3 logits + tanh(gate) * (E3 logits - P3 logits)
```

`gate` 从 0 开始，因此 step 0 严格输出 P3。训练只决定 E3 的修改是否值得写回。

## 实验顺序

```bash
bash experiments/p3_p5_safe_residual_v1/run.sh preflight
bash experiments/p3_p5_safe_residual_v1/run.sh smoke
bash experiments/p3_p5_safe_residual_v1/run.sh probe
bash experiments/p3_p5_safe_residual_v1/run.sh audit
bash experiments/p3_p5_safe_residual_v1/run.sh compare
```

服务器后台运行 5000 步：

```bash
mkdir -p /root/data/BRIGHT-train-logs/P3_P5_SAFE_RESIDUAL_V1
nohup bash experiments/p3_p5_safe_residual_v1/run.sh probe \
  > /root/data/BRIGHT-train-logs/P3_P5_SAFE_RESIDUAL_V1/probe5000.log 2>&1 &
echo "PID=$!"
```

也可以一次连续完成检查、20 步、5000 步、审计和比较：

```bash
nohup bash experiments/p3_p5_safe_residual_v1/run_pipeline.sh \
  > /root/data/BRIGHT-train-logs/P3_P5_SAFE_RESIDUAL_V1/pipeline.log 2>&1 &
echo "PIPELINE PID=$!"
```

## 准入条件

- 相对原始 P3 的 foreground mIoU 必须提高；
- Damaged F1 不下降；
- 至少 10/14 个事件不下降；
- paired bootstrap 95% CI 下界大于 0。

如果 5000 步没有通过，不扩展到 8000/50000 步。
