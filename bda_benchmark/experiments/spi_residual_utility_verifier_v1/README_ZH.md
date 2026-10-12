# Residual Utility Verifier V1

## 目标

本实验不再让 Memory 直接替代 P3 分类，而是回答：某个 Direct/Memory 修改建议是否值得接受。

```text
冻结P3 -> 建筑内SLIC候选 -> Direct/跨事件Memory建议
       -> 低容量Logistic Verifier -> 接受建议或回退P3
```

GT 只用于在 Train 上计算候选的 mIoU 效用标签，不进入任何推理特征。Val450 只评估，Test 禁止访问。

## 变体

- `m2_direct`：仅 Direct 建议和 P3/区域特征。
- `m2_union_basic`：Direct 与 Memory 都可提出候选，但不提供 Memory 置信度。这是候选覆盖对照。
- `m3_class_memory`：在 M2 Union 上加入 Memory 后验、margin、Top-k agreement、跨事件来源多样性等证据。
- M4 暂时锁定；只有 M3 明显超过 M2 Union，才建立 Residual Utility Memory。

## 服务器运行

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash experiments/spi_residual_utility_verifier_v1/run.sh preflight
```

先生成 Train 候选效用缓存。它会完整读取 2918 张 Train，但不会访问 Val/Test：

```bash
mkdir -p /root/data/BRIGHT-train-logs/SPI_RESIDUAL_UTILITY_VERIFIER_V1
nohup bash experiments/spi_residual_utility_verifier_v1/run.sh build_train \
  > /root/data/BRIGHT-train-logs/SPI_RESIDUAL_UTILITY_VERIFIER_V1/build_train.log 2>&1 &
```

查看进度：

```bash
tail -f /root/data/BRIGHT-train-logs/SPI_RESIDUAL_UTILITY_VERIFIER_V1/build_train.log
```

训练并验证最简单的 M2：

```bash
bash experiments/spi_residual_utility_verifier_v1/run.sh train m2_direct
bash experiments/spi_residual_utility_verifier_v1/run.sh build_val
bash experiments/spi_residual_utility_verifier_v1/run.sh evaluate m2_direct
```

M2合理后再比较“只有候选覆盖”和“真正加入Memory证据”：

```bash
bash experiments/spi_residual_utility_verifier_v1/run.sh stage_m3
python -m json.tool \
  /root/data/BRIGHT-eval-results/SPI_RESIDUAL_UTILITY_VERIFIER_V1/m3_vs_m2_union_admission.json
```

## 判定

M3 必须同时满足：

1. 相对主基线 `m2_direct` 提升至少 `+0.20pp`；
2. 相对 `m2_union_basic` 也应提高，才能排除“仅仅多了一批候选”的解释；
3. Damaged F1 不低于 M2；
4. 相对 P3 至少 10/14 事件不下降；
5. 自身绝对准入通过，且接受比例不超过 70%。

未通过时停止，不建立 M4，也不访问 Test。通过后才新增 Residual Utility Memory 与随机效用标签对照。
