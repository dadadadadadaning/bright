# DamageFormer → P3 逐级消融 V1

## 目的

本实验回答“P3相对DamageFormer式像素路径，究竟是哪一部分产生作用”。四个变体使用完全相同的父检查点、训练/验证划分、随机种子、裁剪、有效批量和训练步数。

| 变体 | 新增内容 | 正确解释 |
|---|---|---|
| A0 | 无ROI、无建筑袋损失 | 共同协议下的DamageFormer兼容像素路径，不是原论文从头复现 |
| A1 | soft建筑ROI | A1-A0衡量ROI作用 |
| A2 | ROI+建筑袋三分类损失 | A2-A1衡量建筑级监督整体作用 |
| A3 | A2+Occurrence/Severity/Guard | A3-A2衡量层级辅助监督作用；这是完整P3训练目标 |

P3默认 `mil_residual_scale=0`，因此建筑袋头不会在推理时直接改写像素预测。A2/A3的差异来自训练监督对共享特征的影响。

## 公平性锁定

- 只读取 train/val，禁止test；
- 所有变体必须传入同一个训练前父检查点 `PARENT_CKPT`。严格的DamageFormer→P3消融应使用原始DamageFormer训练完成权重并设置`PARENT_KIND=damageformer`；如果使用S3V2父权重，则实验只能称为“P3直接阶段消融”；
- probe统一5000 optimizer steps、EBS=16、seed=3407；
- 主损失统一CE+Lovasz；
- 每500步完整验证，按valid mIoU保存最佳权重；
- 先跑5000步。只有有希望的模块才进入20000步，最后才考虑50000步。

## 服务器命令

如果尚无50,000步正式DamageFormer父权重，先生成：

```bash
bash experiments/p3_damageformer_ablation_v1/prepare_parent.sh preflight
mkdir -p /root/data/BRIGHT-train-logs/DAMAGEFORMER_PARENT_V1
nohup bash experiments/p3_damageformer_ablation_v1/prepare_parent.sh train \
  > /root/data/BRIGHT-train-logs/DAMAGEFORMER_PARENT_V1/train_final50000.log 2>&1 &
echo "parent PID=$!"
```

查看状态并在完成后验证身份：

```bash
bash experiments/p3_damageformer_ablation_v1/prepare_parent.sh status
bash experiments/p3_damageformer_ablation_v1/prepare_parent.sh verify
```

正式父权重固定输出到：

```text
/root/data/BRIGHT-checkpoints/DAMAGEFORMER_PARENT_V1/
D0_original_damageformer_final50000_seed3407/best_val_miou.pth
```

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

export PARENT_CKPT=/绝对路径/DamageFormer训练完成时的best_val_miou.pth
export PARENT_KIND=damageformer
bash experiments/p3_damageformer_ablation_v1/run.sh preflight
```

先逐个进行20步连通性检查：

```bash
for variant in a0 a1 a2 a3; do
  bash experiments/p3_damageformer_ablation_v1/run.sh smoke "$variant"
done
```

5000步按顺序连续训练：

```bash
mkdir -p /root/data/BRIGHT-train-logs/P3_DAMAGEFORMER_ABLATION_V1
nohup env \
  PARENT_CKPT="$PARENT_CKPT" \
  PARENT_KIND="$PARENT_KIND" \
  bash experiments/p3_damageformer_ablation_v1/run_sequential.sh probe a0 a1 a2 a3 \
  > /root/data/BRIGHT-train-logs/P3_DAMAGEFORMER_ABLATION_V1/launcher_probe5000.log \
  2>&1 &
echo "launcher PID=$!"
```

查看总调度日志以及当前模型日志：

```bash
tail -f /root/data/BRIGHT-train-logs/P3_DAMAGEFORMER_ABLATION_V1/sequence_probe_seed3407.log
tail -f /root/data/BRIGHT-train-logs/P3_DAMAGEFORMER_ABLATION_V1/a0_probe_seed3407.log
```

训练完成后会生成：

```text
/root/data/BRIGHT-checkpoints/P3_DAMAGEFORMER_ABLATION_V1/
├── a0_probe5000_seed3407/{config,parent_checkpoint,history,training_summary}.json
├── a1_probe5000_seed3407/...
├── a2_probe5000_seed3407/...
├── a3_probe5000_seed3407/...
├── ablation_probe5000_seed3407.json
├── ablation_probe5000_seed3407.csv
└── ablation_probe5000_seed3407.md
```

## 进入更长训练的条件

先看相邻差值，而不是只看A3总分：

- mIoU增益应为正；
- Damaged F1不能明显下降；
- 至少10/14事件不下降；
- 若5000步最佳点仍在最后一次验证，可将相关的候选和它的直接基线一起运行20000步，不能只延长候选。

例如A2有希望，必须同时继续A1与A2：

```bash
nohup env PARENT_CKPT="$PARENT_CKPT" \
  bash experiments/p3_damageformer_ablation_v1/run_sequential.sh confirm a1 a2 \
  > /root/data/BRIGHT-train-logs/P3_DAMAGEFORMER_ABLATION_V1/launcher_confirm20000.log 2>&1 &
```
