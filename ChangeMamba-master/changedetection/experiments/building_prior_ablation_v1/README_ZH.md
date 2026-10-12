# ChangeMamba Building Prior Ablation V1

本实验只回答一个问题：在原始ChangeMamba-BRIGHT中加入软建筑先验，是否能改善损毁分类。

## Variant

- `cm0_original`：原始`ChangeMambaMMBDA`。
- `cm1_building_prior`：相同模型与训练设置，仅在`main_clf`之前加入来自RGB建筑分支的detach软先验。
- `cm1s_building_prior_stable`：与CM1相同的模型，但采用稳定微调协议：梯度累积得到EBS16、共享模型低学习率、建筑先验高学习率、warmup+cosine衰减及梯度裁剪。

CM1的transition零初始化，因此step 0与CM0函数一致。本实验不包含Native SAR Adapter、DiffFusion、超像素、Memory或层级损毁头。

## 保存策略

训练器设置`save_iteration_checkpoints = False`。每次验证只覆盖：

- `best_model.pth`：当前Validation四类mIoU最佳；
- `latest.pth`：最近一次验证状态，40k结束时即最后状态。

不会生成`iter_500.pth`一类的逐阶段权重。

## 推荐稳定微调协议

CM1-S必须与CM0使用同一个完整B0检查点初始化，不能从CM0最佳或旧CM1继续训练：

```bash
export INIT_CKPT=/path/to/the/same/audited_B0_best_model.pth
export TRAIN_BATCH_SIZE=2
export ACCUMULATION_STEPS=8
export BACKBONE_LR=1e-5
export NEW_MODULE_LR=1e-4
export WARMUP_STEPS=2000
export GRAD_CLIP_NORM=1.0

STEPS=40000 bash changedetection/experiments/building_prior_ablation_v1/run.sh \
  train cm1s_building_prior_stable
STEPS=40000 bash changedetection/experiments/building_prior_ablation_v1/run.sh \
  eval_val cm1s_building_prior_stable
SUMMARY_CM1_VARIANT=cm1s_building_prior_stable STEPS=40000 \
  bash changedetection/experiments/building_prior_ablation_v1/run.sh summarize
```

此时`STEPS`表示optimizer steps；每一步由8个micro-batch累积而成，有效Batch Size为`2×8=16`。

## 协议

- 默认40,000步；
- 每500步验证；
- 只构建Validation loader；
- Test在训练、选择和独立复现中均不访问；
- 两个variant使用同一VMamba encoder预训练权重、seed、损失与优化参数。

如果服务器没有encoder预训练权重，也可以让两个variant共同使用同一个、经过审计的完整B0检查点：

```bash
export INIT_CKPT=/path/to/original_b0/best_model.pth
```

CM0要求完整匹配；CM1只允许新增的`building_transition`缺失。不能使用B2 DiffFusion检查点冒充B0。

## 推荐运行

```bash
cd /root/data/BRIGHT-master/ChangeMamba-master
conda activate bright4090

bash changedetection/experiments/building_prior_ablation_v1/run.sh preflight

STEPS=40000 bash changedetection/experiments/building_prior_ablation_v1/run.sh \
  train cm0_original
STEPS=40000 bash changedetection/experiments/building_prior_ablation_v1/run.sh \
  eval_val cm0_original

STEPS=40000 bash changedetection/experiments/building_prior_ablation_v1/run.sh \
  train cm1_building_prior
STEPS=40000 bash changedetection/experiments/building_prior_ablation_v1/run.sh \
  eval_val cm1_building_prior

STEPS=40000 bash changedetection/experiments/building_prior_ablation_v1/run.sh summarize
```
