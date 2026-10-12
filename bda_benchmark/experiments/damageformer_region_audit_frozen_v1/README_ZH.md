# Region Audit + Frozen-A1 Diagnostic V1

本包严格按你当前决策：**先 Audit Region，再 Frozen-A1**。不加 Building Reasoning、Memory、Graph，不改 stride、loss、relation、gate。

## 目标

当前 joint 实验已经出现：A1 初始 `valid_mIoU=61.6866 / D-F1=61.0666`，9500 步为 `60.4794 / 59.2809`，但 `region_gain≈0.0403`。因此 Region 已明显参与写回，却没有形成稳定收益。下一步要区分：

- Region 数量是否太少（尺度问题）；
- Region 是否跨建筑/背景（建筑约束问题）；
- Region 是否混合 Intact/Damaged/Destroyed（目标问题）；
- Region 本身合理但写回伤害 A1（feedback 问题）；
- joint fine-tune 是否把 A1 带偏（训练协议问题）。

## 安装

```bash
cd /root/data
unzip -o damageformer_region_audit_frozen_v1.zip
bash /root/data/damageformer_region_audit_frozen_v1/install.sh /root/data/BRIGHT-master/bda_benchmark
cd /root/data/BRIGHT-master/bda_benchmark
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bright4090
```

## 1. Preflight

```bash
bash experiments/damageformer_region_audit_frozen_v1/run.sh preflight
```

## 2. 先检查 `target_instance_level`

```bash
bash experiments/damageformer_region_audit_frozen_v1/run.sh inspect_instances
```

脚本支持 raster instance map (`png/tif/npy`) 和常见 per-image polygon JSON。如果当前实例 JSON 格式不兼容，默认会停止，不会偷偷把语义连通域当正式实例。

如只想做临时诊断，明确允许 fallback：

```bash
export ALLOW_SEMANTIC_CC_FALLBACK=1
```

## 3. Audit 当前 Joint best（默认 best_val_miou.pth）

```bash
bash experiments/damageformer_region_audit_frozen_v1/run.sh audit_best
```

若 checkpoint 被移动：

```bash
export JOINT_BEST_CKPT=/实际路径/best_val_miou.pth
```

## 4. Audit late checkpoint

```bash
export JOINT_LATE_CKPT=/实际路径/step_009000.pth   # 示例
bash experiments/damageformer_region_audit_frozen_v1/run.sh audit_late
```

输出：

```text
building_region_audit.csv
region_level_audit.csv
damaged_evidence_coverage.json
instance_errors.json
summary.json
```

`summary.json` 重点看：

```text
Regions/Building mean/median/P10/P25/P50/P75/P90
1 Region、2 Region、3~4、>=5 比例
Region-building purity
Background crossing
Multi-building crossing
Damage entropy
Dominant damage purity
Damaged evidence Recall@0.5/@0.6/@0.7
```

还会按 Intact / Damaged / Destroyed 和 small / medium / large building 分组。

## 5. 比较 Joint best 与 late

```bash
bash experiments/damageformer_region_audit_frozen_v1/run.sh compare
```

如果 `region_gain` 增大后 purity 没升、damage entropy 没降、Damaged coverage 没升，就能解释：Region residual 只是越来越强地写回无关/错误区域。

## 6. Frozen-A1 前必须做 Step-0 Exact Replay

```bash
bash experiments/damageformer_region_audit_frozen_v1/run.sh exact_replay
```

应接近：

```text
valid_mIoU ≈ 61.6866
mIoU4      ≈ 70.4957
D-F1       ≈ 61.0666
Damaged IoU≈ 43.95
Destroyed  ≈ 64.72
region_gain = 0
```

如果不能复现，不要训练。

## 7. Frozen 20-step smoke

```bash
export FROZEN_OUTPUT_DIR=/root/data/BRIGHT-checkpoints/DAMAGEFORMER_FROZEN_A1_PARTITION_V1/c5_frozenA1_10000_seed3407
bash experiments/damageformer_region_audit_frozen_v1/run.sh frozen_smoke
```

Smoke 成功后如果要从 A1 干净重跑，删除该 smoke 输出目录。

## 8. Frozen-A1 正式 10000 steps

```bash
mkdir -p /root/data/BRIGHT-train-logs/DAMAGEFORMER_FROZEN_A1_PARTITION_V1
nohup bash experiments/damageformer_region_audit_frozen_v1/run.sh frozen10000 \
> /root/data/BRIGHT-train-logs/DAMAGEFORMER_FROZEN_A1_PARTITION_V1/frozen10000.log 2>&1 &

tail -f /root/data/BRIGHT-train-logs/DAMAGEFORMER_FROZEN_A1_PARTITION_V1/frozen10000.log
```

Frozen 实验直接复用已有 `damageformer_a1_cross_modal_superpixel_v1` 的 `c5_aligned_assoc`；该实现调用 `freeze_parent()`，只训练 `cmsp.*`。因此这个实验只改变训练协议，不改变模型结构。

设置：A1 全冻结；plugin LR `1e-4`；weight decay `5e-3`；batch 2；accum 8；BF16；seed 3407；stride 8；10000 optimizer steps；Val 每 500；永久 step checkpoint 每 5000。

## 结果决策

```text
Frozen > A1 + Audit合理
→ joint training 是主要问题，Partition保留。

Frozen ≈ A1 + Audit合理
→ Region有信息，但 writeback 没转化成增益。

Frozen < A1 + 每栋1~2 Region
→ 尺度问题，下一步改 stride/adaptive K。

Frozen < A1 + purity差/跨建筑高
→ 下一步做真正 building-internal candidates。

Frozen < A1 + Region多但 damage entropy高
→ 下一步才加入 task-specific L_damage-recon。

Audit很好但 Frozen仍下降
→ 下一步优先改 uncertainty-gated feedback。
```
