# DamageFormer Building-Internal Change-Aware Partition V2

## 1. 这版代码解决什么

这不是在 V1 上简单把 `stride=8` 改成 4，也不是继续把 `RGB structure + SAR relation` 直接拼接后输出 9-way association。

V2 把问题重新拆成三件事：

1. **Where：Region 只能在哪里形成？** 先由冻结 A1 的 `p_building` 生成 predicted building instances，Region candidate 被硬限制在同一栋建筑内。
2. **Change：灾后 SAR 什么时候可以改变 Region？** RGB structural logits 是基础分区；RGB-SAR change branch 只能通过一个有界、零初始化 residual 去修正 association logits。
3. **Evidence / Use：Region 里有什么，什么时候值得写回 A1？** Region token 不再只有 mean，而是保留 mean/std/soft-peak；Stage-2 才通过 A1 uncertainty + learned utility gate 保守写回。

核心公式：

```text
L_final = L_struct + alpha * DeltaL_change
alpha = alpha_max * tanh(change_gain) * sigmoid(gate)
Q = masked_softmax(L_final, same_building_candidates)
```

其中 `change_gain=0` 初始化，因此 Stage-1 刚开始时 SAR 没有权破坏 RGB structural partition，但可以在训练中逐渐获得有限的修边权。

## 2. 与 V1 的关键差异

V1：

```text
Global regular grid
+ soft p_building hint
+ concat(RGB structure, relation, edge, building)
-> 9-way association
-> mean token
-> building * confidence * residual
```

V2：

```text
A1 p_building
-> predicted building instances
-> adaptive per-building region seeds
-> same-building 3x3 candidates (hard constraint)

RGB structure -> L_struct
RGB/SAR aligned relation + confidence + entropy + A1 uncertainty
                -> DeltaL_change + alpha
L_struct + alpha * DeltaL_change -> Q

Q pooling:
RGB mean
SAR mean
Relation mean + std + soft-peak
confidence mean + entropy mean + area ratio
-> Region Token

Stage-1: weak building-level bag supervision
Stage-2: A1 uncertainty * utility gate * zero-init feedback residual
```

### 明确删除/改变的旧逻辑

- 不再 `relation *= confidence`。
- 不再 `aligned_sar *= confidence` 后再进入 token。
- 不再 `building * confidence * update` 作为最终写回 gate。
- 不再使用全图固定 `region_stride` 作为 Region 数的唯一尺度。
- 不再让 GT damage pixel reconstruction 假装提供建筑内部局部损毁真值。
- 不再用全图所有 Region 的统一平均质量做 balance。

## 3. Building-first candidate

`building_ops.py`：

```text
p_building
-> threshold
-> 4-connected components
-> remove tiny predicted instances
-> each building independently estimates target K
-> choose a compact local grid
-> seed snaps to nearest building pixel
-> each building pixel sees only same-building local 3x3 seeds
```

Region 0 永远是 NULL / background。

因此结构上保证：

```text
building(pixel) != building(token)
=> candidate invalid
=> Q_pk = 0
```

不是用 loss “希望不跨楼”，而是根本没有跨楼候选。

### Adaptive K

代码使用：

```text
desired_K ~= ceil(building_area / TARGET_REGION_AREA)
K <= MAX_REGIONS_PER_BUILDING
K <= floor(building_area / MIN_REGION_AREA)
```

默认：

```text
TARGET_REGION_AREA=16
MIN_REGION_AREA=4
MAX_REGIONS_PER_BUILDING=16
```

这些只是**安全启动值**。最终请根据 V1/V2 Audit 的 feature-level building area 分布调整，而不是把 16 当论文最终参数。

## 4. Structure-constrained + Change-guided association

### Structural head

输入：

```text
RGB structure (32)
RGB structural edge (1)
building-local x/y (2)
```

输出 `9` 个 structural logits。

另外有固定 compact distance bias，保证即便 change-only control 也不能做任意远距离 assignment。

### Change head

change encoder 输入：

```text
RGB alignment value
aligned SAR value
abs difference
product
alignment confidence
alignment entropy
```

注意 confidence 与 entropy 是**独立特征**，不再是乘法硬门。

随后 change head 还看到：

```text
relation feature
confidence
entropy
A1 uncertainty
local relation variation
```

输出：

```text
DeltaL_change: Bx9xHxW
alpha gate:     Bx1xHxW
```

最终：

```text
L = L_struct + alpha * DeltaL_change
```

### 为什么 change_gain 可以为负

`alpha = alpha_max * tanh(change_gain) * sigmoid(gate)`。

它允许全局 residual scale 带符号，以保持 `change_gain=0` 时**严格 identity 且梯度非零**。真正的修改幅度仍被 `alpha_max` 限制；默认 `0.30`。

## 5. Evidence-preserving token

V1 只有 mean，弱损毁很容易被平均。

V2 region token 包含：

```text
RGB mean
SAR mean
Relation mean
Relation std
Relation soft-peak
Alignment confidence mean
Alignment entropy mean
Region area / Building area
```

Soft-peak：

```text
score_p = sigmoid(evidence_logit_p)
weight_pk = Q_pk * exp(score_p / tau)
T_peak_k = normalized weighted sum
```

默认 `PEAK_TEMPERATURE=0.20`。

它不是 hard max，因此仍可微；但比纯 mean 更能保护少量高 evidence pixel。

## 6. Weak building-level task supervision

V2 **没有**使用建筑内部 pixel-level damage reconstruction。

GT 只在 `losses.py::weak_building_bag_loss()` 中用于：

```text
predicted building instance
-> collect GT semantic labels within that predicted building
-> majority Intact/Damaged/Destroyed
-> low-purity bag skip
-> region evidence logits aggregate
-> building-level CE
```

默认仅使用 label purity >= `0.70` 的 predicted building bag。

聚合可选：

```text
logmeanexp (default)
topk
```

这样 `L_bag` 从 Stage-1 第一步就可以沿：

```text
Building loss
-> Region logits
-> Region token
-> Q_pk
-> structural/change association
```

直接给 Partition 任务相关梯度，不依赖 final `feedback_gain` 先打开。

## 7. Building-local balance

V1 balance 是全图所有非空 Region 对一个 global mean。

V2：

```text
expected_mass_k = building_area / K_b
L_balance = mean((mass_k / expected_mass_k - 1)^2)
```

每栋建筑内部防 collapse，不强迫小楼和大楼的 Region 一样大。

## 8. 三阶段训练

### Stage-0：structure

目的：先学稳定的 building-internal structural partition。

训练：

```text
rgb_structure
structural_head
```

不训练 change，不写回 A1。

Loss：

```text
0.05 compact
+0.10 structure reconstruction
+0.01 building-local balance
```

建议 500–1000 step。

### Stage-1：change

加载 Stage-0 `latest.pth`。

训练：

```text
LCA matching/evidence path（不使用 LCA parent residual gain）
rgb structure / structural head
change encoder/head/gate
change_gain
evidence score
token encoder
region evidence head
```

不写回 A1。

Loss：

```text
0.05 compact
+0.05 structure
+0.005 local balance
+0.02 evidence preservation
+0.01 change residual regularizer
+1.0 weak building bag loss
```

建议 2000–3000 step。

Stage-1 的 pixel mIoU 理论上仍应保持 A1，因为 feedback 关闭。此阶段重点看：

```text
weak bag accuracy
association entropy
change alpha
effective regions/building
soft-peak vs mean
```

### Stage-2：feedback

加载 Stage-1 `best.pth`（按 weak bag accuracy 选）。

默认冻结 Region formation，仅训练：

```text
region_update
utility_gate
feedback_gain
```

最终：

```text
F' = F_A1 + tanh(feedback_gain)
             * hard_building
             * A1_uncertainty
             * utility_gate
             * update
```

这里不再乘 alignment confidence。

Stage-2 才重新用 pixel damage CE 作为主训练目标。

## 9. 核心消融

```text
p_s_struct_only
  Region geometry = RGB structure only
  Cross-modal evidence仍可进入 token，但不能修边

p_c_change_only
  去掉 learned RGB structural logits，只保留弱 compact bias + change refinement
  是接近 V1 风险的 control

p_sc_residual
  主方法：RGB structural base + bounded SAR residual refinement

p_sc_no_peak
  去 soft-peak

p_sc_mean_only
  回到 mean-only token

p_sc_no_bag
  去 weak task supervision

p_sc_no_uncertainty
  Stage-2 去 A1 uncertainty gate

p_sc_shuffled_sar
  SAR 空间错位 negative control
```

论文最核心三组：

```text
P-S  = p_s_struct_only
P-C  = p_c_change_only
P-SC = p_sc_residual
```

## 10. 安装

把 ZIP 上传到服务器，例如：

```bash
/root/data/damageformer_building_internal_change_partition_v2.zip
```

解压：

```bash
cd /root/data
unzip -o damageformer_building_internal_change_partition_v2.zip
```

安装：

```bash
bash /root/data/damageformer_building_internal_change_partition_v2/install.sh \
  /root/data/BRIGHT-master/bda_benchmark
```

## 11. 第一轮只跑这些

```bash
cd /root/data/BRIGHT-master/bda_benchmark
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bright4090

bash experiments/damageformer_building_internal_change_partition_v2/run.sh \
  preflight p_sc_residual
```

再：

```bash
bash experiments/damageformer_building_internal_change_partition_v2/run.sh \
  exact_replay p_sc_residual
```

必须先确认 exact replay 仍为 A1：

```text
valid_mIoU ~ 61.6866
mIoU4      ~ 70.4957
D-F1       ~ 61.0666
IoU I/D/X  ~ 76.39 / 43.95 / 64.72
```

然后才：

```bash
bash experiments/damageformer_building_internal_change_partition_v2/run.sh \
  smoke0 p_sc_residual
```

Smoke0 成功后：

```bash
bash experiments/damageformer_building_internal_change_partition_v2/run.sh \
  stage0 p_sc_residual
```

Stage-0 完成后：

```bash
bash experiments/damageformer_building_internal_change_partition_v2/run.sh \
  smoke1 p_sc_residual

bash experiments/damageformer_building_internal_change_partition_v2/run.sh \
  stage1 p_sc_residual
```

Stage-1 完成后先 Audit：

```bash
bash experiments/damageformer_building_internal_change_partition_v2/run.sh \
  audit1 p_sc_residual
```

Audit 合格再 Stage-2：

```bash
bash experiments/damageformer_building_internal_change_partition_v2/run.sh \
  smoke2 p_sc_residual

bash experiments/damageformer_building_internal_change_partition_v2/run.sh \
  stage2 p_sc_residual
```

## 12. 为什么没有直接自动 `pipeline`

虽然 `run.sh pipeline` 已提供，但第一轮不建议直接跑。因为：

1. predicted-building connected components 是 CPU 非微分步骤，先检查速度；
2. adaptive K 的默认面积参数必须经过 Audit；
3. Stage-1 的 weak bag coverage / purity 必须先确认；
4. Stage-2 只有在 Region representation 真正形成后才有意义。

## 13. Audit 正确解释

`audit.py` 不再把 BRIGHT semantic class consistency 称为“sub-building local damage localization”。

重点输出：

```text
building area distribution
nominal/effective regions per building
region area ratio
association entropy
change alpha/change gate
alignment confidence/entropy
soft-peak minus mean
weak building bag accuracy
hard candidate cross-building violations (必须为0)
```

## 14. 性能注意

V2 每个 forward 会在 feature resolution 对 predicted building mask 做 connected components，并构建 adaptive candidate metadata。当前实现优先保证逻辑正确和可审计，第一轮可能比 V1 慢。

如果模型机制验证有效，再优化为缓存/torch connected-component/矢量化 candidate builder；不要为了速度先牺牲实验因果性。
