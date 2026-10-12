# DamageFormer Evidence-Relation Memory V1

这是针对当前正式 Stage1-3000 cache 编写的实验包，目标是把后续实验固定为：

```text
A1 / Stage1
  ↓
Base-Intact candidate
  ├── Current evidence: C1 Specialist
  └── Historical evidence
       ├── Single Prototype
       ├── Global Case Memory
       ├── Local Region Memory
       └── Critical-weighted Local Region Memory
                ↓
        Agreement / Conflict
                ↓
        Utility Verifier
                ↓
        Intact -> Damaged only
```

依据当前实验，简单 `z(Specialist)+z(Prototype)` 不再作为正式融合方式。当前研究重点是先验证历史证据本身，再把“当前证据 vs 历史证据”的一致/冲突关系交给 Utility。

## 1. 数据协议

真实 cache 字段已经按当前项目适配：

```text
features            [N,512]
base_probs          [N,3]
base_pred           [N]
uncertainty         [N]
peak_mean_delta     [N]
region_offsets      [N+1]
region_tokens       [R,128]
region_critical     [R]
region_logits       [R,3]
base_pixel_cm       [4,4]
building_pixel_cm   [N,4,4]
```

Building-level 标签：

```text
-1 unknown/impure
 0 Intact
 1 Damaged
 2 Destroyed
```

Pixel confusion 标签：

```text
0 Background
1 Intact
2 Damaged
3 Destroyed
```

代码强制检查 Stage1 checkpoint SHA：

```text
0b76a887a16d4c1c5371cba1570a3a6c641435ef2c97494363323b73aa4239f7
```

并拒绝任何路径中包含 `test` 的 cache。

## 2. Train-Fit / Train-Cal

Official Train 按 image/name 且 event-stratified 确定性拆分：

```text
Train-Fit 80%
Train-Cal 20%
```

不是随机 building 拆分，因此同一影像中的建筑不会一部分进入 Fit、一部分进入 Cal。

Train-Fit：
- Specialist
- Prototype support
- Case Memory support

Train-Cal：
- Specialist checkpoint selection（AP/AUPRC）
- Agreement/Conflict 阈值
- Utility training

Val：
- 只做方法验证、消融、固定 2% coverage 排序评估

Test：
- 当前包完全禁止访问。

## 3. Specialist

自包含版本 C1：

```text
base_probs[3]
+ uncertainty[1]
+ RegionMean[128]
= 132D
```

训练只用 Train-Fit 中：

```text
I/D
purity >= 0.70
```

checkpoint 不再根据 Val 选择，而是根据 Train-Cal 的 `AP` 选择。

如果需要与旧 `damageformer_late_decision_correction_v1` 的历史 C1 **逐样本完全一致**，应额外导出旧 C1 分数并替换本包的 score 文件；本包 Specialist 是为后续证据关系实验提供的可复现自包含版本，不冒充旧 C1 完全复现。

## 4. E2.1 Agreement / Conflict Audit

高/低阈值只来自 Train-Cal 的 90% quantile：

```text
HH = Specialist high + Prototype high
HL = Specialist high + Prototype low
LH = Specialist low  + Prototype high
LL = Specialist low  + Prototype low
```

Val GT 只用于事后审计：

```text
Damaged prevalence
Destroyed count
unknown/impure
mean exact pixel utility
positive utility rate
mean building area
D->I rescue pixels
I->D harm pixels
X->D pixels
```

这一步不使用 Val GT 选择阈值。

## 5. E3-A Global Class-Conditional Memory

当前 Train Memory 严重不平衡，因此不做“所有历史病例混在一起 Top-K 多数投票”。

分别建立：

```text
Intact Memory
Damaged Memory
Destroyed Memory
```

每一类都：
1. 排除 query 同事件；
2. 用 RegionMean[128] 做 cosine coarse retrieval；
3. 各类独立 Top-50；
4. 用该类 Top-K 平均相似度得到 `S_I/S_D/S_X`。

主分数：

```text
Delta_mem = S_D - S_I
```

Destroyed 证据：

```text
X_veto = S_X - S_D
```

因此类别数量差异不会直接决定检索投票数量。

## 6. E3-B Local Region Memory

Global Top-50 只是候选病例。

对每个历史候选，再使用完整 Region Set：

```text
Query Regions [<=16,128]
vs
Historical Regions [<=16,128]
```

做对称 Chamfer / DN4-style matching：

```text
Query -> Memory 每个 Region 找最佳历史 Region
Memory -> Query 每个 Region 找最佳 Query Region
两方向平均
```

最终：

```text
Global score * 0.35
+ Local score * 0.65
```

每类重新取 Top-20，得到新的 `S_I/S_D/S_X`。

最关键因果比较：

```text
E3-B > E3-A
```

只有成立，才能说明“局部 Region/超像素病例”比单纯 Building Mean 有额外价值。

## 7. E3-C Critical-weighted Local Memory

局部匹配权重使用当前 cache 真实已有的：

```text
region_critical
region_logits
```

不是虚构 alignment entropy。

Region 权重：

```text
critical
+ 0.50 * (p_D - p_I)
- 0.25 * entropy(region_logits)
```

softmax temperature = 0.25。

最关键比较：

```text
E3-C > E3-B
```

只有成立，才能说“关键局部损毁状态相关证据”比等权 Region 有额外价值。

注意：BRIGHT 没有真实建筑内部损毁 mask，因此论文中不能把 Region 写成“真实局部损毁区域”，应写成：

> 建筑级损毁状态相关的局部证据区域。

## 8. Controls

提供：

```text
shuffle_labels
shuffle_events
wrong_regions
swapped_id
```

正式 Memory 如果真的依赖：
- 正确历史标签；
- 正确事件排除；
- 正确局部 Region 对应；

那么 Real Memory 应显著好于这些 control。

## 9. E4 Utility

只有 E3 通过后再运行。

Utility 输入：

```text
Specialist
Prototype delta / X veto
Global Memory delta / X veto
Local Memory delta / X veto
Weighted Local delta / X veto
base uncertainty
peak_mean_delta
base p_D-p_I
log building area
```

监督来自 Train-Cal 中每栋建筑的精确：

```text
DeltaU_i =
valid_mIoU(C_base - C_i + C_i_after_I_to_D)
- valid_mIoU(C_base)
```

Utility 使用：
- sign head：是否 `DeltaU > 0`
- delta head：回归标准化 `DeltaU`

最终固定 2% 排序用 delta head 预测的 pixel utility，不用 Val GT 训练。

## 10. 运行

解压到：

```text
/root/data/BRIGHT-master/bda_benchmark/experiments/
damageformer_evidence_relation_memory_v1/
```

然后：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash experiments/damageformer_evidence_relation_memory_v1/run.sh preflight
bash experiments/damageformer_evidence_relation_memory_v1/run.sh specialist

# E2.1 + E3
bash experiments/damageformer_evidence_relation_memory_v1/run.sh history none
bash experiments/damageformer_evidence_relation_memory_v1/run.sh relation
bash experiments/damageformer_evidence_relation_memory_v1/run.sh e3 none
```

或者：

```bash
bash experiments/damageformer_evidence_relation_memory_v1/run.sh all-e3
```

等 E3 real memory 明确通过后，再运行：

```bash
bash experiments/damageformer_evidence_relation_memory_v1/run.sh controls
bash experiments/damageformer_evidence_relation_memory_v1/run.sh utility
bash experiments/damageformer_evidence_relation_memory_v1/run.sh e4
```

## 11. 输出

```text
/root/data/BRIGHT-checkpoints/DAMAGEFORMER_EVIDENCE_RELATION_MEMORY_V1/
  specialist_c1/
    best.pth
    train_scores.pt
    val_scores.pt
  history_none.pt
  history_shuffle_labels.pt
  history_shuffle_events.pt
  history_wrong_regions.pt
  history_swapped_id.pt
  utility/
    best.pth
    val_utility_scores.pt

/root/data/BRIGHT-eval-results/DAMAGEFORMER_EVIDENCE_RELATION_MEMORY_V1/
  E2_1_relation_audit.json
  E3_none.json
  E3_shuffle_labels.json
  E3_shuffle_events.json
  E3_wrong_regions.json
  E3_swapped_id.json
  E4_utility.json
```

## 12. 如何判断是否继续

最重要的不是某个单一 mIoU 数字，而是因果顺序：

```text
Prototype
   ↓
Global Memory
   ↓  E3B > E3A ?
Local Region Memory
   ↓  E3C > E3B ?
Weighted Local Memory
   ↓  Real > controls ?
Utility
```

如果 `Local <= Global`，不能把收益归因于超像素/Region。

如果 `Weighted <= Local`，不能把收益归因于 critical Region weighting。

如果 Real 与 shuffled/wrong-region control 差不多，Memory 不能进入最终方法。

如果 E3 都不成立，就不要用 Utility 把无效 Memory “救起来”。

## 13. 当前 Prototype 的位置

当前已有严格验证中，2% 的 single global Prototype 是历史证据基线。已有实验表明它能带来约 +0.31pp valid mIoU，但 bootstrap CI 仍跨 0、正收益事件只有少数，因此本包的任务不是把 Prototype 宣布为最终成功，而是进一步检验：

> 具体历史病例和局部 Region 是否能把这种“有信号但跨事件不稳定”的 Prototype 提升成更稳定的历史证据。
