# DamageFormer Causal Research Chain V1

## 0. 目的

这个实验包不是新的“总模型”，而是把当前论文路线拆成可验证、可停止、可回退的因果研究链：

```text
A1
 -> A. Superpixel / Region Evidence Formation
 -> B. Tail Representation Diagnosis & Separation
 -> C. Conditional Damaged Specialist
 -> D. Single / Multi-Prototype Oracle
 -> E. Region-aware Instance Memory Oracle & Verifier
```

核心原则：**前一级没有证明成立，不用后一级掩盖问题。**

本包不会覆盖历史 V1/V2，也不会读取 Test。所有实验都以独立输出目录保存，Stage-B 以后只读取冻结的 V2 表征缓存，避免新实验反向改变超像素划分。

---

# 1. 与现有 V2 的关系

你现有的：

```text
experiments/damageformer_building_internal_change_partition_v2
```

继续负责 Stage-A：

```text
Stage-0 structure   1000 steps
Stage-1 change      3000 steps
Stage-2 feedback    5000 steps
```

本包**不重写、不覆盖**这套代码。它默认读取：

```text
/root/data/BRIGHT-checkpoints/
DAMAGEFORMER_BUILDING_INTERNAL_CHANGE_PARTITION_V2/
p_sc_residual_seed3407/
  stage1_change/latest.pth
  stage2_feedback/latest.pth
```

Stage-B/C/D/E 全部建立在 Stage-1 的 Region Evidence 上。

原因：Stage-1 已形成 Region，但还没有 Region→A1 feedback，最适合做表征审计；这样 Tail/Prototype/Memory 不会把 Stage-A 的 partition 重新拉坏。

---

# 2. 实验隔离

输出根目录：

```text
/root/data/BRIGHT-checkpoints/DAMAGEFORMER_CAUSAL_RESEARCH_CHAIN_V1/
/root/data/BRIGHT-eval-results/DAMAGEFORMER_CAUSAL_RESEARCH_CHAIN_V1/
/root/data/BRIGHT-train-logs/DAMAGEFORMER_CAUSAL_RESEARCH_CHAIN_V1/
```

目录结构：

```text
DAMAGEFORMER_CAUSAL_RESEARCH_CHAIN_V1/
├── cache/
│   └── p_sc_residual_seed3407/
│       ├── stage1_train.pt
│       ├── stage1_val.pt
│       └── stage2_val.pt
├── B_tail/
│   ├── ce_seed3407/
│   ├── balanced_ce_seed3407/
│   ├── blv_seed3407/
│   ├── seesaw_seed3407/
│   └── logit_adjust_seed3407/
├── B_multi_proto/
│   ├── k1_seed3407/
│   ├── k2_seed3407/
│   ├── k4_seed3407/
│   ├── best_selection.json
│   └── best_tail.pth
├── C_expert/
│   └── seed3407/
└── E_memory_verifier/
    └── seed3407/
```

每个实验只允许读取共享的**不可变 cache/checkpoint**；优化器、模型权重、结果目录互不共享。

---

# 3. Stage-A 缓存到底保存什么

`export_v2_cache.py` 从冻结的 V2 checkpoint 提取 predicted building 及 Region evidence。

每栋 predicted building 保存：

```text
GT majority label（只用于训练/审计，不进入 forward partition）
A1/V2 building-level I/D/X probability
A1 uncertainty
Region count
Peak-Mean evidence delta
所有 Region Tokens
所有 Region critical scores
```

Building feature 不是简单 RawMean，而是：

```text
Z_cache = concat(
    RegionToken Mean,
    RegionToken Std,
    Top-1 critical Region,
    Top-2 critical Region
)
```

`critical` 是 label-free 的：

```text
mean_abs(relation_peak - relation_mean)
+
mean_abs(relation_std)
```

所以没有用 GT 选择“Damage Region”。

为了使 Train cache 稳定，默认即使读取 `train_set.txt` 也使用 Dataset `type=val` 的非训练增强模式；这个 cache 只用于下游冻结表征诊断，不用于重新训练 V2 backbone。

---

# 4. Gate A：Superpixel / Region 是否成立

Stage-A 的正式训练仍由 V2 执行。

若你当前已经在跑 62000-step Stage-A 总实验，不要重复运行。若未来要从零完整重跑：

```bash
bash experiments/damageformer_causal_research_chain_v1/run_stage_a_all.sh
```

核心训练量：

```text
主 P-SC:         1000 + 3000 + 5000 = 9000 steps
P-S/P-C等消融:         3000 + 5000 = 8000 steps/variant
no-uncertainty:                 5000 steps only
```

Stage-A 完成后建立 cache：

```bash
bash experiments/damageformer_causal_research_chain_v1/run.sh cache_stage1
bash experiments/damageformer_causal_research_chain_v1/run.sh audit_region
```

如果 Stage-2 已完成：

```bash
bash experiments/damageformer_causal_research_chain_v1/run.sh cache_stage2_val
bash experiments/damageformer_causal_research_chain_v1/run.sh compare_feedback
```

`audit_region` 会输出：

```text
Region/building P10/P25/P50/P75/P90
Peak-Mean delta by Intact/Damaged/Destroyed
Peak-delta D-vs-I AUROC
```

`compare_feedback` 会输出：

```text
Base vs Stage-2 building metrics
Rescue
Harm
Net
Rescue/Harm by event
non-negative event count
```

---

# 5. Gate B0：Classifier Bias 还是 Representation Overlap

Stage-B 首先**冻结所有 V2 / Superpixel**，只训练一个小 linear probe。

五个诊断方法：

```text
CE
Balanced CE
BLV-adapted building-logit diagnostic
Seesaw classification component
Logit Adjustment
```

BLV 原论文是 pixel-level semantic segmentation；本包明确把它标记为 frozen building representation 上的 diagnostic adaptation，不把它伪装成原论文完全相同的 dense setting。

运行：

```bash
for M in ce balanced_ce blv seesaw logit_adjust; do
  bash experiments/damageformer_causal_research_chain_v1/run.sh tail_probe "$M"
done
```

默认每个：

```text
2000 steps
batch=512 cached buildings
Validation every 200 steps
selection = DI AUROC
```

这部分非常快，因为不再跑 RGB/SAR backbone。

判断：

```text
Balanced/BLV/Seesaw明显提高
 -> classifier / gradient bias 很重要

这些方法都几乎无效
 -> representation overlap 更可能是主瓶颈
```

---

# 6. Gate B1：Event-balanced Multi-Prototype Tail Separation

如果 Representation 仍需加强，训练：

```text
Projection head
+ K prototypes/class
+ class CE
+ Damaged-vs-Intact margin
+ prototype diversity loss
```

训练 batch 使用 class-event-balanced sampling，避免 prototype 只记住 Turkey/Hawaii/Libya 等事件。

分别跑：

```bash
for K in 1 2 4; do
  bash experiments/damageformer_causal_research_chain_v1/run.sh multi_proto "$K"
done
```

默认每个：

```text
3000 steps
Validation every 250
selection = DI AUROC
```

然后自动选 K：

```bash
python -m bda_benchmark.experiments.damageformer_causal_research_chain_v1.select_best_tail \
  --root /root/data/BRIGHT-checkpoints/DAMAGEFORMER_CAUSAL_RESEARCH_CHAIN_V1/B_multi_proto \
  --seed 3407
```

除分类指标外，还记录：

```text
Projected D/I separation ratio
D/I AUROC
Prototype assignment event entropy
每个 prototype 的 dominant-event fraction
```

如果某个 prototype 90% 都来自同一事件，应视作 Event Shortcut 风险。

---

# 7. Gate C：Conditional Damaged Specialist

先做 Oracle：

```bash
bash experiments/damageformer_causal_research_chain_v1/run.sh expert_oracle
```

候选默认：

```text
BasePred=Intact
OR
Top-2={Intact,Damaged}
```

Oracle 输出：

```text
候选池有多少 GT Damaged / Intact
候选中 Damaged prevalence
理论最大 Rescue
uncertainty-only Rescue/Harm curve
```

只有候选池存在足够 Damaged headroom 才值得训练 Expert。

训练：

```bash
bash experiments/damageformer_causal_research_chain_v1/run.sh expert_train
```

默认：

```text
3000 steps
class-balanced I/D candidates
Validation every 250
```

Expert 不输出四分类替代结果，只输出一个 I/D correction ranking score。

必须看：

```text
DI AUROC
Rescue/Harm @ 1%,2%,5%,10%,20% coverage
correction precision
```

---

# 8. Gate D：Single -> Multi Prototype Oracle

这一步不是训练新大模型，而是验证“类级历史知识是否已经足够”。

使用 Train-only support，Val-only query，并默认**排除 query 同事件的 Train support**：

```bash
for K in 1 2 4; do
  bash experiments/damageformer_causal_research_chain_v1/run.sh proto_oracle "$K"
done
```

默认使用 Stage-B 选出的 `best_tail.pth` projected embedding。

关注：

```text
proto_delta = S_D - S_I
BasePred=Intact 候选上的 DI AUROC
missed-Damaged delta mean
correct-Intact delta mean
```

如果：

```text
K=1 ≈ K=2 ≈ K=4
```

则没有证据支持复杂 Multi-Prototype。

如果 K>1 稳定提高，说明 Damaged 多模态假设得到支持。

---

# 9. Gate E：Region-aware Instance Memory Oracle

只有 Prototype 仍不足时才有必要保留 Full Memory。

真实 Memory：

```bash
bash experiments/damageformer_causal_research_chain_v1/run.sh memory_oracle none
```

默认：

```text
Coarse retrieval:
  Stage-B projected Building Evidence
  Top-50

Fine reranking:
  Superpixel Region Token Set
  top critical regions
  symmetric Chamfer-like cosine

Final:
  Top-20
  exclude same event
```

所以超像素没有被丢掉；Memory 的 fine matching 直接比较 Region Evidence Set。

必须同时跑三个负对照：

```bash
bash experiments/damageformer_causal_research_chain_v1/run.sh memory_oracle shuffle_labels
bash experiments/damageformer_causal_research_chain_v1/run.sh memory_oracle shuffle_events
bash experiments/damageformer_causal_research_chain_v1/run.sh memory_oracle wrong_regions
```

`wrong_regions` 保留 Building mean/std，但故意用**最低 critical region**替代 Top critical region，从而检验收益是不是来自真正局部 evidence。

Memory Oracle 输出：

```text
Recall@1/3/5/10/20 on Base-missed Damaged
Memory DI AUROC
Top-k Damaged fraction
Event diversity
Memory delta on missed-Damaged vs correct-Intact
```

只有：

```text
Memory AUROC > Prototype AUROC
且
Real Memory > shuffled-label / wrong-region controls
```

才有理由继续训练 verifier。

---

# 10. Gate E-final：Memory Utility Verifier

Verifier 训练**禁止使用 Validation**。

Train cache 会按 class-event 做确定性：

```text
80% fit-memory
20% calibration-query
```

用 calibration-query 训练 verifier；参数固定后，Validation 才只读评估。

运行：

```bash
bash experiments/damageformer_causal_research_chain_v1/run.sh memory_verifier
```

默认：

```text
3000 steps
Top-20 memory features
same-event exclusion
```

Verifier 输入：

```text
A1 uncertainty
base I/D margin
memory S_D-S_I
Top-k Damaged fraction
Top1 similarity
retrieved-event diversity
Top-k score dispersion
```

最终仍用：

```text
Rescue/Harm @ fixed coverage
```

验证是否“安全修正”，而不是看 retrieval accuracy。

---

# 11. 一键跑所有下游实验

**等主 P-SC Stage-1 完成即可开始 B/D/E 表征实验；Stage-2 只影响 feedback compare。**

安装：

```bash
cd /root/data
unzip -o damageformer_causal_research_chain_v1.zip
bash /root/data/damageformer_causal_research_chain_v1/install.sh \
  /root/data/BRIGHT-master/bda_benchmark
```

预检：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bright4090

bash experiments/damageformer_causal_research_chain_v1/run.sh preflight
```

后台一键运行所有 B-E：

```bash
mkdir -p /root/data/BRIGHT-train-logs/DAMAGEFORMER_CAUSAL_RESEARCH_CHAIN_V1

nohup bash \
experiments/damageformer_causal_research_chain_v1/run_all_downstream.sh \
> /root/data/BRIGHT-train-logs/DAMAGEFORMER_CAUSAL_RESEARCH_CHAIN_V1/all_downstream.log \
2>&1 &

echo $!
```

查看：

```bash
tail -f /root/data/BRIGHT-train-logs/DAMAGEFORMER_CAUSAL_RESEARCH_CHAIN_V1/all_downstream.log
```

如果你想连 Stage-A 也从头重跑：

```bash
RUN_STAGE_A=1 nohup bash \
experiments/damageformer_causal_research_chain_v1/run_all_downstream.sh \
> /root/data/BRIGHT-train-logs/DAMAGEFORMER_CAUSAL_RESEARCH_CHAIN_V1/all_A_to_E.log \
2>&1 &
```

**当前你已经在跑 Stage-A，因此不要设置 `RUN_STAGE_A=1`。**

---

# 12. 下游实验总训练量

缓存构建不计 optimizer steps。

```text
B0 Frozen tail diagnostics:
  5 methods x 2000 = 10,000 cached-feature steps

B1 Multi-Prototype:
  K=1/2/4 x 3000 = 9,000 cached-feature steps

C Damaged Specialist:
  3000 cached-feature steps

D Prototype Oracle:
  0 train steps

E Memory Oracle + controls:
  0 train steps

E Memory Verifier:
  3000 cached-feature steps

Total downstream optimizer steps = 25,000
```

这些不是 25k 次 DamageFormer forward；绝大多数只在缓存后的 Building features 上训练，因此比 Stage-A 便宜很多。

---

# 13. Go / No-Go 汇总

所有实验完成后：

```bash
python -m bda_benchmark.experiments.damageformer_causal_research_chain_v1.go_no_go \
  --eval_root /root/data/BRIGHT-eval-results/DAMAGEFORMER_CAUSAL_RESEARCH_CHAIN_V1 \
  --output /root/data/BRIGHT-eval-results/DAMAGEFORMER_CAUSAL_RESEARCH_CHAIN_V1/go_no_go.json
```

它是 advisory，不会替你“自动证明论文结论”。

核心逻辑：

```text
G1 Region不成立        -> 不做 Tail/Memory
G5 重平衡即可解决       -> 不必复杂化 representation
G6 MultiProto无增益    -> 保留 Single Prototype
G7 Expert Rescue<=Harm -> 砍 Expert
G8 Prototype已够       -> 不做 Full Memory
G9 Memory≈Prototype    -> 砍 Memory
G9 Real≈shuffled       -> Memory 因果控制失败
```

---

# 14. 论文指标层级

Stage-A 最终仍必须看原始 BRIGHT pixel metrics：

```text
Damaged IoU
Damaged F1
Damaged->Intact
valid mIoU
Event mean/worst
non-negative events/14
```

Stage-B/D/E 的 cache 是 building-level diagnostic，不应该把 building DI AUROC 冒充为最终 BDA mIoU。

Stage-B/C/D/E 通过以后，最终候选方案还必须重新接回像素模型做正式 BRIGHT Validation/Test。

Test 在所有方法/阈值/K/coverage 固定前禁止访问。
