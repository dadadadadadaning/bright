# DamageFormer Late Decision Correction V1

目标：**完全停止 256D feature writeback**。A1/Stage1 只负责产生稳定 base prediction 与 Region evidence；后续模块只对 `A1=Intact` 的 predicted building 做选择性 `Intact -> Damaged` 最终决策修正。

## 科研约束

- Test 禁止访问。
- Stage1 必须固定为最终 `stage1_change/latest.pth`；Train/Val cache 的 source SHA 必须相同。
- Val 候选选择严禁使用 GT label/purity。GT 只允许用于训练标签、Oracle 上限和事后评估。
- 最终 pixel 指标不是 building F1 的替代品。每个 predicted building 缓存一个 4x4 pixel confusion contribution，因此 late override 可精确重算官方 pixel confusion，而无需缓存 450 张完整掩膜。
- 第一版只允许 `A1=Intact -> Damaged`；BG/Destroyed/已预测 Damaged 均不主动修改。

## 0. 安装目录

把整个目录复制到：

```text
/root/data/BRIGHT-master/bda_benchmark/experiments/damageformer_late_decision_correction_v1
```

然后：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090
bash experiments/damageformer_late_decision_correction_v1/run.sh preflight
```

## 1. E-cache：重建正式 Stage1-3000 全 predicted-building cache

```bash
FORCE_CACHE=1 bash experiments/damageformer_late_decision_correction_v1/run.sh cache
```

**验证什么：**
1. Train/Val 都来自同一个最终 Stage1 checkpoint；
2. Val cache 不再用 GT purity 删除候选，避免部署不可用的 GT candidate filter；
3. 保存每栋 predicted building 的 Region evidence 和 pixel confusion contribution。

必须看到 `CACHE IDENTITY PASS`。如果 source step 不是 3000 或 SHA 不同，立即停止后续实验。

## 2. E0：Building-level Late-Correction Oracle

```bash
bash experiments/damageformer_late_decision_correction_v1/run.sh oracle
cat /root/data/BRIGHT-eval-results/DAMAGEFORMER_LATE_DECISION_CORRECTION_V1/E0_oracle.json
```

同时输出：
- `intact_only`: 只把 selected building 内当前预测为 Intact 的像素改为 Damaged；
- `whole`: 整个 selected predicted building 改为 Damaged。

预算：1% / 2% / 5%。Oracle 用 GT 只做“理论上限选择”，candidate pool 本身仍只由 A1 base prediction 决定。

**验证什么：**不写回 feature、只在最终决策阶段修改，是否存在足够 pixel-mIoU / Damaged-IoU 上限。

建议 Go：2–5% 预算下 `intact_only` 能带来明显正的 pixel gain（实践上可先看 valid-mIoU +0.5pp 或 Damaged-IoU +1pp 量级）。若 Oracle 都很低，停止 building-level override，转 Region/cluster-local correction。

## 3. E1/E4：Specialist + Region 因果消融 C0-C3

```bash
for V in C0 C1 C2 C3; do
  bash experiments/damageformer_late_decision_correction_v1/run.sh specialist "$V"
done
```

输入定义：
- C0 = A1 3-class building probs + uncertainty
- C1 = C0 + RegionToken mean
- C2 = C0 + RegionToken mean/std
- C3 = C0 + mean/std + top1/top2 critical Region + Peak-Mean delta

训练时只用 Train 中 I/D 且 purity>=0.70 的监督；**Val 推理的 base-Intact candidate pool 不使用 GT**。

每个 summary 直接报告 1%/2%/5% coverage 的最终 pixel valid-mIoU、D-IoU、D-F1、D->I，以及 selected Destroyed/unknown 数量。

**验证什么：**
- C0：只靠 A1 calibration/hardness 是否已经足够；
- C1：building-global Region mean 是否有额外信息；
- C2：Region 分布/heterogeneity 是否有额外信息；
- C3：critical local Region / peak evidence 是否再提供独立收益。

关键因果比较不是“C3 自己好不好”，而是 **C3 > C1/C2**。否则不能把收益归因于局部 Region evidence。

## 4. E2：Single Prototype historical verifier

先选择 E1 最好的 Specialist（默认 C3）：

```bash
bash experiments/damageformer_late_decision_correction_v1/run.sh prototype C3
```

Prototype 使用 Train-only support，并对每个 Val query 排除同事件 support。输出三条相同 coverage 曲线：
- Specialist-only
- Prototype-only
- Specialist + Prototype (`z(spec)+z(proto_delta)`, lambda=1，不用 GT 调参)

**验证什么：**类级历史知识是否能减少 Specialist 的 Harm，而不是只是重新做一个分类器。

Go 条件：相同 1/2/5% coverage 下，组合应优于 Specialist-only，尤其 valid-mIoU/D-IoU 上升且 Intact/Destroyed 保护更好。若无增益，不进入复杂 Memory。

## 5. E3：Region-aware Memory verifier oracle/control

仅当 E2 仍有 headroom 时运行：

```bash
bash experiments/damageformer_late_decision_correction_v1/run.sh memory none
bash experiments/damageformer_late_decision_correction_v1/run.sh memory shuffle_labels
bash experiments/damageformer_late_decision_correction_v1/run.sh memory shuffle_events
bash experiments/damageformer_late_decision_correction_v1/run.sh memory wrong_regions
```

真实 Memory：building feature coarse top-50 + top critical Region set local reranking + top-20，排除同事件。

输出相同 coverage 的：
- Specialist-only
- Memory-only
- Specialist + Memory

**验证什么：**实例级跨事件历史病例是否提供 Single Prototype 之外的额外决策价值，以及收益是否真正来自 label/event/local-region semantics。

Go 条件必须同时满足：
1. Real Specialist+Memory > Specialist-only；
2. Real > shuffle_labels / shuffle_events / wrong_regions；
3. 相同 coverage 比较，而不是靠选得更少降低 Harm。

若不满足，Memory 不进入最终模型。

## 6. Utility Verifier 为什么暂时不训练

本包故意把 learned Utility Verifier 放在 E3 之后。先验证固定、无额外参数的 Specialist/Prototype/Memory score 是否有实际 pixel utility。只有 E3 Real Memory 明确超过 controls 后，才值得用 Train-only calibration split 训练 verifier；否则 verifier 很容易“掩盖前一级无效”。

## 7. 推荐最终路线

```text
A1 dense prediction
   -> Stage1 Region Evidence (frozen)
   -> Specialist trigger (only A1=Intact candidates)
   -> [optional] Prototype / Memory historical verification
   -> fixed-coverage / later utility gate
   -> Intact-only late override
   -> official pixel Val metrics
```

旧 V2 Stage2 full-feature feedback 保留为 W0 negative control，不再作为主线。
