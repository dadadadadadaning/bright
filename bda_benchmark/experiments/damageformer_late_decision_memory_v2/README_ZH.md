# DamageFormer Late Decision Memory V2

## 总体路线

```text
A1 / Stage1 frozen evidence
        ↓
Damaged Specialist
        ↓
Event-balanced Prototype
        ↓
Region-aware Cross-event Case Memory
        ↓
Utility Verifier
        ↓
Intact-only Late Override
```

目标不是继续改 A1 的中间特征，而是回答：

> A1 已经判为 Intact 的预测建筑组件中，哪些真正值得修改成 Damaged？

## Memory 保存什么

每个 Train predicted-building component 保存一个病例：

```text
global building key      [Dg]
region tokens            [K, Dr]
region valid mask         [K]
region metadata           [K, 4+]
  0 area_ratio
  1 alignment_confidence
  2 matching_entropy
  3 peak_minus_mean
GT label                  I / D / X
event id
building id
base probabilities        optional
```

不要把 hard superpixel mask 当主检索对象；hard mask 只用于可视化审计。

## Soft Superpixel 如何进入 Memory

已有 soft association：

```text
Q_pk = pixel p 属于 Region k 的概率
```

池化：

```text
z_k = sum_p Q_pk f_p / sum_p Q_pk
```

`superpixel_memory.py` 同时生成：

```text
Region Token
Area Ratio
Alignment Confidence
Matching Entropy
Peak-Mean
```

BRIGHT 没有建筑内部真实局部损毁像素标注，因此论文中建议称为：

> 建筑级损毁状态相关的局部证据区域

而不是“真实局部损毁区域”。

## 两级 Memory Retrieval

### Global coarse retrieval

```text
mean(valid Region Tokens)
      ↓ L2 normalize
exclude same event
      ↓
Top-50
```

### Local reranking

对 Top-50 的 Query/Memory Region set 做双向局部匹配：

```text
Query -> Memory max local matching
Memory -> Query max local matching
```

再根据：

```text
Peak-Mean
alignment confidence
matching entropy
area ratio
```

设置 Region reliability weight，最终保留 Top-20。

## Prototype

Prototype 先在事件内求类别原型，再在事件间平均，Query event 必须排除。

输出：

```text
proto_delta_DI = S_D - S_I
proto_X_veto   = S_X - S_D
```

## Memory Support

Top-20 不做简单多数投票，而输出：

```text
support_I
support_D
support_X
delta_DI
X_veto
agreement
event_diversity
effective_neighbors
score_margin
```

Memory 是 verifier，不直接替代 Specialist。

## Utility

Utility 不预测“是不是 Damaged”，而预测：

```text
执行 Intact -> Damaged 后有没有净收益？
```

如果现有 Late Decision cache 已保存每栋 predicted building 的 `4x4 pixel confusion contribution`：

```text
C_new = C_base - C_building + C_building_after_I_to_D
```

训练标签：

```text
DeltaUtility = valid_mIoU(C_new) - valid_mIoU(C_base)
utility_target = 1[DeltaUtility > 0]
```

Utility 只能用 Train-Cal 训练，不允许使用 Val/Test GT。

## 数据协议

```text
Official Train
├── Train-Fit
│   ├── Specialist
│   ├── Prototype support
│   └── Memory Bank
└── Train-Cal
    ├── Utility training
    └── threshold calibration

Official Val
└── method/ablation selection

Official Test
└── frozen final evaluation only
```

建议 Fit/Cal 按 image/tile 分组拆，不要随机 building 拆。

## 推荐实验

```text
E1 Specialist-only
E2 Specialist + Prototype
E3-A Specialist + Global Memory
E3-B + Local Region reranking
E3-C + Reliability weighting
E4 + Utility Verifier
```

始终固定比较：

```text
1% / 2% / 5% coverage
```

指标：

```text
valid_mIoU
mIoU4
Damaged IoU
Damaged F1
D->I
Destroyed IoU
Rescue / Harm
```

Memory controls：

```text
shuffle_labels
shuffle_events
wrong_regions
global_only
local_only
real_memory
```

## 安装

把整个目录复制到：

```text
/root/data/BRIGHT-master/bda_benchmark/experiments/damageformer_late_decision_memory_v2/
```

运行：

```bash
cd /root/data/BRIGHT-master/bda_benchmark
conda activate bright4090

bash experiments/damageformer_late_decision_memory_v2/run.sh preflight
```

## 第一步：检查当前 Stage1 cache

```bash
bash experiments/damageformer_late_decision_memory_v2/run.sh \
inspect-cache \
/root/data/BRIGHT-checkpoints/DAMAGEFORMER_LATE_DECISION_CORRECTION_V1/cache/p_sc_residual_seed3407/stage1_train_all_buildings.pt
```

当前服务器 `.pt` 的真实内部字段名必须以输出为准。

如果不是 canonical keys，只改：

```text
adapter_template.py
```

不要猜字段。

Canonical 输入至少应映射成：

```python
{
  "region_tokens": Tensor[N,K,D],
  "region_mask": Tensor[N,K],
  "region_meta": Tensor[N,K,M],
  "labels": Tensor[N],
  "event_ids": Tensor[N],
  "building_ids": Tensor[N],
  "base_probs": Tensor[N,3] | None,
  "source_sha256": str | None,
}
```

之后：

```bash
bash experiments/damageformer_late_decision_memory_v2/run.sh \
build-memory \
/path/canonical_train.pt \
/root/data/BRIGHT-checkpoints/DAMAGEFORMER_LATE_DECISION_MEMORY_V2/memory_train.pt
```

## 文件说明

```text
schema.py              MemoryBank 数据结构
specialist.py          C0-C3 Specialist feature + MLP
prototype.py           event-balanced Prototype
superpixel_memory.py   soft superpixel -> Region Memory token
local_matcher.py       reliability-weighted local set matching
memory_retriever.py    Top50 -> local rerank -> Top20
utility.py             exact pixel utility + UtilityVerifier
pipeline.py            最终单建筑 late-decision gate
build_memory.py        Memory 构建与 cache inspect
adapter_template.py    适配当前真实 Stage1 cache
train_utility.py       Train-Cal Utility 训练
evaluate.py            fixed coverage pixel evaluator
test_core.py           核心 smoke test
run.sh                 常用入口
```
